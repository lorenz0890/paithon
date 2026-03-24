import traceback
from typing import Any, Callable

from ..context import build_call_summary, build_snapshot
from ..exceptions import CodeGenerationError, CodeRepairError, RuntimePolicyError, SafetyViolationError
from ..models import ImplementationRequest, RepairRequest
from ..provider import extract_python_source
from .structures import _ManagedFunction, _VariantState


class RuntimeExecutionMixin:
    def _ensure_initialized(self, state: _ManagedFunction, variant: _VariantState, args, kwargs) -> None:
        if variant.initialized:
            return
        with state.lock:
            if variant.initialized:
                return
            cached = self.cache.load(variant.key)
            if cached:
                cached = self._normalize_cache_payload(cached, cache_key=variant.key)
            if cached and "source" in cached:
                if self._allows_cache_payload(cached):
                    self._install_source(state, variant, cached["source"], persist=False)
                    self._audit(
                        "cache_loaded",
                        cache_key=variant.key,
                        qualname=state.template.__qualname__,
                        approval_status=self._approval_status(cached),
                    )
                    variant.initialized = True
                    return
                self._audit(
                    "cache_load_blocked",
                    cache_key=variant.key,
                    qualname=state.template.__qualname__,
                    approval_status=self._approval_status(cached),
                )
            if state.mode == "implement":
                if not self._allows_runtime_generation():
                    self._audit(
                        "generation_blocked",
                        cache_key=variant.key,
                        qualname=state.template.__qualname__,
                    )
                    raise RuntimePolicyError(
                        "operating_mode={0} forbids runtime generation for {1}".format(
                            self.config.operating_mode,
                            state.template.__qualname__,
                        )
                    )
                generated_source = self._generate(state, variant, args, kwargs)
                self._install_with_safety_repair(state, variant, generated_source, args, kwargs, persist=True)
            else:
                variant.current = state.template
                variant.source = state.template_source
            variant.initialized = True

    def _generate(self, state: _ManagedFunction, variant: _VariantState, args, kwargs) -> str:
        snapshot = build_snapshot(
            state.template,
            contract=state.contract,
            max_chars=self.config.max_value_chars,
            source=variant.source,
            args=args,
            kwargs=kwargs,
            state_fields=state.state_fields,
            serializer_registry=self.state_serializers,
            max_depth=self.config.max_state_depth,
            max_items=self.config.max_collection_items,
            max_mapping_items=self.config.max_mapping_items,
            max_sequence_items=self.config.max_sequence_items,
            max_set_items=self.config.max_set_items,
            redacted_field_paths=self.config.redacted_field_paths,
            redaction_placeholder=self.config.redaction_placeholder,
            redacted_field_names=self.config.redacted_field_names,
            redacted_field_patterns=self.config.redacted_field_patterns,
        )
        request = ImplementationRequest(snapshot=snapshot)
        try:
            generated = self.provider.implement_function(request, self.config.model)
            self._audit(
                "generated",
                cache_key=variant.key,
                qualname=state.template.__qualname__,
                model=self.config.model,
            )
            return generated
        except Exception as exc:
            raise CodeGenerationError("failed to generate {0}: {1}".format(state.template.__qualname__, exc))

    def _repair(self, state: _ManagedFunction, variant: _VariantState, error: Exception, args, kwargs) -> None:
        with state.lock:
            snapshot = build_snapshot(
                state.template,
                contract=state.contract,
                max_chars=self.config.max_value_chars,
                source=variant.source,
                args=args,
                kwargs=kwargs,
                state_fields=state.state_fields,
                serializer_registry=self.state_serializers,
                max_depth=self.config.max_state_depth,
                max_items=self.config.max_collection_items,
                max_mapping_items=self.config.max_mapping_items,
                max_sequence_items=self.config.max_sequence_items,
                max_set_items=self.config.max_set_items,
                redacted_field_paths=self.config.redacted_field_paths,
                redaction_placeholder=self.config.redaction_placeholder,
                redacted_field_names=self.config.redacted_field_names,
                redacted_field_patterns=self.config.redacted_field_patterns,
            )
            request = RepairRequest(
                snapshot=snapshot,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback_text=traceback.format_exc(),
                call_summary=build_call_summary(
                    state.template,
                    args,
                    kwargs,
                    self.config.max_value_chars,
                    serializer_registry=self.state_serializers,
                    max_depth=self.config.max_state_depth,
                    max_items=self.config.max_collection_items,
                    max_mapping_items=self.config.max_mapping_items,
                    max_sequence_items=self.config.max_sequence_items,
                    max_set_items=self.config.max_set_items,
                    redacted_field_paths=self.config.redacted_field_paths,
                    redaction_placeholder=self.config.redaction_placeholder,
                    redacted_field_names=self.config.redacted_field_names,
                    redacted_field_patterns=self.config.redacted_field_patterns,
                ),
            )
            try:
                repaired_source = self.provider.repair_function(request, self.config.model)
                self._audit(
                    "repaired",
                    cache_key=variant.key,
                    qualname=state.template.__qualname__,
                    model=self.config.model,
                    error_type=type(error).__name__,
                )
                self._install_with_safety_repair(state, variant, repaired_source, args, kwargs, persist=True)
            except Exception as exc:
                raise CodeRepairError("failed to repair {0}: {1}".format(state.template.__qualname__, exc))

    def _install_with_safety_repair(
        self,
        state: _ManagedFunction,
        variant: _VariantState,
        source: str,
        args,
        kwargs,
        *,
        persist: bool,
    ) -> None:
        current_source = source
        repair_attempts = 0
        while True:
            try:
                self._install_source(state, variant, current_source, persist=persist)
                return
            except SafetyViolationError as error:
                self._audit(
                    "safety_violation",
                    cache_key=variant.key,
                    qualname=state.template.__qualname__,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                if repair_attempts >= self.config.max_heal_attempts or not self._allows_runtime_healing():
                    if not self._allows_runtime_healing():
                        self._audit(
                            "safety_repair_blocked",
                            cache_key=variant.key,
                            qualname=state.template.__qualname__,
                            error_type=type(error).__name__,
                        )
                    raise
                repair_attempts += 1
                try:
                    current_source = self._repair_generated_source_for_safety(
                        state,
                        variant,
                        current_source,
                        error,
                        args,
                        kwargs,
                    )
                except CodeRepairError as repair_error:
                    self._reraise_original_error(error, repair_error)

    def _repair_generated_source_for_safety(
        self,
        state: _ManagedFunction,
        variant: _VariantState,
        source: str,
        error: SafetyViolationError,
        args,
        kwargs,
    ) -> str:
        snapshot = build_snapshot(
            state.template,
            contract=state.contract,
            max_chars=self.config.max_value_chars,
            source=extract_python_source(source),
            args=args,
            kwargs=kwargs,
            state_fields=state.state_fields,
            serializer_registry=self.state_serializers,
            max_depth=self.config.max_state_depth,
            max_items=self.config.max_collection_items,
            max_mapping_items=self.config.max_mapping_items,
            max_sequence_items=self.config.max_sequence_items,
            max_set_items=self.config.max_set_items,
            redacted_field_paths=self.config.redacted_field_paths,
            redaction_placeholder=self.config.redaction_placeholder,
            redacted_field_names=self.config.redacted_field_names,
            redacted_field_patterns=self.config.redacted_field_patterns,
        )
        request = RepairRequest(
            snapshot=snapshot,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_text=(
                "SafetyViolationError: {0}\n"
                "Generated source was rejected by static safety validation before execution."
            ).format(error),
            call_summary=build_call_summary(
                state.template,
                args,
                kwargs,
                self.config.max_value_chars,
                serializer_registry=self.state_serializers,
                max_depth=self.config.max_state_depth,
                max_items=self.config.max_collection_items,
                max_mapping_items=self.config.max_mapping_items,
                max_sequence_items=self.config.max_sequence_items,
                max_set_items=self.config.max_set_items,
                redacted_field_paths=self.config.redacted_field_paths,
                redaction_placeholder=self.config.redaction_placeholder,
                redacted_field_names=self.config.redacted_field_names,
                redacted_field_patterns=self.config.redacted_field_patterns,
            ),
        )
        try:
            repaired_source = self.provider.repair_function(request, self.config.model)
            self._audit(
                "safety_repaired",
                cache_key=variant.key,
                qualname=state.template.__qualname__,
                model=self.config.model,
                error_type=type(error).__name__,
            )
            return repaired_source
        except Exception as exc:
            raise CodeRepairError(
                "failed to repair unsafe generated source for {0}: {1}".format(state.template.__qualname__, exc)
            )

    def _install_source(self, state: _ManagedFunction, variant: _VariantState, source: str, persist: bool) -> None:
        normalized_source = extract_python_source(source)
        self._validate_generated_source(normalized_source, expected_name=state.template.__name__)
        self._probe_source_if_needed(normalized_source, state.template.__name__)
        replacement = self._compile_replacement(normalized_source, state.template)
        variant.current = replacement
        variant.source = normalized_source
        if persist:
            self.cache.save(
                variant.key,
                {
                    "mode": state.mode,
                    "module": state.template.__module__,
                    "qualname": state.template.__qualname__,
                    "source": normalized_source,
                    "model": self.config.model,
                    "state_fields": list(state.state_fields),
                    "mutable_state_fields": list(state.mutable_state_fields or ()),
                    "rollback_fields": list(state.rollback_fields),
                    "contract": state.contract,
                    "contract_revision": state.contract_revision,
                    "template_source": state.template_source,
                    "source_path": state.source_path,
                    "source_lineno": state.source_lineno,
                    "approval_status": self._default_approval_status(),
                    "generated_at": self._utcnow(),
                    "operating_mode": self.config.operating_mode,
                    "context": variant.context,
                },
            )
            self._audit(
                "cache_saved",
                cache_key=variant.key,
                qualname=state.template.__qualname__,
                approval_status=self._default_approval_status(),
            )

    def _compile_replacement(self, source: str, template: Callable[..., Any]) -> Callable[..., Any]:
        namespace = dict(template.__globals__)
        if template.__closure__:
            for name, cell in zip(template.__code__.co_freevars, template.__closure__):
                try:
                    namespace.setdefault(name, cell.cell_contents)
                except ValueError:
                    continue
        filename = "<paithon:{0}.{1}>".format(template.__module__, template.__qualname__)
        exec(compile(source, filename, "exec"), namespace, namespace)
        replacement = namespace.get(template.__name__)
        if not callable(replacement):
            raise CodeGenerationError("generated source did not define callable {0}".format(template.__name__))
        replacement.__module__ = template.__module__
        replacement.__qualname__ = template.__qualname__
        replacement.__doc__ = replacement.__doc__ or template.__doc__
        replacement.__annotations__ = dict(getattr(template, "__annotations__", {}))
        replacement.__defaults__ = template.__defaults__
        replacement.__kwdefaults__ = template.__kwdefaults__
        replacement.__dict__.update(getattr(template, "__dict__", {}))
        replacement.__paithon_source__ = source
        return replacement
