import copy
import hashlib
import importlib.util
import inspect
import json
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ..context import bind_call_arguments, build_snapshot, get_function_source
from ..exceptions import StateMutationError, StateRollbackError, UnsupportedFunctionError
from ..snapshots import SNAPSHOT_MISSING, ValueSnapshot, deep_equal
from .structures import _ManagedFunction, _StateSnapshot, _VariantState


class RuntimeStateMixin:
    def _build_state(
        self,
        func: Callable[..., Any],
        contract: str,
        mode: str,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
    ) -> _ManagedFunction:
        source = get_function_source(func)
        declared_state_fields = tuple(state_fields or ())
        source_path, source_lineno = self._source_location(func)
        return _ManagedFunction(
            template=func,
            contract=contract,
            contract_revision=contract_revision,
            state_fields=declared_state_fields,
            mutable_state_fields=tuple(mutable_state_fields) if mutable_state_fields is not None else None,
            rollback_fields=tuple(rollback_fields or ()),
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            mode=mode,
            template_source=source,
            source_path=source_path,
            source_lineno=source_lineno,
        )

    def _build_key(
        self,
        func: Callable[..., Any],
        contract: str,
        source: str,
        mode: str,
        state_fields: Sequence[str],
        mutable_state_fields: Optional[Sequence[str]],
        rollback_fields: Sequence[str],
        strict_rollback: bool,
        context: Dict[str, Any],
        contract_revision: Optional[str],
    ) -> str:
        payload = {
            "contract": contract,
            "contract_revision": contract_revision,
            "context": context,
            "mode": mode,
            "mutable_state_fields": list(mutable_state_fields or ()),
            "module": func.__module__,
            "qualname": func.__qualname__,
            "rollback_fields": list(rollback_fields),
            "signature": str(inspect.signature(func)),
            "source": source,
            "state_fields": list(state_fields),
            "strict_rollback": strict_rollback,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _get_variant(self, state: _ManagedFunction, args, kwargs) -> _VariantState:
        context = self._build_variant_context(state, args, kwargs)
        context_token = json.dumps(context, sort_keys=True)
        with state.lock:
            variant = state.variants.get(context_token)
            if variant is None:
                key = self._build_key(
                    state.template,
                    state.contract,
                    state.template_source,
                    state.mode,
                    state.state_fields,
                    state.mutable_state_fields,
                    state.rollback_fields,
                    state.strict_rollback,
                    context,
                    state.contract_revision,
                )
                variant = _VariantState(
                    key=key,
                    context=context,
                    current=state.template,
                    source=state.template_source,
                )
                state.variants[context_token] = variant
            return variant

    def _build_variant_context(self, state: _ManagedFunction, args, kwargs) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if state.contract_revision is not None:
            context["contract_revision"] = state.contract_revision
        if state.mutable_state_fields is not None:
            context["mutable_state_fields"] = list(state.mutable_state_fields)
        if state.rollback_fields:
            context["rollback_fields"] = list(state.rollback_fields)
        if state.strict_rollback:
            context["strict_rollback"] = True
        if self.config.redacted_field_names or self.config.redacted_field_patterns:
            context["redaction"] = {
                "field_names": list(self.config.redacted_field_names),
                "patterns": list(self.config.redacted_field_patterns),
            }
        if self.config.redacted_field_paths:
            context.setdefault("redaction", {})
            context["redaction"]["paths"] = list(self.config.redacted_field_paths)
        if self.config.redaction_placeholder != "<redacted>":
            context.setdefault("redaction", {})
            context["redaction"]["placeholder"] = self.config.redaction_placeholder
        bound = bind_call_arguments(state.template, args, kwargs)
        if state.cache_by_class and bound is not None and bound.arguments:
            _, first_value = next(iter(bound.arguments.items()))
            cls = first_value if inspect.isclass(first_value) else getattr(first_value, "__class__", None)
            if cls is not None:
                context["class"] = {
                    "module": cls.__module__,
                    "qualname": cls.__qualname__,
                    "version": getattr(cls, "__paithon_version__", getattr(cls, "__version__", None)),
                }
        if state.state_fields:
            snapshot = build_snapshot(
                state.template,
                contract=state.contract,
                max_chars=self.config.max_value_chars,
                source=state.template_source,
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
            context["state_schema"] = snapshot.state_schema
        return context

    @staticmethod
    def _ensure_sync_function(func: Callable[..., Any]) -> None:
        if inspect.iscoroutinefunction(func):
            raise UnsupportedFunctionError("async functions are not supported yet")

    @staticmethod
    def _source_location(func: Callable[..., Any]) -> Tuple[Optional[str], Optional[int]]:
        try:
            return inspect.getsourcefile(func), inspect.getsourcelines(func)[1]
        except (OSError, IOError, TypeError):
            return None, None

    def _capture_state_snapshot(self, state: _ManagedFunction, args, kwargs) -> Optional[_StateSnapshot]:
        if state.mutable_state_fields is None and not state.rollback_on_failure:
            return None
        bound = bind_call_arguments(state.template, args, kwargs)
        if bound is None or not bound.arguments:
            return None
        _, target = next(iter(bound.arguments.items()))
        tracked_fields = self._snapshot_field_names(state, target)
        values = {}
        existed = {}
        for field_name in tracked_fields:
            exists = hasattr(target, field_name)
            existed[field_name] = exists
            if not exists:
                continue
            values[field_name] = self._capture_field_value(state, target, field_name)
        cleanup_extras = bool(state.mutable_state_fields is not None)
        if state.rollback_on_failure and hasattr(target, "__dict__") and not state.rollback_fields:
            cleanup_extras = True
        return _StateSnapshot(target=target, values=values, existed=existed, cleanup_extras=cleanup_extras)

    def _snapshot_field_names(self, state: _ManagedFunction, target: Any) -> Sequence[str]:
        tracked_fields = set(state.state_fields)
        tracked_fields.update(state.rollback_fields)
        if state.mutable_state_fields is not None:
            tracked_fields.update(state.mutable_state_fields)
            if hasattr(target, "__dict__"):
                tracked_fields.update(target.__dict__.keys())
        elif state.rollback_on_failure and hasattr(target, "__dict__") and not state.rollback_fields:
            tracked_fields.update(target.__dict__.keys())
        return tuple(sorted(tracked_fields))

    def _capture_field_value(self, state: _ManagedFunction, target: Any, field_name: str) -> ValueSnapshot:
        value = getattr(target, field_name)
        try:
            return self.snapshot_strategies.capture(value)
        except Exception as exc:
            if state.strict_rollback:
                raise StateRollbackError(
                    "failed to snapshot field {0} on {1}: {2}".format(
                        field_name,
                        state.template.__qualname__,
                        exc,
                    )
                )
            try:
                cloned = copy.deepcopy(value)
            except Exception:
                return ValueSnapshot(
                    strategy_name="unsupported",
                    original_value=value,
                    restore_payload=None,
                    compare_payload=None,
                    restore_fn=lambda current, snapshot: current,
                    compare_fn=lambda current, snapshot: True,
                    supported=False,
                )
            return ValueSnapshot(
                strategy_name="copy-fallback",
                original_value=value,
                restore_payload=cloned,
                compare_payload=copy.deepcopy(cloned),
                restore_fn=lambda current, snapshot: copy.deepcopy(snapshot.restore_payload),
                compare_fn=lambda current, snapshot: deep_equal(current, snapshot.compare_payload),
            )

    def _restore_state_snapshot(self, snapshot: Optional[_StateSnapshot]) -> None:
        if snapshot is None:
            return
        if snapshot.cleanup_extras and hasattr(snapshot.target, "__dict__"):
            extra_fields = set(snapshot.target.__dict__.keys()).difference(snapshot.existed.keys())
            for field_name in extra_fields:
                delattr(snapshot.target, field_name)
        for field_name in sorted(snapshot.existed.keys()):
            existed_before = snapshot.existed[field_name]
            if not existed_before:
                if hasattr(snapshot.target, field_name):
                    delattr(snapshot.target, field_name)
                continue
            field_snapshot = snapshot.values.get(field_name)
            if field_snapshot is None:
                continue
            current_value = getattr(snapshot.target, field_name, SNAPSHOT_MISSING)
            restored_value = self.snapshot_strategies.restore(current_value, field_snapshot)
            setattr(snapshot.target, field_name, restored_value)

    def _enforce_mutation_policy(self, state: _ManagedFunction, snapshot: Optional[_StateSnapshot], args, kwargs) -> None:
        if snapshot is None or state.mutable_state_fields is None:
            return
        changed_fields = self._detect_changed_fields(snapshot)
        disallowed = sorted(changed_fields.difference(state.mutable_state_fields))
        if disallowed:
            diff = self._describe_changed_fields(snapshot, only_fields=disallowed)
            self._restore_state_snapshot(snapshot)
            self._audit(
                "state_mutation_blocked",
                qualname=state.template.__qualname__,
                changed_fields=disallowed,
                state_diff=diff,
            )
            raise StateMutationError(
                "{0} modified disallowed state fields: {1}. diff={2}".format(
                    state.template.__qualname__,
                    ", ".join(disallowed),
                    diff,
                )
            )

    def _detect_changed_fields(self, snapshot: _StateSnapshot) -> set:
        changed = set()
        field_names = set(snapshot.existed.keys())
        if hasattr(snapshot.target, "__dict__"):
            field_names.update(snapshot.target.__dict__.keys())
        for key in field_names:
            existed_before = snapshot.existed.get(key, False)
            exists_now = hasattr(snapshot.target, key)
            if existed_before != exists_now:
                changed.add(key)
                continue
            if not existed_before:
                continue
            old_value = snapshot.values.get(key)
            if old_value is None:
                continue
            new_value = getattr(snapshot.target, key, SNAPSHOT_MISSING)
            if not old_value.supported:
                continue
            if not self.snapshot_strategies.matches(new_value, old_value):
                changed.add(key)
        return changed

    def _describe_changed_fields(self, snapshot: _StateSnapshot, *, only_fields: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        field_names = set(only_fields or snapshot.existed.keys())
        if only_fields is None and hasattr(snapshot.target, "__dict__"):
            field_names.update(snapshot.target.__dict__.keys())
        changes = {}
        for key in sorted(field_names):
            existed_before = snapshot.existed.get(key, False)
            exists_now = hasattr(snapshot.target, key)
            before_value = snapshot.values.get(key)
            if existed_before != exists_now:
                changes[key] = {
                    "before": "<missing>" if not existed_before else "<present>",
                    "after": "<missing>" if not exists_now else "<present>",
                }
                continue
            if not existed_before or before_value is None or not before_value.supported:
                continue
            current_value = getattr(snapshot.target, key, SNAPSHOT_MISSING)
            diff = self.snapshot_strategies.diff(
                current_value,
                before_value,
                max_items=self.config.max_collection_items,
                max_chars=self.config.max_value_chars,
                max_depth=self.config.max_state_depth,
            )
            if diff:
                changes[key] = diff
        return changes

    @staticmethod
    def _needs_polyfill(dependency: Optional[str], platforms: Optional[Sequence[str]]) -> bool:
        if dependency and importlib.util.find_spec(dependency) is None:
            return True
        if platforms and sys.platform not in tuple(platforms):
            return True
        return False
