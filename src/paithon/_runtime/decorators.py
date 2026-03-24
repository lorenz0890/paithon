import functools
import inspect
import sys
from typing import Optional, Sequence, Tuple, Type

from ..context import is_placeholder_function
from ..exceptions import CodeRepairError
from .structures import _ManagedFunction


class RuntimeDecoratorMixin:
    def runtime_implemented(
        self,
        func=None,
        *,
        contract: Optional[str] = None,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
        heal_errors: bool = True,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        if func is None:
            return lambda wrapped: self.runtime_implemented(
                wrapped,
                contract=contract,
                state_fields=state_fields,
                mutable_state_fields=mutable_state_fields,
                rollback_fields=rollback_fields,
                rollback_on_failure=rollback_on_failure,
                strict_rollback=strict_rollback,
                cache_by_class=cache_by_class,
                contract_revision=contract_revision,
                heal_errors=heal_errors,
                heal_on=heal_on,
            )
        self._ensure_sync_function(func)
        resolved_contract = contract or inspect.getdoc(func) or ""
        if not resolved_contract.strip():
            raise ValueError("runtime_implemented requires a docstring or explicit contract")
        if not is_placeholder_function(func):
            raise ValueError("runtime_implemented expects a placeholder body")
        state = self._build_state(
            func,
            resolved_contract,
            mode="implement",
            state_fields=state_fields,
            mutable_state_fields=mutable_state_fields,
            rollback_fields=rollback_fields,
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            contract_revision=contract_revision,
        )
        return self._wrap(state, heal_errors=heal_errors, heal_on=heal_on)

    def self_healing(
        self,
        func=None,
        *,
        contract: Optional[str] = None,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
        max_attempts: Optional[int] = None,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        if func is None:
            return lambda wrapped: self.self_healing(
                wrapped,
                contract=contract,
                state_fields=state_fields,
                mutable_state_fields=mutable_state_fields,
                rollback_fields=rollback_fields,
                rollback_on_failure=rollback_on_failure,
                strict_rollback=strict_rollback,
                cache_by_class=cache_by_class,
                contract_revision=contract_revision,
                max_attempts=max_attempts,
                heal_on=heal_on,
            )
        self._ensure_sync_function(func)
        resolved_contract = contract or inspect.getdoc(func) or ""
        state = self._build_state(
            func,
            resolved_contract,
            mode="heal",
            state_fields=state_fields,
            mutable_state_fields=mutable_state_fields,
            rollback_fields=rollback_fields,
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            contract_revision=contract_revision,
        )
        return self._wrap(state, heal_errors=True, max_attempts=max_attempts, heal_on=heal_on)

    def schema_adapter(
        self,
        func=None,
        *,
        schema: Optional[str] = None,
        contract: Optional[str] = None,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
        heal_errors: bool = True,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        if func is None:
            return lambda wrapped: self.schema_adapter(
                wrapped,
                schema=schema,
                contract=contract,
                state_fields=state_fields,
                mutable_state_fields=mutable_state_fields,
                rollback_fields=rollback_fields,
                rollback_on_failure=rollback_on_failure,
                strict_rollback=strict_rollback,
                cache_by_class=cache_by_class,
                contract_revision=contract_revision,
                heal_errors=heal_errors,
                heal_on=heal_on,
            )
        base_contract = contract or inspect.getdoc(func) or ""
        parts = [
            "This function is a schema adapter / parser.",
            "Validate, normalize, and convert the input into the documented target shape.",
        ]
        if schema:
            parts.append("Schema:\n{0}".format(schema))
        if base_contract.strip():
            parts.append("Function contract:\n{0}".format(base_contract))
        combined_contract = "\n\n".join(parts)
        return self.runtime_implemented(
            func,
            contract=combined_contract,
            state_fields=state_fields,
            mutable_state_fields=mutable_state_fields,
            rollback_fields=rollback_fields,
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            contract_revision=contract_revision,
            heal_errors=heal_errors,
            heal_on=heal_on,
        )

    def response_adapter(
        self,
        func=None,
        *,
        contract: Optional[str] = None,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
        max_attempts: int = 2,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        default_heal_on = (
            heal_on if heal_on is not None else (KeyError, TypeError, ValueError, AttributeError, IndexError)
        )
        if func is None:
            return lambda wrapped: self.response_adapter(
                wrapped,
                contract=contract,
                state_fields=state_fields,
                mutable_state_fields=mutable_state_fields,
                rollback_fields=rollback_fields,
                rollback_on_failure=rollback_on_failure,
                strict_rollback=strict_rollback,
                cache_by_class=cache_by_class,
                contract_revision=contract_revision,
                max_attempts=max_attempts,
                heal_on=default_heal_on,
            )
        base_contract = contract or inspect.getdoc(func) or ""
        parts = [
            "This function adapts or parses external API responses.",
            "Prefer resilient handling of response-shape drift while preserving the documented return contract.",
        ]
        if base_contract.strip():
            parts.append("Function contract:\n{0}".format(base_contract))
        combined_contract = "\n\n".join(parts)
        return self.self_healing(
            func,
            contract=combined_contract,
            state_fields=state_fields,
            mutable_state_fields=mutable_state_fields,
            rollback_fields=rollback_fields,
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            contract_revision=contract_revision,
            max_attempts=max_attempts,
            heal_on=default_heal_on,
        )

    def create_function(
        self,
        name: str,
        signature: str,
        contract: str,
        *,
        namespace: Optional[dict] = None,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
        heal_errors: bool = True,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        globals_dict = namespace if namespace is not None else {"__name__": "__paithon_dynamic__"}
        template = self._make_placeholder_function(
            name=name,
            signature=signature,
            contract=contract,
            globals_dict=globals_dict,
        )
        return self.runtime_implemented(
            template,
            contract=contract,
            state_fields=state_fields,
            mutable_state_fields=mutable_state_fields,
            rollback_fields=rollback_fields,
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            contract_revision=contract_revision,
            heal_errors=heal_errors,
            heal_on=heal_on,
        )

    def polyfill(
        self,
        func=None,
        *,
        dependency: Optional[str] = None,
        platforms: Optional[Sequence[str]] = None,
        contract: Optional[str] = None,
        state_fields: Optional[Sequence[str]] = None,
        mutable_state_fields: Optional[Sequence[str]] = None,
        rollback_fields: Optional[Sequence[str]] = None,
        rollback_on_failure: bool = False,
        strict_rollback: bool = False,
        cache_by_class: bool = True,
        contract_revision: Optional[str] = None,
        heal_errors: bool = True,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        if func is None:
            return lambda wrapped: self.polyfill(
                wrapped,
                dependency=dependency,
                platforms=platforms,
                contract=contract,
                state_fields=state_fields,
                mutable_state_fields=mutable_state_fields,
                rollback_fields=rollback_fields,
                rollback_on_failure=rollback_on_failure,
                strict_rollback=strict_rollback,
                cache_by_class=cache_by_class,
                contract_revision=contract_revision,
                heal_errors=heal_errors,
                heal_on=heal_on,
            )
        if not self._needs_polyfill(dependency=dependency, platforms=platforms):
            return func
        resolved_contract = contract or inspect.getdoc(func) or ""
        if dependency:
            resolved_contract = "\n\n".join(
                part
                for part in [
                    "This function is a runtime polyfill because dependency '{0}' is unavailable.".format(dependency),
                    resolved_contract,
                ]
                if part.strip()
            )
        elif platforms:
            resolved_contract = "\n\n".join(
                part
                for part in [
                    "This function is a runtime polyfill because the current platform '{0}' is not in {1}.".format(
                        sys.platform,
                        tuple(platforms),
                    ),
                    resolved_contract,
                ]
                if part.strip()
            )
        template = func
        if not is_placeholder_function(func):
            template = self._make_placeholder_clone(func, resolved_contract)
        return self.runtime_implemented(
            template,
            contract=resolved_contract,
            state_fields=state_fields,
            mutable_state_fields=mutable_state_fields,
            rollback_fields=rollback_fields,
            rollback_on_failure=rollback_on_failure,
            strict_rollback=strict_rollback,
            cache_by_class=cache_by_class,
            contract_revision=contract_revision,
            heal_errors=heal_errors,
            heal_on=heal_on,
        )

    def _wrap(
        self,
        state: _ManagedFunction,
        heal_errors: bool,
        max_attempts: Optional[int] = None,
        heal_on: Optional[Sequence[Type[BaseException]]] = None,
    ):
        attempts = self.config.max_heal_attempts if max_attempts is None else max_attempts
        heal_on_tuple = tuple(heal_on) if heal_on is not None else None

        @functools.wraps(state.template)
        def wrapper(*args, **kwargs):
            variant = self._get_variant(state, args, kwargs)
            self._ensure_initialized(state, variant, args, kwargs)
            repair_count = 0
            while True:
                snapshot = self._capture_state_snapshot(state, args, kwargs)
                try:
                    result = variant.current(*args, **kwargs)
                    self._enforce_mutation_policy(state, snapshot, args, kwargs)
                    return result
                except Exception as error:
                    if state.rollback_on_failure:
                        rollback_diff = self._describe_changed_fields(snapshot) if snapshot is not None else {}
                        self._restore_state_snapshot(snapshot)
                        if rollback_diff:
                            self._audit(
                                "rollback_applied",
                                cache_key=variant.key,
                                qualname=state.template.__qualname__,
                                error_type=type(error).__name__,
                                state_diff=rollback_diff,
                            )
                    if (
                        not heal_errors
                        or repair_count >= attempts
                        or not self._should_heal(error, heal_on_tuple)
                        or not self._allows_runtime_healing()
                    ):
                        if heal_errors and self._should_heal(error, heal_on_tuple) and not self._allows_runtime_healing():
                            self._audit(
                                "repair_blocked",
                                cache_key=variant.key,
                                qualname=state.template.__qualname__,
                                error_type=type(error).__name__,
                            )
                        raise
                    repair_count += 1
                    try:
                        self._repair(state, variant, error, args, kwargs)
                    except CodeRepairError as repair_error:
                        self._reraise_original_error(error, repair_error)

        return wrapper

    @staticmethod
    def _reraise_original_error(error: Exception, repair_error: CodeRepairError) -> None:
        try:
            setattr(error, "__paithon_repair_error__", repair_error)
        except Exception:
            pass
        raise error.with_traceback(error.__traceback__) from repair_error

    @staticmethod
    def _should_heal(error: Exception, heal_on: Optional[Tuple[Type[BaseException], ...]]) -> bool:
        if heal_on is None:
            return True
        return isinstance(error, heal_on)
