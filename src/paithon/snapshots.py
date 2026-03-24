import copy
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple


SNAPSHOT_MISSING = object()


@dataclass
class ValueSnapshot:
    strategy_name: str
    original_value: Any
    restore_payload: Any
    compare_payload: Any
    restore_fn: Callable[[Any, "ValueSnapshot"], Any]
    compare_fn: Callable[[Any, "ValueSnapshot"], bool]
    supported: bool = True


class SnapshotStrategyRegistry:
    def __init__(self):
        self._entries: List[
            Tuple[
                Callable[[Any], bool],
                Callable[[Any], Any],
                Callable[[Any, ValueSnapshot], Any],
                Callable[[Any, ValueSnapshot], bool],
                str,
            ]
        ] = []
        self.register(
            self._is_keras_weights_like,
            self._capture_keras_weights_like,
            self._restore_keras_weights_like,
            self._compare_keras_weights_like,
            name="keras_weights",
            first=True,
        )
        self.register(
            self._is_state_accessor_like,
            self._capture_state_accessor_like,
            self._restore_state_accessor_like,
            self._compare_state_accessor_like,
            name="state_accessor",
            first=True,
        )
        self.register(
            self._is_pandas_dataframe_like,
            self._capture_pandas_dataframe_like,
            self._restore_pandas_dataframe_like,
            self._compare_pandas_dataframe_like,
            name="pandas_dataframe",
            first=True,
        )
        self.register(
            self._is_pandas_series_like,
            self._capture_pandas_series_like,
            self._restore_pandas_series_like,
            self._compare_pandas_series_like,
            name="pandas_series",
            first=True,
        )
        self.register(
            self._is_torch_tensor_like,
            self._capture_torch_tensor_like,
            self._restore_torch_tensor_like,
            self._compare_torch_tensor_like,
            name="torch_tensor",
            first=True,
        )
        self.register(
            self._is_numpy_array_like,
            self._capture_numpy_like,
            self._restore_numpy_like,
            self._compare_numpy_like,
            name="numpy_array",
            first=True,
        )
        self.register(
            self._is_state_dict_like,
            self._capture_state_dict_like,
            self._restore_state_dict_like,
            self._compare_state_dict_like,
            name="state_dict",
            first=True,
        )
        self.register(
            lambda value: True,
            self._capture_copy,
            self._restore_copy,
            self._compare_copy,
            name="copy",
        )

    def register(
        self,
        target,
        capture: Callable[[Any], Any],
        restore: Callable[[Any, ValueSnapshot], Any],
        compare: Callable[[Any, ValueSnapshot], bool],
        *,
        name: str,
        first: bool = False,
    ) -> None:
        matcher = target if callable(target) and not inspect.isclass(target) else self._build_type_matcher(target)
        entry = (matcher, capture, restore, compare, name)
        if first:
            self._entries.insert(0, entry)
        else:
            self._entries.append(entry)

    def capture(self, value: Any) -> ValueSnapshot:
        for matcher, capture, restore, compare, name in self._entries:
            if not matcher(value):
                continue
            raw = capture(value)
            if isinstance(raw, ValueSnapshot):
                return raw
            if isinstance(raw, tuple) and len(raw) == 2:
                restore_payload, compare_payload = raw
            else:
                restore_payload = raw
                compare_payload = copy.deepcopy(raw)
            return ValueSnapshot(
                strategy_name=name,
                original_value=value,
                restore_payload=restore_payload,
                compare_payload=compare_payload,
                restore_fn=restore,
                compare_fn=compare,
            )
        raise TypeError("no snapshot strategy matched value of type {0}".format(type(value).__qualname__))

    @staticmethod
    def restore(current_value: Any, snapshot: ValueSnapshot) -> Any:
        return snapshot.restore_fn(current_value, snapshot)

    @staticmethod
    def matches(current_value: Any, snapshot: ValueSnapshot) -> bool:
        return snapshot.compare_fn(current_value, snapshot)

    def diff(
        self,
        current_value: Any,
        snapshot: ValueSnapshot,
        *,
        max_items: int = 8,
        max_chars: int = 120,
        max_depth: int = 2,
    ):
        before_value = snapshot.compare_payload
        after_value = self._diff_target_value(current_value, snapshot)
        return describe_diff(
            before_value,
            after_value,
            max_items=max_items,
            max_chars=max_chars,
            max_depth=max_depth,
            kind_hint=snapshot.strategy_name,
        )

    @staticmethod
    def _build_type_matcher(target):
        if not inspect.isclass(target):
            raise TypeError("snapshot target must be a type or predicate")
        return lambda value, expected=target: isinstance(value, expected)

    @staticmethod
    def _capture_copy(value: Any):
        cloned = copy.deepcopy(value)
        return cloned, copy.deepcopy(cloned)

    @staticmethod
    def _restore_copy(current_value: Any, snapshot: ValueSnapshot) -> Any:
        return copy.deepcopy(snapshot.restore_payload)

    @staticmethod
    def _compare_copy(current_value: Any, snapshot: ValueSnapshot) -> bool:
        return deep_equal(current_value, snapshot.compare_payload)

    @staticmethod
    def _is_keras_weights_like(value: Any) -> bool:
        return _is_keras_weights_object(value)

    @staticmethod
    def _capture_keras_weights_like(value: Any):
        weights = copy.deepcopy(value.get_weights())
        trainable = getattr(value, "trainable", SNAPSHOT_MISSING)
        return {"weights": weights, "trainable": trainable}, {"weights": copy.deepcopy(weights), "trainable": trainable}

    @staticmethod
    def _restore_keras_weights_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        target = current_value if current_value is not SNAPSHOT_MISSING else snapshot.original_value
        if target is SNAPSHOT_MISSING:
            return copy.deepcopy(snapshot.original_value)
        payload = snapshot.restore_payload
        target.set_weights(copy.deepcopy(payload["weights"]))
        trainable = payload.get("trainable", SNAPSHOT_MISSING)
        if trainable is not SNAPSHOT_MISSING and hasattr(target, "trainable"):
            target.trainable = trainable
        return target

    @staticmethod
    def _compare_keras_weights_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        if current_value is SNAPSHOT_MISSING:
            return False
        try:
            payload = {
                "weights": current_value.get_weights(),
                "trainable": getattr(current_value, "trainable", SNAPSHOT_MISSING),
            }
        except Exception:
            return False
        return deep_equal(payload, snapshot.compare_payload)

    @staticmethod
    def _is_state_accessor_like(value: Any) -> bool:
        return _has_state_accessors(value) and not SnapshotStrategyRegistry._is_state_dict_like(value)

    @staticmethod
    def _capture_state_accessor_like(value: Any):
        get_state = _state_getter(value)
        state = copy.deepcopy(get_state())
        return state, copy.deepcopy(state)

    @staticmethod
    def _restore_state_accessor_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        target = current_value if current_value is not SNAPSHOT_MISSING else snapshot.original_value
        if target is SNAPSHOT_MISSING:
            return copy.deepcopy(snapshot.original_value)
        set_state = _state_setter(target)
        set_state(copy.deepcopy(snapshot.restore_payload))
        return target

    @staticmethod
    def _compare_state_accessor_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        if current_value is SNAPSHOT_MISSING:
            return False
        try:
            state = _state_getter(current_value)()
        except Exception:
            return False
        return deep_equal(state, snapshot.compare_payload)

    @staticmethod
    def _is_pandas_dataframe_like(value: Any) -> bool:
        return _is_pandas_dataframe(value)

    @staticmethod
    def _capture_pandas_dataframe_like(value: Any):
        cloned = _copy_pandas_like(value)
        return cloned, _copy_pandas_like(cloned)

    @staticmethod
    def _restore_pandas_dataframe_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        return _copy_pandas_like(snapshot.restore_payload)

    @staticmethod
    def _compare_pandas_dataframe_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        return _pandas_equals(current_value, snapshot.compare_payload)

    @staticmethod
    def _is_pandas_series_like(value: Any) -> bool:
        return _is_pandas_series(value)

    @staticmethod
    def _capture_pandas_series_like(value: Any):
        cloned = _copy_pandas_like(value)
        return cloned, _copy_pandas_like(cloned)

    @staticmethod
    def _restore_pandas_series_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        return _copy_pandas_like(snapshot.restore_payload)

    @staticmethod
    def _compare_pandas_series_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        return _pandas_equals(current_value, snapshot.compare_payload)

    @staticmethod
    def _is_numpy_array_like(value: Any) -> bool:
        return _is_numpy_array(value)

    @staticmethod
    def _capture_numpy_like(value: Any):
        cloned = value.copy()
        return cloned, cloned.copy()

    @staticmethod
    def _restore_numpy_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        payload = snapshot.restore_payload.copy()
        if current_value is not SNAPSHOT_MISSING:
            try:
                current_value[...] = payload
                return current_value
            except Exception:
                pass
        return payload

    @staticmethod
    def _compare_numpy_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        return deep_equal(current_value, snapshot.compare_payload)

    @staticmethod
    def _is_torch_tensor_like(value: Any) -> bool:
        return _is_torch_tensor(value) and hasattr(value, "clone")

    @staticmethod
    def _capture_torch_tensor_like(value: Any):
        cloned = value.detach().clone()
        device = str(getattr(value, "device", SNAPSHOT_MISSING))
        return {"tensor": cloned, "device": device}, {"tensor": cloned.detach().clone(), "device": device}

    @staticmethod
    def _restore_torch_tensor_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        payload = snapshot.restore_payload["tensor"]
        if current_value is not SNAPSHOT_MISSING and hasattr(current_value, "copy_"):
            try:
                current_value.copy_(payload.to(getattr(current_value, "device", payload.device)))
                return current_value
            except Exception:
                pass
        restored = payload.clone()
        device = snapshot.restore_payload.get("device", SNAPSHOT_MISSING)
        if device is not SNAPSHOT_MISSING and hasattr(restored, "to"):
            try:
                restored = restored.to(device)
            except Exception:
                pass
        return restored

    @staticmethod
    def _compare_torch_tensor_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        if current_value is SNAPSHOT_MISSING:
            return False
        try:
            return deep_equal(current_value, snapshot.compare_payload["tensor"])
        except Exception:
            return False

    @staticmethod
    def _is_state_dict_like(value: Any) -> bool:
        return callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None))

    @staticmethod
    def _capture_state_dict_like(value: Any):
        state_dict = copy.deepcopy(value.state_dict())
        training = getattr(value, "training", SNAPSHOT_MISSING)
        return {"state_dict": state_dict, "training": training}, {"state_dict": state_dict, "training": training}

    @staticmethod
    def _restore_state_dict_like(current_value: Any, snapshot: ValueSnapshot) -> Any:
        target = current_value if current_value is not SNAPSHOT_MISSING else snapshot.original_value
        if target is SNAPSHOT_MISSING:
            return copy.deepcopy(snapshot.original_value)
        payload = snapshot.restore_payload
        target.load_state_dict(copy.deepcopy(payload["state_dict"]))
        training = payload.get("training", SNAPSHOT_MISSING)
        if training is not SNAPSHOT_MISSING and hasattr(target, "train") and callable(getattr(target, "train")):
            target.train(bool(training))
        return target

    @staticmethod
    def _compare_state_dict_like(current_value: Any, snapshot: ValueSnapshot) -> bool:
        if current_value is SNAPSHOT_MISSING:
            return False
        try:
            current_state = current_value.state_dict()
        except Exception:
            return False
        payload = snapshot.compare_payload
        current_payload = {
            "state_dict": current_state,
            "training": getattr(current_value, "training", SNAPSHOT_MISSING),
        }
        return deep_equal(current_payload, payload)

    @staticmethod
    def _diff_target_value(current_value: Any, snapshot: ValueSnapshot):
        if snapshot.strategy_name == "keras_weights":
            if current_value is SNAPSHOT_MISSING:
                return SNAPSHOT_MISSING
            try:
                return {
                    "weights": current_value.get_weights(),
                    "trainable": getattr(current_value, "trainable", SNAPSHOT_MISSING),
                }
            except Exception:
                return "<unavailable>"
        if snapshot.strategy_name == "state_accessor":
            if current_value is SNAPSHOT_MISSING:
                return SNAPSHOT_MISSING
            try:
                return _state_getter(current_value)()
            except Exception:
                return "<unavailable>"
        if snapshot.strategy_name in {"pandas_dataframe", "pandas_series"}:
            if current_value is SNAPSHOT_MISSING:
                return SNAPSHOT_MISSING
            try:
                return _copy_pandas_like(current_value)
            except Exception:
                return "<unavailable>"
        if snapshot.strategy_name == "state_dict":
            if current_value is SNAPSHOT_MISSING:
                return SNAPSHOT_MISSING
            try:
                return {
                    "state_dict": current_value.state_dict(),
                    "training": getattr(current_value, "training", SNAPSHOT_MISSING),
                }
            except Exception:
                return "<unavailable>"
        if snapshot.strategy_name == "torch_tensor":
            if current_value is SNAPSHOT_MISSING:
                return SNAPSHOT_MISSING
            try:
                return {
                    "tensor": current_value.detach().clone(),
                    "device": str(getattr(current_value, "device", SNAPSHOT_MISSING)),
                }
            except Exception:
                return "<unavailable>"
        if snapshot.strategy_name == "numpy_array":
            if current_value is SNAPSHOT_MISSING:
                return SNAPSHOT_MISSING
            try:
                return current_value.copy()
            except Exception:
                return "<unavailable>"
        return current_value


def deep_equal(left: Any, right: Any) -> bool:
    if left is SNAPSHOT_MISSING or right is SNAPSHOT_MISSING:
        return left is right
    if type(left) != type(right) and not _tensor_pair(left, right):
        return False
    if _is_pandas_dataframe(left) and _is_pandas_dataframe(right):
        return _pandas_equals(left, right)
    if _is_pandas_series(left) and _is_pandas_series(right):
        return _pandas_equals(left, right)
    if _is_torch_tensor(left) and _is_torch_tensor(right):
        try:
            return bool(left.detach().cpu().equal(right.detach().cpu()))
        except Exception:
            return False
    if _is_numpy_array(left) and _is_numpy_array(right):
        try:
            return bool(left.shape == right.shape and (left == right).all())
        except Exception:
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(deep_equal(left[key], right[key]) for key in left.keys())
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(deep_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    if isinstance(left, (set, frozenset)) and isinstance(right, (set, frozenset)):
        return left == right
    try:
        return left == right
    except Exception:
        return False


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.startswith("torch") and hasattr(value, "detach") and hasattr(value, "cpu")


def _tensor_pair(left: Any, right: Any) -> bool:
    return _is_torch_tensor(left) and _is_torch_tensor(right)


def _is_numpy_array(value: Any) -> bool:
    return type(value).__module__.startswith("numpy") and hasattr(value, "shape") and hasattr(value, "dtype")


def _is_keras_weights_object(value: Any) -> bool:
    module_name = type(value).__module__
    return (
        module_name.startswith("keras")
        or module_name.startswith("tensorflow")
    ) and callable(getattr(value, "get_weights", None)) and callable(getattr(value, "set_weights", None))


def _has_state_accessors(value: Any) -> bool:
    return callable(_state_getter(value)) and callable(_state_setter(value))


def _state_getter(value: Any):
    getter = getattr(value, "get_state", None)
    if callable(getter):
        return getter
    getter = getattr(value, "getstate", None)
    if callable(getter):
        return getter
    return None


def _state_setter(value: Any):
    setter = getattr(value, "set_state", None)
    if callable(setter):
        return setter
    setter = getattr(value, "setstate", None)
    if callable(setter):
        return setter
    return None


def _is_pandas_dataframe(value: Any) -> bool:
    return type(value).__module__.startswith("pandas") and hasattr(value, "columns") and hasattr(value, "dtypes") and hasattr(value, "shape")


def _is_pandas_series(value: Any) -> bool:
    return type(value).__module__.startswith("pandas") and hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "name")


def _copy_pandas_like(value: Any):
    copy_method = getattr(value, "copy", None)
    if not callable(copy_method):
        raise TypeError("pandas-like object does not define copy()")
    try:
        return copy_method(deep=True)
    except TypeError:
        return copy_method()


def _pandas_equals(left: Any, right: Any) -> bool:
    if left is SNAPSHOT_MISSING or right is SNAPSHOT_MISSING:
        return left is right
    equals = getattr(left, "equals", None)
    if not callable(equals):
        return False
    try:
        return bool(equals(right))
    except Exception:
        return False


def describe_diff(
    before: Any,
    after: Any,
    *,
    max_items: int = 8,
    max_chars: int = 120,
    max_depth: int = 2,
    kind_hint: Optional[str] = None,
):
    if deep_equal(before, after):
        return {}
    payload = {
        "before": _summarize_diff_value(before, max_items=max_items, max_chars=max_chars, max_depth=max_depth),
        "after": _summarize_diff_value(after, max_items=max_items, max_chars=max_chars, max_depth=max_depth),
    }
    details = _build_diff_details(
        before,
        after,
        max_items=max_items,
        max_chars=max_chars,
        max_depth=max_depth,
        kind_hint=kind_hint,
    )
    if details:
        payload["details"] = details
        summary = _build_diff_summary(details)
        if summary:
            payload["summary"] = summary
    return payload


def _safe_preview(value: Any, max_chars: int) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:
        rendered = "<unrepresentable {0}: {1}>".format(type(value).__name__, exc)
    if len(rendered) > max_chars:
        return rendered[: max_chars - 3] + "..."
    return rendered


def _summarize_diff_value(value: Any, *, max_items: int, max_chars: int, max_depth: int):
    if value is SNAPSHOT_MISSING:
        return "<missing>"
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 12] + "...<len={0}>".format(len(value))
    if _is_keras_weights_object(value):
        payload = {
            "class": type(value).__qualname__,
            "trainable": getattr(value, "trainable", None),
        }
        count_params = getattr(value, "count_params", None)
        if callable(count_params):
            try:
                payload["param_count"] = int(count_params())
            except Exception:
                pass
        try:
            weights = value.get_weights()
            payload["weight_count"] = len(weights)
        except Exception:
            pass
        return payload
    if _is_pandas_dataframe(value):
        payload = {
            "shape": tuple(getattr(value, "shape", ())),
            "columns": list(getattr(value, "columns", ()))[:max_items],
            "dtypes": _pandas_dtype_mapping(getattr(value, "dtypes", {}), max_items=max_items),
        }
        preview = _pandas_dataframe_preview(value, limit=min(max_items, 3))
        if preview is not None:
            payload["preview"] = preview
        return payload
    if _is_pandas_series(value):
        payload = {
            "name": getattr(value, "name", None),
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "<unknown>")),
        }
        preview = _pandas_series_preview(value, limit=max_items)
        if preview is not None:
            payload["values"] = preview
        return payload
    if _is_torch_tensor(value):
        payload = {
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "<unknown>")),
            "device": str(getattr(value, "device", "<unknown>")),
        }
        try:
            if int(getattr(value, "numel")()) <= max_items:
                payload["values"] = value.detach().cpu().tolist()
        except Exception:
            pass
        return payload
    if _is_numpy_array(value):
        payload = {
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "<unknown>")),
        }
        try:
            if int(getattr(value, "size", 0)) <= max_items:
                payload["values"] = value.tolist()
        except Exception:
            pass
        return payload
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if max_depth <= 0:
        return _safe_preview(value, max_chars)
    if isinstance(value, Mapping):
        items = list(value.items())
        summary = {
            _safe_preview(key, min(max_chars, 40)): _summarize_diff_value(
                item,
                max_items=max_items,
                max_chars=max_chars,
                max_depth=max_depth - 1,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            summary["<truncated_items>"] = len(items) - max_items
        return summary
    if isinstance(value, (list, tuple)):
        rendered = [
            _summarize_diff_value(item, max_items=max_items, max_chars=max_chars, max_depth=max_depth - 1)
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            rendered.append("<{0} more items>".format(len(value) - max_items))
        return tuple(rendered) if isinstance(value, tuple) else rendered
    if isinstance(value, (set, frozenset)):
        rendered = sorted((_safe_preview(item, min(max_chars, 40)) for item in value))[:max_items]
        if len(value) > max_items:
            rendered.append("<{0} more items>".format(len(value) - max_items))
        return rendered
    return _safe_preview(value, max_chars)


def _build_diff_details(before: Any, after: Any, *, max_items: int, max_chars: int, max_depth: int, kind_hint: Optional[str] = None):
    if before is SNAPSHOT_MISSING or after is SNAPSHOT_MISSING:
        return {}
    if kind_hint == "torch_tensor" and isinstance(before, Mapping) and isinstance(after, Mapping):
        return _numeric_diff_details(before.get("tensor"), after.get("tensor"), kind="tensor", max_items=max_items)
    if kind_hint == "keras_weights" and isinstance(before, Mapping) and isinstance(after, Mapping):
        before_weights = before.get("weights", ())
        after_weights = after.get("weights", ())
        return {
            "kind": "keras_weights",
            "weight_count": len(before_weights),
            "changed_weight_indices": [
                index
                for index, (before_weight, after_weight) in enumerate(zip(before_weights, after_weights))
                if not deep_equal(before_weight, after_weight)
            ][:max_items],
            "trainable_before": before.get("trainable"),
            "trainable_after": after.get("trainable"),
        }
    if isinstance(before, str) and isinstance(after, str):
        return {
            "kind": "string",
            "before_length": len(before),
            "after_length": len(after),
            "first_changed_index": _first_changed_index(before, after),
        }
    if _is_torch_tensor(before) and _is_torch_tensor(after):
        return _numeric_diff_details(before, after, kind="tensor", max_items=max_items)
    if _is_numpy_array(before) and _is_numpy_array(after):
        return _numeric_diff_details(before, after, kind="ndarray", max_items=max_items)
    if _is_keras_weights_object(before) and _is_keras_weights_object(after):
        before_weights = before.get_weights()
        after_weights = after.get_weights()
        return {
            "kind": "keras_weights",
            "weight_count": len(before_weights),
            "changed_weight_indices": [
                index
                for index, (before_weight, after_weight) in enumerate(zip(before_weights, after_weights))
                if not deep_equal(before_weight, after_weight)
            ][:max_items],
            "trainable_before": getattr(before, "trainable", None),
            "trainable_after": getattr(after, "trainable", None),
        }
    if _is_pandas_dataframe(before) and _is_pandas_dataframe(after):
        before_columns = list(getattr(before, "columns", ()))
        after_columns = list(getattr(after, "columns", ()))
        return {
            "kind": "dataframe",
            "added_columns": after_columns[:max_items] if not before_columns else [column for column in after_columns if column not in before_columns][:max_items],
            "removed_columns": [column for column in before_columns if column not in after_columns][:max_items],
            "before_shape": tuple(getattr(before, "shape", ())),
            "after_shape": tuple(getattr(after, "shape", ())),
            "changed_preview_rows": _preview_changed_rows(before, after, limit=max_items),
        }
    if _is_pandas_series(before) and _is_pandas_series(after):
        return {
            "kind": "series",
            "before_shape": tuple(getattr(before, "shape", ())),
            "after_shape": tuple(getattr(after, "shape", ())),
            "before_name": getattr(before, "name", None),
            "after_name": getattr(after, "name", None),
            "first_changed_index": _first_changed_index(_pandas_series_preview(before, limit=max_items), _pandas_series_preview(after, limit=max_items)),
        }
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        changed_raw_keys = [
            key
            for key in before_keys.intersection(after_keys)
            if not deep_equal(before[key], after[key])
        ]
        return {
            "kind": "mapping",
            "added_keys": [_safe_preview(key, min(max_chars, 40)) for key in list(after_keys.difference(before_keys))[:max_items]],
            "removed_keys": [_safe_preview(key, min(max_chars, 40)) for key in list(before_keys.difference(after_keys))[:max_items]],
            "changed_keys": [_safe_preview(key, min(max_chars, 40)) for key in changed_raw_keys[:max_items]],
            "changed": {
                _safe_preview(key, min(max_chars, 40)): {
                    "before": _summarize_diff_value(before[key], max_items=max_items, max_chars=max_chars, max_depth=max_depth - 1),
                    "after": _summarize_diff_value(after[key], max_items=max_items, max_chars=max_chars, max_depth=max_depth - 1),
                }
                for key in changed_raw_keys[:max_items]
            },
        }
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        first_changed_index = None
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            if not deep_equal(before_item, after_item):
                first_changed_index = index
                break
        if first_changed_index is None and len(before) != len(after):
            first_changed_index = min(len(before), len(after))
        return {
            "kind": "sequence",
            "before_length": len(before),
            "after_length": len(after),
            "first_changed_index": first_changed_index,
            "before_preview": _sequence_change_preview(before, first_changed_index, max_items=max_items, max_chars=max_chars, max_depth=max_depth),
            "after_preview": _sequence_change_preview(after, first_changed_index, max_items=max_items, max_chars=max_chars, max_depth=max_depth),
        }
    if isinstance(before, (set, frozenset)) and isinstance(after, (set, frozenset)):
        return {
            "kind": "set",
            "added_items": sorted((_safe_preview(item, min(max_chars, 40)) for item in after.difference(before)))[:max_items],
            "removed_items": sorted((_safe_preview(item, min(max_chars, 40)) for item in before.difference(after)))[:max_items],
        }
    return {}


def _build_diff_summary(details: dict) -> Optional[str]:
    kind = details.get("kind")
    if kind == "mapping":
        return "mapping changed {0} key(s), added {1}, removed {2}".format(
            len(details.get("changed_keys", ())),
            len(details.get("added_keys", ())),
            len(details.get("removed_keys", ())),
        )
    if kind == "sequence":
        return "sequence length {0} -> {1}, first change at {2}".format(
            details.get("before_length"),
            details.get("after_length"),
            details.get("first_changed_index"),
        )
    if kind == "set":
        return "set added {0} item(s), removed {1}".format(
            len(details.get("added_items", ())),
            len(details.get("removed_items", ())),
        )
    if kind == "dataframe":
        return "dataframe shape {0} -> {1}, columns +{2}/-{3}".format(
            details.get("before_shape"),
            details.get("after_shape"),
            len(details.get("added_columns", ())),
            len(details.get("removed_columns", ())),
        )
    if kind == "series":
        return "series shape {0} -> {1}".format(details.get("before_shape"), details.get("after_shape"))
    if kind in {"tensor", "ndarray"}:
        return "{0} changed {1} element(s)".format(kind, details.get("changed_elements", 0))
    if kind == "string":
        return "string length {0} -> {1}, first change at {2}".format(
            details.get("before_length"),
            details.get("after_length"),
            details.get("first_changed_index"),
        )
    if kind == "keras_weights":
        return "keras weights changed at indices {0}".format(details.get("changed_weight_indices", ()))
    return None


def _pandas_dtype_mapping(dtypes: Any, *, max_items: int):
    if isinstance(dtypes, Mapping):
        return {str(key): str(value) for key, value in list(dtypes.items())[:max_items]}
    if hasattr(dtypes, "items"):
        try:
            items = list(dtypes.items())
            return {str(key): str(value) for key, value in items[:max_items]}
        except Exception:
            return {}
    return {}


def _pandas_dataframe_preview(value: Any, *, limit: int):
    head = getattr(value, "head", None)
    if not callable(head):
        return None
    try:
        preview = head(limit)
        to_dict = getattr(preview, "to_dict", None)
        if callable(to_dict):
            return to_dict(orient="records")
    except Exception:
        return None
    return None


def _pandas_series_preview(value: Any, *, limit: int):
    head = getattr(value, "head", None)
    if callable(head):
        try:
            preview = head(limit)
            tolist = getattr(preview, "tolist", None)
            if callable(tolist):
                return list(tolist())
        except Exception:
            return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return list(tolist()[:limit])
        except Exception:
            return None
    return None


def _preview_changed_rows(before: Any, after: Any, *, limit: int):
    before_rows = _pandas_dataframe_preview(before, limit=limit)
    after_rows = _pandas_dataframe_preview(after, limit=limit)
    if before_rows is None or after_rows is None:
        return []
    changes = []
    for index, (before_row, after_row) in enumerate(zip(before_rows, after_rows)):
        if before_row != after_row:
            changes.append({"index": index, "before": before_row, "after": after_row})
        if len(changes) >= limit:
            break
    return changes


def _first_changed_index(before: Any, after: Any) -> Optional[int]:
    if before is None or after is None:
        return None
    max_length = min(len(before), len(after))
    for index in range(max_length):
        if before[index] != after[index]:
            return index
    if len(before) != len(after):
        return max_length
    return None


def _sequence_change_preview(value: Sequence, index: Optional[int], *, max_items: int, max_chars: int, max_depth: int):
    if index is None:
        return _summarize_diff_value(value, max_items=max_items, max_chars=max_chars, max_depth=max_depth)
    start = max(0, index - 1)
    end = min(len(value), start + max_items)
    window = list(value[start:end])
    return {
        "start_index": start,
        "items": _summarize_diff_value(window, max_items=max_items, max_chars=max_chars, max_depth=max_depth),
    }


def _numeric_diff_details(before: Any, after: Any, *, kind: str, max_items: int):
    before_values = _extract_numeric_values(before)
    after_values = _extract_numeric_values(after)
    details = {
        "kind": kind,
        "before_shape": tuple(getattr(before, "shape", ())),
        "after_shape": tuple(getattr(after, "shape", ())),
    }
    if before_values is None or after_values is None or len(before_values) != len(after_values):
        return details
    changed_elements = 0
    max_abs_delta = 0.0
    sample = []
    for index, (before_value, after_value) in enumerate(zip(before_values, after_values)):
        if before_value == after_value:
            continue
        changed_elements += 1
        try:
            max_abs_delta = max(max_abs_delta, abs(float(after_value) - float(before_value)))
        except Exception:
            pass
        if len(sample) < max_items:
            sample.append({"index": index, "before": before_value, "after": after_value})
    details["changed_elements"] = changed_elements
    details["max_abs_delta"] = max_abs_delta
    if sample:
        details["sample"] = sample
    return details


def _extract_numeric_values(value: Any):
    raw = None
    if _is_torch_tensor(value):
        try:
            raw = value.detach().cpu().tolist()
        except Exception:
            return None
    elif _is_numpy_array(value):
        try:
            raw = value.tolist()
        except Exception:
            return None
    elif isinstance(value, list):
        raw = value
    else:
        return None
    flattened = []
    stack = [raw]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
            continue
        if isinstance(current, (int, float, bool)):
            flattened.append(current)
            continue
        return None
    return flattened
