import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from uuid import UUID


REDACTED_TEXT = "<redacted>"


def normalize_redaction_tokens(values) -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(value).strip().lower()
                for value in (values or ())
                if str(value).strip()
            }
        )
    )


def normalize_redaction_paths(values) -> Tuple[Tuple[str, ...], ...]:
    normalized_paths = []
    for value in values or ():
        raw = str(value).strip().lower()
        if not raw:
            continue
        segments = tuple(segment for segment in raw.split(".") if segment)
        if segments and segments not in normalized_paths:
            normalized_paths.append(segments)
    return tuple(normalized_paths)


def is_redacted_name(
    name: Any,
    *,
    redacted_field_names=(),
    redacted_field_patterns=(),
) -> bool:
    if not isinstance(name, str):
        return False
    candidate = name.strip().lower()
    if not candidate:
        return False
    if candidate in redacted_field_names:
        return True
    return any(pattern and pattern in candidate for pattern in redacted_field_patterns)


def is_redacted_path(path, *, redacted_field_paths=()) -> bool:
    normalized_path = tuple(
        str(segment).strip().lower()
        for segment in (path or ())
        if str(segment).strip()
    )
    if not normalized_path:
        return False
    return any(_match_redaction_path(normalized_path, pattern) for pattern in redacted_field_paths)


def should_redact(
    *,
    path=(),
    name: Any = None,
    redacted_field_names=(),
    redacted_field_patterns=(),
    redacted_field_paths=(),
) -> bool:
    return is_redacted_name(
        name,
        redacted_field_names=redacted_field_names,
        redacted_field_patterns=redacted_field_patterns,
    ) or is_redacted_path(path, redacted_field_paths=redacted_field_paths)


def _match_redaction_path(path: Tuple[str, ...], pattern: Tuple[str, ...]) -> bool:
    if not pattern:
        return False
    return _match_redaction_segments(path, pattern, 0, 0)


def _match_redaction_segments(path: Tuple[str, ...], pattern: Tuple[str, ...], path_index: int, pattern_index: int) -> bool:
    while pattern_index < len(pattern):
        token = pattern[pattern_index]
        if token == "**":
            if pattern_index == len(pattern) - 1:
                return True
            return any(
                _match_redaction_segments(path, pattern, candidate_index, pattern_index + 1)
                for candidate_index in range(path_index, len(path) + 1)
            )
        if path_index >= len(path):
            return False
        if token != "*" and token != path[path_index]:
            return False
        path_index += 1
        pattern_index += 1
    return path_index == len(path)


@dataclass(frozen=True)
class SerializedValue:
    text: str
    schema: str


class StateSerializerRegistry:
    def __init__(self):
        self._entries: List[Tuple[Callable[[Any], bool], Callable[[Any], Any], Optional[str]]] = []
        self.register(self._is_dataclass_instance, self._serialize_dataclass, schema_name="dataclass")
        self.register(self._is_pydantic_like, self._serialize_pydantic, schema_name="pydantic")
        self.register(self._is_attrs_like, self._serialize_attrs, schema_name="attrs")
        self.register(self._is_namedtuple_instance, self._serialize_namedtuple, schema_name="namedtuple")
        self.register(self._is_enum_instance, self._serialize_enum, schema_name="enum")
        self.register(self._is_datetime_like, self._serialize_datetime, schema_name="datetime")
        self.register(self._is_path_like, self._serialize_path, schema_name="path")
        self.register(self._is_uuid_like, self._serialize_uuid, schema_name="uuid")
        self.register(self._is_decimal_like, self._serialize_decimal, schema_name="decimal")
        self.register(self._is_sqlalchemy_like, self._serialize_sqlalchemy, schema_name="sqlalchemy")
        self.register(self._is_sqlalchemy_session_like, self._serialize_sqlalchemy_session, schema_name="sqlalchemy-session")
        self.register(self._is_django_model_like, self._serialize_django_model, schema_name="django-model")
        self.register(self._is_django_queryset_like, self._serialize_django_queryset, schema_name="queryset")
        self.register(self._is_http_response_like, self._serialize_http_response, schema_name="http-response")
        self.register(self._is_sklearn_estimator_like, self._serialize_sklearn_estimator, schema_name="sklearn")
        self.register(self._is_keras_model_like, self._serialize_keras_model, schema_name="keras")
        self.register(self._is_pandas_dataframe_like, self._serialize_pandas_dataframe, schema_name="dataframe")
        self.register(self._is_pandas_series_like, self._serialize_pandas_series, schema_name="series")
        self.register(self._is_numpy_like, self._serialize_numpy_like, schema_name="ndarray")
        self.register(self._is_torch_tensor_like, self._serialize_torch_tensor_like, schema_name="tensor")
        self.register(self._is_stateful_object_like, self._serialize_stateful_object, schema_name="stateful")

    def register(self, target, serializer: Callable[[Any], Any], schema_name: Optional[str] = None, first: bool = False):
        matcher = target if callable(target) and not inspect.isclass(target) else self._build_type_matcher(target)
        entry = (matcher, serializer, schema_name)
        if first:
            self._entries.insert(0, entry)
        else:
            self._entries.append(entry)

    def serialize(
        self,
        value: Any,
        *,
        safe_repr: Callable[[Any, int], str],
        max_chars: int,
        max_depth: int = 2,
        max_items: int = 8,
        max_mapping_items: Optional[int] = None,
        max_sequence_items: Optional[int] = None,
        max_set_items: Optional[int] = None,
        redacted_field_names=(),
        redacted_field_patterns=(),
        redacted_field_paths=(),
        redaction_placeholder: str = REDACTED_TEXT,
        root_path=(),
    ) -> SerializedValue:
        seen = set()
        normalized_redacted_field_names = normalize_redaction_tokens(redacted_field_names)
        normalized_redacted_field_patterns = normalize_redaction_tokens(redacted_field_patterns)
        normalized_redacted_field_paths = normalize_redaction_paths(redacted_field_paths)
        for matcher, serializer, schema_name in self._entries:
            try:
                if not matcher(value):
                    continue
                raw = serializer(value)
                return self._normalize(
                    raw,
                    value,
                    safe_repr=safe_repr,
                    max_chars=max_chars,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_mapping_items=max_mapping_items,
                    max_sequence_items=max_sequence_items,
                    max_set_items=max_set_items,
                    redacted_field_names=normalized_redacted_field_names,
                    redacted_field_patterns=normalized_redacted_field_patterns,
                    redacted_field_paths=normalized_redacted_field_paths,
                    redaction_placeholder=redaction_placeholder,
                    path=tuple(root_path or ()),
                    schema_name=schema_name,
                    seen=seen,
                )
            except Exception:
                continue
        summarized = self._summarize_value(
            value,
            safe_repr=safe_repr,
            max_chars=max_chars,
            max_depth=max_depth,
            max_items=max_items,
            max_mapping_items=max_mapping_items,
            max_sequence_items=max_sequence_items,
            max_set_items=max_set_items,
            redacted_field_names=normalized_redacted_field_names,
            redacted_field_patterns=normalized_redacted_field_patterns,
            redacted_field_paths=normalized_redacted_field_paths,
            redaction_placeholder=redaction_placeholder,
            path=tuple(root_path or ()),
            seen=seen,
        )
        return SerializedValue(
            text=safe_repr(summarized, max_chars),
            schema=self._default_schema(value),
        )

    @staticmethod
    def _build_type_matcher(target):
        if not inspect.isclass(target):
            raise TypeError("serializer target must be a type or predicate")
        return lambda value, expected=target: isinstance(value, expected)

    def _normalize(
        self,
        raw,
        value: Any,
        *,
        safe_repr: Callable[[Any, int], str],
        max_chars: int,
        max_depth: int,
        max_items: int,
        max_mapping_items: Optional[int],
        max_sequence_items: Optional[int],
        max_set_items: Optional[int],
        redacted_field_names,
        redacted_field_patterns,
        redacted_field_paths,
        redaction_placeholder: str,
        path,
        schema_name: Optional[str],
        seen: set,
    ) -> SerializedValue:
        if isinstance(raw, SerializedValue):
            return SerializedValue(text=raw.text[:max_chars], schema=raw.schema)
        payload = raw
        schema = schema_name or self._default_schema(value)
        if isinstance(raw, tuple) and len(raw) == 2:
            payload, schema = raw
        summarized = self._summarize_root_payload(
            payload,
            safe_repr=safe_repr,
            max_chars=max_chars,
            max_depth=max_depth,
            max_items=max_items,
            max_mapping_items=max_mapping_items,
            max_sequence_items=max_sequence_items,
            max_set_items=max_set_items,
            redacted_field_names=redacted_field_names,
            redacted_field_patterns=redacted_field_patterns,
            redacted_field_paths=redacted_field_paths,
            redaction_placeholder=redaction_placeholder,
            path=path,
            seen=seen,
        )
        return SerializedValue(
            text=safe_repr(summarized, max_chars),
            schema=str(schema),
        )

    def _summarize_root_payload(
        self,
        payload: Any,
        *,
        safe_repr: Callable[[Any, int], str],
        max_chars: int,
        max_depth: int,
        max_items: int,
        max_mapping_items: Optional[int],
        max_sequence_items: Optional[int],
        max_set_items: Optional[int],
        redacted_field_names,
        redacted_field_patterns,
        redacted_field_paths,
        redaction_placeholder: str,
        path,
        seen: set,
    ):
        if isinstance(payload, Mapping):
            limit = self._mapping_limit(max_items, max_mapping_items)
            summary = {}
            for key, item in list(payload.items())[:limit]:
                rendered_key = key if isinstance(key, str) else safe_repr(key, min(max_chars, 40))
                next_path = tuple(path or ()) + (str(key),)
                if should_redact(
                    path=next_path,
                    name=str(key),
                    redacted_field_names=redacted_field_names,
                    redacted_field_patterns=redacted_field_patterns,
                    redacted_field_paths=redacted_field_paths,
                ):
                    summary[rendered_key] = redaction_placeholder
                    continue
                summary[rendered_key] = self._summarize_value(
                    item,
                    safe_repr=safe_repr,
                    max_chars=max_chars,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_mapping_items=max_mapping_items,
                    max_sequence_items=max_sequence_items,
                    max_set_items=max_set_items,
                    redacted_field_names=redacted_field_names,
                    redacted_field_patterns=redacted_field_patterns,
                    redacted_field_paths=redacted_field_paths,
                    redaction_placeholder=redaction_placeholder,
                    seen=seen,
                    path=next_path,
                )
            total_items = len(payload)
            if total_items > limit:
                summary["<truncated_items>"] = total_items - limit
                summary["<total_items>"] = total_items
            return summary
        if isinstance(payload, (list, tuple)) and not isinstance(payload, (str, bytes, bytearray)):
            limit = self._sequence_limit(max_items, max_sequence_items)
            items = [
                self._summarize_value(
                    item,
                    safe_repr=safe_repr,
                    max_chars=max_chars,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_mapping_items=max_mapping_items,
                    max_sequence_items=max_sequence_items,
                    max_set_items=max_set_items,
                    redacted_field_names=redacted_field_names,
                    redacted_field_patterns=redacted_field_patterns,
                    redacted_field_paths=redacted_field_paths,
                    redaction_placeholder=redaction_placeholder,
                    seen=seen,
                    path=tuple(path or ()) + (str(index),),
                )
                for index, item in enumerate(list(payload)[:limit])
            ]
            if len(payload) > limit:
                items.append("<{0} more items; total={1}>".format(len(payload) - limit, len(payload)))
            if isinstance(payload, tuple):
                return tuple(items)
            return items
        return self._summarize_value(
            payload,
            safe_repr=safe_repr,
            max_chars=max_chars,
            max_depth=max_depth,
            max_items=max_items,
            max_mapping_items=max_mapping_items,
            max_sequence_items=max_sequence_items,
            max_set_items=max_set_items,
            redacted_field_names=redacted_field_names,
            redacted_field_patterns=redacted_field_patterns,
            redacted_field_paths=redacted_field_paths,
            redaction_placeholder=redaction_placeholder,
            seen=seen,
            path=path,
        )

    def _summarize_value(
        self,
        value: Any,
        *,
        safe_repr: Callable[[Any, int], str],
        max_chars: int,
        max_depth: int,
        max_items: int,
        max_mapping_items: Optional[int],
        max_sequence_items: Optional[int],
        max_set_items: Optional[int],
        redacted_field_names,
        redacted_field_patterns,
        redacted_field_paths,
        redaction_placeholder: str,
        seen: set,
        path=(),
    ):
        current_name = path[-1] if path else None
        if should_redact(
            path=path,
            name=current_name,
            redacted_field_names=redacted_field_names,
            redacted_field_patterns=redacted_field_patterns,
            redacted_field_paths=redacted_field_paths,
        ):
            return redaction_placeholder
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate_string(value, max_chars)
        if isinstance(value, (bytes, bytearray)):
            return safe_repr(value, max_chars)
        if max_depth <= 0:
            return "<truncated_depth:{0}>".format(self._default_schema(value))
        object_id = id(value)
        if object_id in seen:
            return "<cycle:{0}>".format(self._default_schema(value))
        is_recursive = isinstance(value, (Mapping, Sequence, set, frozenset)) and not isinstance(value, (str, bytes, bytearray))
        if is_recursive:
            seen.add(object_id)
        try:
            if isinstance(value, Mapping):
                summary = {}
                items = list(value.items())
                limit = self._mapping_limit(max_items, max_mapping_items)
                for key, item in items[:limit]:
                    rendered_key = key if isinstance(key, str) else safe_repr(key, min(max_chars, 40))
                    next_path = tuple(path or ()) + (str(key),)
                    if should_redact(
                        path=next_path,
                        name=str(key),
                        redacted_field_names=redacted_field_names,
                        redacted_field_patterns=redacted_field_patterns,
                        redacted_field_paths=redacted_field_paths,
                    ):
                        summary[rendered_key] = redaction_placeholder
                        continue
                    summary[rendered_key] = self._summarize_value(
                        item,
                        safe_repr=safe_repr,
                        max_chars=max_chars,
                        max_depth=max_depth - 1,
                        max_items=max_items,
                        max_mapping_items=max_mapping_items,
                        max_sequence_items=max_sequence_items,
                        max_set_items=max_set_items,
                        redacted_field_names=redacted_field_names,
                        redacted_field_patterns=redacted_field_patterns,
                        redacted_field_paths=redacted_field_paths,
                        redaction_placeholder=redaction_placeholder,
                        seen=seen,
                        path=next_path,
                    )
                if len(items) > limit:
                    summary["<truncated_items>"] = len(items) - limit
                    summary["<total_items>"] = len(items)
                return summary
            if isinstance(value, (list, tuple)):
                limit = self._sequence_limit(max_items, max_sequence_items)
                items = [
                    self._summarize_value(
                        item,
                        safe_repr=safe_repr,
                        max_chars=max_chars,
                        max_depth=max_depth - 1,
                        max_items=max_items,
                        max_mapping_items=max_mapping_items,
                        max_sequence_items=max_sequence_items,
                        max_set_items=max_set_items,
                        redacted_field_names=redacted_field_names,
                        redacted_field_patterns=redacted_field_patterns,
                        redacted_field_paths=redacted_field_paths,
                        redaction_placeholder=redaction_placeholder,
                        seen=seen,
                        path=tuple(path or ()) + (str(index),),
                    )
                    for index, item in enumerate(list(value)[:limit])
                ]
                if len(value) > limit:
                    items.append("<{0} more items; total={1}>".format(len(value) - limit, len(value)))
                if isinstance(value, tuple):
                    return tuple(items)
                return items
            if isinstance(value, (set, frozenset)):
                limit = self._set_limit(max_items, max_set_items)
                rendered_items = sorted(list(value), key=lambda item: safe_repr(item, min(max_chars, 40)))
                rendered = [
                    self._summarize_value(
                        item,
                        safe_repr=safe_repr,
                        max_chars=max_chars,
                        max_depth=max_depth - 1,
                        max_items=max_items,
                        max_mapping_items=max_mapping_items,
                        max_sequence_items=max_sequence_items,
                        max_set_items=max_set_items,
                        redacted_field_names=redacted_field_names,
                        redacted_field_patterns=redacted_field_patterns,
                        redacted_field_paths=redacted_field_paths,
                        redaction_placeholder=redaction_placeholder,
                        seen=seen,
                        path=tuple(path or ()) + (str(index),),
                    )
                    for index, item in enumerate(rendered_items[:limit])
                ]
                if len(value) > limit:
                    rendered.append("<{0} more items; total={1}>".format(len(value) - limit, len(value)))
                return rendered
        finally:
            if is_recursive:
                seen.discard(object_id)
        return safe_repr(value, max_chars)

    @staticmethod
    def _truncate_string(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        if max_chars <= 12:
            return value[: max_chars - 3] + "..."
        return value[: max_chars - 12] + "...<len={0}>".format(len(value))

    @staticmethod
    def _mapping_limit(max_items: int, explicit_limit: Optional[int]) -> int:
        return max(1, explicit_limit if explicit_limit is not None else max_items)

    @staticmethod
    def _sequence_limit(max_items: int, explicit_limit: Optional[int]) -> int:
        return max(1, explicit_limit if explicit_limit is not None else max_items)

    @staticmethod
    def _set_limit(max_items: int, explicit_limit: Optional[int]) -> int:
        return max(1, explicit_limit if explicit_limit is not None else max_items)

    @staticmethod
    def _default_schema(value: Any) -> str:
        value_type = value if inspect.isclass(value) else type(value)
        return "{0}.{1}".format(value_type.__module__, value_type.__qualname__)

    @staticmethod
    def _format_schema(prefix: str, value: Any, field_names) -> str:
        return "{0}:{1}({2})".format(prefix, type(value).__qualname__, ", ".join(field_names))

    @staticmethod
    def _is_dataclass_instance(value: Any) -> bool:
        return is_dataclass(value) and not inspect.isclass(value)

    @staticmethod
    def _serialize_dataclass(value: Any):
        field_names = [field.name for field in fields(value)]
        payload = {name: getattr(value, name) for name in field_names}
        return payload, StateSerializerRegistry._format_schema("dataclass", value, field_names)

    @staticmethod
    def _is_pydantic_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        return hasattr(value, "model_dump") and (
            hasattr(type(value), "model_fields") or hasattr(type(value), "__fields__")
        )

    @staticmethod
    def _serialize_pydantic(value: Any):
        fields_map = getattr(type(value), "model_fields", None) or getattr(type(value), "__fields__", {})
        field_names = list(fields_map.keys())
        return value.model_dump(), StateSerializerRegistry._format_schema("pydantic", value, field_names)

    @staticmethod
    def _is_attrs_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        attrs = getattr(type(value), "__attrs_attrs__", None)
        return bool(attrs)

    @staticmethod
    def _serialize_attrs(value: Any):
        attrs = getattr(type(value), "__attrs_attrs__", ())
        field_names = [attribute.name for attribute in attrs]
        payload = {name: getattr(value, name) for name in field_names}
        return payload, StateSerializerRegistry._format_schema("attrs", value, field_names)

    @staticmethod
    def _is_namedtuple_instance(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        return isinstance(value, tuple) and hasattr(type(value), "_fields")

    @staticmethod
    def _serialize_namedtuple(value: Any):
        field_names = list(getattr(type(value), "_fields", ()))
        payload = {name: getattr(value, name) for name in field_names}
        return payload, StateSerializerRegistry._format_schema("namedtuple", value, field_names)

    @staticmethod
    def _is_enum_instance(value: Any) -> bool:
        return isinstance(value, Enum)

    @staticmethod
    def _serialize_enum(value: Enum):
        schema = "enum:{0}.{1}".format(type(value).__module__, type(value).__qualname__)
        return {"name": value.name, "value": value.value}, schema

    @staticmethod
    def _is_datetime_like(value: Any) -> bool:
        return isinstance(value, (datetime, date, time))

    @staticmethod
    def _serialize_datetime(value: Any):
        return value.isoformat(), "datetime:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_path_like(value: Any) -> bool:
        return isinstance(value, Path)

    @staticmethod
    def _serialize_path(value: Path):
        return str(value), "path:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_uuid_like(value: Any) -> bool:
        return isinstance(value, UUID)

    @staticmethod
    def _serialize_uuid(value: UUID):
        return str(value), "uuid:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_decimal_like(value: Any) -> bool:
        return isinstance(value, Decimal)

    @staticmethod
    def _serialize_decimal(value: Decimal):
        return str(value), "decimal:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_sqlalchemy_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        return hasattr(type(value), "__table__") or hasattr(type(value), "__mapper__")

    @staticmethod
    def _serialize_sqlalchemy(value: Any):
        column_names = []
        relationship_names = []
        mapper = getattr(type(value), "__mapper__", None)
        if mapper is not None:
            for column_attr in getattr(mapper, "column_attrs", ()):
                name = getattr(column_attr, "key", None) or getattr(column_attr, "name", None)
                if name and name not in column_names:
                    column_names.append(name)
            for relationship in getattr(mapper, "relationships", ()):
                name = getattr(relationship, "key", None) or getattr(relationship, "name", None)
                if name and name not in relationship_names:
                    relationship_names.append(name)
        table = getattr(type(value), "__table__", None)
        columns = getattr(table, "columns", ()) if table is not None else ()
        for column in columns:
            name = getattr(column, "name", str(column))
            if name not in column_names:
                column_names.append(name)
        payload = {name: getattr(value, name, None) for name in column_names[:20]}
        for name in relationship_names[:8]:
            payload[name] = StateSerializerRegistry._preview_relation_value(getattr(value, name, None))
        schema_fields = column_names[:20] + ["{0}[rel]".format(name) for name in relationship_names[:8]]
        return payload, StateSerializerRegistry._format_schema("sqlalchemy", value, schema_fields)

    @staticmethod
    def _is_sqlalchemy_session_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        module_name = type(value).__module__
        return "sqlalchemy" in module_name and all(
            hasattr(value, attribute_name)
            for attribute_name in ("new", "dirty", "deleted", "identity_map")
        )

    @staticmethod
    def _serialize_sqlalchemy_session(value: Any):
        bind = getattr(value, "bind", None)
        bind_label = None
        if bind is not None:
            bind_label = getattr(bind, "url", None) or getattr(bind, "name", None) or type(bind).__qualname__
        payload = {
            "identity_count": StateSerializerRegistry._safe_len(getattr(value, "identity_map", ())),
            "new_count": StateSerializerRegistry._safe_len(getattr(value, "new", ())),
            "dirty_count": StateSerializerRegistry._safe_len(getattr(value, "dirty", ())),
            "deleted_count": StateSerializerRegistry._safe_len(getattr(value, "deleted", ())),
        }
        if bind_label is not None:
            payload["bind"] = str(bind_label)
        return payload, "sqlalchemy-session:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_django_model_like(value: Any) -> bool:
        if inspect.isclass(value) or not hasattr(value, "_meta"):
            return False
        get_fields = getattr(value._meta, "get_fields", None)
        return callable(get_fields)

    @staticmethod
    def _serialize_django_model(value: Any):
        meta = value._meta
        concrete_fields = list(getattr(meta, "concrete_fields", ()) or ())
        many_to_many = list(getattr(meta, "many_to_many", ()) or ())
        if not concrete_fields and hasattr(meta, "get_fields"):
            concrete_fields = [
                field
                for field in meta.get_fields()
                if hasattr(field, "name") and not getattr(field, "many_to_many", False)
            ]
        field_names = []
        payload = {}
        for field in concrete_fields[:20]:
            name = getattr(field, "attname", None) or getattr(field, "name", None)
            if not name:
                continue
            field_names.append(name)
            payload[name] = getattr(value, name, None)
        for field in many_to_many[:8]:
            name = getattr(field, "name", None)
            if not name:
                continue
            field_names.append("{0}[m2m]".format(name))
            payload[name] = StateSerializerRegistry._preview_relation_value(getattr(value, name, None))
        return payload, StateSerializerRegistry._format_schema("django", value, field_names)

    @staticmethod
    def _is_django_queryset_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        module_name = type(value).__module__
        return "django" in module_name and callable(getattr(value, "count", None)) and hasattr(value, "model")

    @staticmethod
    def _serialize_django_queryset(value: Any):
        model = getattr(value, "model", None)
        preview_items = []
        try:
            preview_items = [StateSerializerRegistry._relation_identity(item) for item in list(value[:5])]
        except Exception:
            try:
                preview_items = [StateSerializerRegistry._relation_identity(item) for item in list(value)[:5]]
            except Exception:
                preview_items = []
        payload = {
            "model": getattr(model, "__qualname__", getattr(model, "__name__", type(model).__qualname__ if model is not None else "<unknown>")),
            "count": StateSerializerRegistry._safe_count(value),
            "items": preview_items,
        }
        return payload, "queryset:{0}".format(getattr(model, "__qualname__", type(value).__qualname__))

    @staticmethod
    def _is_http_response_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        module_name = type(value).__module__
        return (
            module_name.startswith("requests")
            or module_name.startswith("httpx")
            or "response" in type(value).__qualname__.lower()
        ) and hasattr(value, "status_code") and hasattr(value, "headers")

    @staticmethod
    def _serialize_http_response(value: Any):
        headers = dict(getattr(value, "headers", {}) or {})
        request = getattr(value, "request", None)
        payload = {
            "status_code": getattr(value, "status_code", None),
            "url": str(getattr(value, "url", getattr(request, "url", None))),
            "method": getattr(value, "request_method", None) or getattr(request, "method", None),
            "headers": {
                str(key): headers[key]
                for key in list(headers.keys())[:12]
            },
        }
        content_type = headers.get("content-type") or headers.get("Content-Type")
        if content_type:
            payload["content_type"] = content_type
        body_preview = StateSerializerRegistry._http_response_body_preview(value)
        if body_preview is not None:
            payload["body_preview"] = body_preview
        return payload, "http-response:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_sklearn_estimator_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        return type(value).__module__.startswith("sklearn") and callable(getattr(value, "get_params", None))

    @staticmethod
    def _serialize_sklearn_estimator(value: Any):
        try:
            params = value.get_params(deep=False)
        except Exception:
            params = {}
        param_items = list(params.items())[:12] if isinstance(params, Mapping) else []
        payload = {
            "class": type(value).__qualname__,
            "params": {str(key): item for key, item in param_items},
        }
        for attribute_name in ("n_features_in_", "classes_", "feature_names_in_"):
            if hasattr(value, attribute_name):
                payload[attribute_name] = getattr(value, attribute_name)
        return payload, "sklearn:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_keras_model_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        module_name = type(value).__module__
        return (
            module_name.startswith("keras")
            or module_name.startswith("tensorflow")
        ) and callable(getattr(value, "get_config", None))

    @staticmethod
    def _serialize_keras_model(value: Any):
        try:
            config = value.get_config()
        except Exception:
            config = {}
        config_keys = list(config.keys())[:12] if isinstance(config, Mapping) else []
        payload = {
            "name": getattr(value, "name", None),
            "trainable": getattr(value, "trainable", None),
            "built": getattr(value, "built", None),
            "config_keys": config_keys,
        }
        count_params = getattr(value, "count_params", None)
        if callable(count_params):
            try:
                payload["param_count"] = int(count_params())
            except Exception:
                pass
        return payload, "keras:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_pandas_dataframe_like(value: Any) -> bool:
        return type(value).__module__.startswith("pandas") and hasattr(value, "columns") and hasattr(value, "dtypes") and hasattr(value, "shape")

    @staticmethod
    def _serialize_pandas_dataframe(value: Any):
        columns = list(getattr(value, "columns", ()))
        dtypes = StateSerializerRegistry._dtype_mapping(getattr(value, "dtypes", {}), limit=12)
        payload = {
            "shape": tuple(getattr(value, "shape", ())),
            "columns": columns[:12],
            "dtypes": dtypes,
        }
        preview = StateSerializerRegistry._dataframe_preview(value, limit=3)
        if preview is not None:
            payload["preview"] = preview
        return payload, "dataframe:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_pandas_series_like(value: Any) -> bool:
        return type(value).__module__.startswith("pandas") and hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "name")

    @staticmethod
    def _serialize_pandas_series(value: Any):
        values = None
        try:
            values = list(value.head(8).tolist())
        except Exception:
            try:
                values = list(value.tolist()[:8])
            except Exception:
                values = None
        payload = {
            "name": getattr(value, "name", None),
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "<unknown>")),
        }
        if values is not None:
            payload["values"] = values
        return payload, "series:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_numpy_like(value: Any) -> bool:
        return type(value).__module__.startswith("numpy") and hasattr(value, "shape") and hasattr(value, "dtype")

    @staticmethod
    def _serialize_numpy_like(value: Any):
        preview = None
        try:
            if getattr(value, "ndim", 2) <= 1 and getattr(value, "size", 0) <= 8:
                preview = value.tolist()
        except Exception:
            preview = None
        payload = {
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "<unknown>")),
        }
        if preview is not None:
            payload["values"] = preview
        return payload, "ndarray:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_torch_tensor_like(value: Any) -> bool:
        return type(value).__module__.startswith("torch") and hasattr(value, "shape") and hasattr(value, "dtype")

    @staticmethod
    def _serialize_torch_tensor_like(value: Any):
        payload = {
            "shape": tuple(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "<unknown>")),
            "device": str(getattr(value, "device", "<unknown>")),
        }
        try:
            if int(getattr(value, "numel")()) <= 8:
                payload["values"] = value.detach().cpu().tolist()
        except Exception:
            pass
        return payload, "tensor:{0}".format(type(value).__qualname__)

    @staticmethod
    def _is_stateful_object_like(value: Any) -> bool:
        if inspect.isclass(value):
            return False
        return callable(getattr(value, "state_dict", None)) and callable(getattr(value, "load_state_dict", None))

    @staticmethod
    def _serialize_stateful_object(value: Any):
        try:
            state = value.state_dict()
        except Exception:
            state = {}
        keys = list(state.keys())[:12] if isinstance(state, Mapping) else []
        payload = {
            "class": "{0}.{1}".format(type(value).__module__, type(value).__qualname__),
            "state_keys": keys,
            "num_state_keys": len(state) if isinstance(state, Mapping) else 0,
        }
        return payload, "stateful:{0}".format(type(value).__qualname__)

    @staticmethod
    def _preview_relation_value(value: Any):
        if value is None:
            return None
        manager_all = getattr(value, "all", None)
        if callable(manager_all):
            try:
                related = list(manager_all())
                return {
                    "count": len(related),
                    "items": [StateSerializerRegistry._relation_identity(item) for item in related[:8]],
                }
            except Exception:
                count = getattr(value, "count", None)
                if callable(count):
                    try:
                        return {"count": int(count())}
                    except Exception:
                        pass
                return {"type": StateSerializerRegistry._default_schema(value)}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return {
                "count": len(value),
                "items": [StateSerializerRegistry._relation_identity(item) for item in list(value)[:8]],
            }
        return StateSerializerRegistry._relation_identity(value)

    @staticmethod
    def _relation_identity(value: Any):
        for attribute_name in ("pk", "id", "identifier", "name"):
            if hasattr(value, attribute_name):
                try:
                    return {attribute_name: getattr(value, attribute_name)}
                except Exception:
                    continue
        return {"type": StateSerializerRegistry._default_schema(value)}

    @staticmethod
    def _safe_len(value: Any) -> int:
        try:
            return len(value)
        except Exception:
            return 0

    @staticmethod
    def _safe_count(value: Any) -> int:
        count = getattr(value, "count", None)
        if callable(count):
            try:
                return int(count())
            except Exception:
                return 0
        return StateSerializerRegistry._safe_len(value)

    @staticmethod
    def _dtype_mapping(dtypes: Any, *, limit: int) -> dict:
        if isinstance(dtypes, Mapping):
            return {
                str(key): str(value)
                for key, value in list(dtypes.items())[:limit]
            }
        if hasattr(dtypes, "items"):
            try:
                items = list(dtypes.items())
                return {str(key): str(value) for key, value in items[:limit]}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _dataframe_preview(value: Any, *, limit: int):
        head = getattr(value, "head", None)
        if callable(head):
            try:
                preview = head(limit)
                to_dict = getattr(preview, "to_dict", None)
                if callable(to_dict):
                    return to_dict(orient="records")
            except Exception:
                return None
        return None

    @staticmethod
    def _http_response_body_preview(value: Any):
        json_method = getattr(value, "json", None)
        if callable(json_method):
            try:
                return json_method()
            except Exception:
                pass
        text_value = getattr(value, "text", None)
        if isinstance(text_value, str):
            return text_value[:200]
        content = getattr(value, "content", None)
        if isinstance(content, (bytes, bytearray)):
            try:
                return bytes(content[:200]).decode("utf-8", errors="replace")
            except Exception:
                return repr(bytes(content[:60]))
        return None
