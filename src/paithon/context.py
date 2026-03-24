import ast
import builtins
import inspect
import textwrap
from typing import Any, Dict, Iterable, Optional, Sequence

from .models import FunctionSnapshot
from .serializers import (
    REDACTED_TEXT,
    StateSerializerRegistry,
    is_redacted_name,
    is_redacted_path,
    normalize_redaction_paths,
    normalize_redaction_tokens,
    should_redact,
)


def safe_repr(value: Any, max_chars: int) -> str:
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive branch
        rendered = "<unrepresentable {0}: {1}>".format(type(value).__name__, exc)
    if len(rendered) > max_chars:
        return rendered[: max_chars - 3] + "..."
    return rendered


def strip_leading_decorators(source: str) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            return "\n".join(lines[index:]).strip() + "\n"
    return source.strip() + "\n"


def get_function_source(func: Any) -> str:
    generated_source = getattr(func, "__paithon_source__", None)
    if generated_source:
        return generated_source
    try:
        lines, _ = inspect.getsourcelines(func)
    except (OSError, IOError, TypeError):
        return "def {0}{1}:\n    raise NotImplementedError('Source unavailable')\n".format(
            func.__name__,
            inspect.signature(func),
        )
    source = textwrap.dedent("".join(lines))
    return strip_leading_decorators(source)


def is_placeholder_function(func: Any) -> bool:
    source = get_function_source(func)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = list(tree.body[0].body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis
    if not isinstance(statement, ast.Raise):
        return False
    exc = statement.exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def summarize_globals(
    func: Any,
    source: str,
    max_chars: int,
    limit: int = 20,
    redacted_field_names=(),
    redacted_field_patterns=(),
    redacted_field_paths=(),
    redaction_placeholder: str = REDACTED_TEXT,
) -> Dict[str, str]:
    normalized_redacted_field_names = normalize_redaction_tokens(redacted_field_names)
    normalized_redacted_field_patterns = normalize_redaction_tokens(redacted_field_patterns)
    normalized_redacted_field_paths = normalize_redaction_paths(redacted_field_paths)
    names = []
    for name in func.__code__.co_names:
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in dir(builtins):
            continue
        if name in func.__globals__ and name not in names:
            names.append(name)
    summary = {}
    for name in names[:limit]:
        if should_redact(
            path=(name,),
            name=name,
            redacted_field_names=normalized_redacted_field_names,
            redacted_field_patterns=normalized_redacted_field_patterns,
            redacted_field_paths=normalized_redacted_field_paths,
        ):
            summary[name] = redaction_placeholder
            continue
        summary[name] = safe_repr(func.__globals__[name], max_chars)
    return summary


def summarize_closure(
    func: Any,
    max_chars: int,
    redacted_field_names=(),
    redacted_field_patterns=(),
    redacted_field_paths=(),
    redaction_placeholder: str = REDACTED_TEXT,
) -> Dict[str, str]:
    normalized_redacted_field_names = normalize_redaction_tokens(redacted_field_names)
    normalized_redacted_field_patterns = normalize_redaction_tokens(redacted_field_patterns)
    normalized_redacted_field_paths = normalize_redaction_paths(redacted_field_paths)
    if not func.__closure__:
        return {}
    summary = {}
    for name, cell in zip(func.__code__.co_freevars, func.__closure__):
        if should_redact(
            path=(name,),
            name=name,
            redacted_field_names=normalized_redacted_field_names,
            redacted_field_patterns=normalized_redacted_field_patterns,
            redacted_field_paths=normalized_redacted_field_paths,
        ):
            summary[name] = redaction_placeholder
            continue
        try:
            summary[name] = safe_repr(cell.cell_contents, max_chars)
        except ValueError:
            summary[name] = "<empty closure cell>"
    return summary


def bind_call_arguments(func: Any, args: Iterable[Any], kwargs: Dict[str, Any]):
    try:
        return inspect.signature(func).bind_partial(*args, **kwargs)
    except TypeError:
        return None


def summarize_state(
    func: Any,
    args: Iterable[Any],
    kwargs: Dict[str, Any],
    state_fields: Sequence[str],
    max_chars: int,
    serializer_registry: Optional[StateSerializerRegistry] = None,
    max_depth: int = 2,
    max_items: int = 8,
    max_mapping_items: Optional[int] = None,
    max_sequence_items: Optional[int] = None,
    max_set_items: Optional[int] = None,
    redacted_field_paths=(),
    redaction_placeholder: str = REDACTED_TEXT,
    redacted_field_names=(),
    redacted_field_patterns=(),
):
    if not state_fields:
        return {}, {}
    bound = bind_call_arguments(func, args, kwargs)
    if bound is None or not bound.arguments:
        return {}, {}
    _, instance = next(iter(bound.arguments.items()))
    normalized_redacted_field_names = normalize_redaction_tokens(redacted_field_names)
    normalized_redacted_field_patterns = normalize_redaction_tokens(redacted_field_patterns)
    normalized_redacted_field_paths = normalize_redaction_paths(redacted_field_paths)
    summary = {}
    schema = {}
    for field in state_fields:
        try:
            if not hasattr(instance, field):
                summary[field] = "<missing attribute>"
                schema[field] = "<missing attribute>"
                continue
            value = getattr(instance, field)
            if should_redact(
                path=(field,),
                name=field,
                redacted_field_names=normalized_redacted_field_names,
                redacted_field_patterns=normalized_redacted_field_patterns,
                redacted_field_paths=normalized_redacted_field_paths,
            ):
                summary[field] = redaction_placeholder
                schema[field] = "{0}.{1}".format(type(value).__module__, type(value).__qualname__)
                continue
            if serializer_registry is None:
                summary[field] = safe_repr(value, max_chars)
                schema[field] = "{0}.{1}".format(type(value).__module__, type(value).__qualname__)
            else:
                serialized = serializer_registry.serialize(
                    value,
                    safe_repr=safe_repr,
                    max_chars=max_chars,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_mapping_items=max_mapping_items,
                    max_sequence_items=max_sequence_items,
                    max_set_items=max_set_items,
                    redacted_field_paths=redacted_field_paths,
                    redaction_placeholder=redaction_placeholder,
                    redacted_field_names=normalized_redacted_field_names,
                    redacted_field_patterns=normalized_redacted_field_patterns,
                    root_path=(field,),
                )
                summary[field] = serialized.text
                schema[field] = serialized.schema
        except Exception as exc:  # pragma: no cover - defensive branch
            summary[field] = "<error reading attribute {0}: {1}>".format(field, exc)
            schema[field] = "<error>"
    return summary, schema


def build_call_summary(
    func: Any,
    args: Iterable[Any],
    kwargs: Dict[str, Any],
    max_chars: int,
    serializer_registry: Optional[StateSerializerRegistry] = None,
    max_depth: int = 2,
    max_items: int = 8,
    max_mapping_items: Optional[int] = None,
    max_sequence_items: Optional[int] = None,
    max_set_items: Optional[int] = None,
    redacted_field_paths=(),
    redaction_placeholder: str = REDACTED_TEXT,
    redacted_field_names=(),
    redacted_field_patterns=(),
) -> Dict[str, str]:
    bound = bind_call_arguments(func, args, kwargs)
    if bound is None:
        items = [("args", tuple(args)), ("kwargs", kwargs)]
    else:
        items = bound.arguments.items()
    normalized_redacted_field_names = normalize_redaction_tokens(redacted_field_names)
    normalized_redacted_field_patterns = normalize_redaction_tokens(redacted_field_patterns)
    normalized_redacted_field_paths = normalize_redaction_paths(redacted_field_paths)
    summary = {}
    for name, value in items:
        if should_redact(
            path=(name,),
            name=name,
            redacted_field_names=normalized_redacted_field_names,
            redacted_field_patterns=normalized_redacted_field_patterns,
            redacted_field_paths=normalized_redacted_field_paths,
        ):
            summary[name] = redaction_placeholder
            continue
        if serializer_registry is None:
            summary[name] = safe_repr(value, max_chars)
            continue
        serialized = serializer_registry.serialize(
            value,
            safe_repr=safe_repr,
            max_chars=max_chars,
            max_depth=max_depth,
            max_items=max_items,
            max_mapping_items=max_mapping_items,
            max_sequence_items=max_sequence_items,
            max_set_items=max_set_items,
            redacted_field_paths=redacted_field_paths,
            redaction_placeholder=redaction_placeholder,
            redacted_field_names=normalized_redacted_field_names,
            redacted_field_patterns=normalized_redacted_field_patterns,
            root_path=(name,),
        )
        summary[name] = serialized.text
    return summary


def build_snapshot(
    func: Any,
    contract: str,
    max_chars: int,
    source: Optional[str] = None,
    args: Iterable[Any] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    state_fields: Sequence[str] = (),
    serializer_registry: Optional[StateSerializerRegistry] = None,
    max_depth: int = 2,
    max_items: int = 8,
    max_mapping_items: Optional[int] = None,
    max_sequence_items: Optional[int] = None,
    max_set_items: Optional[int] = None,
    redacted_field_paths=(),
    redaction_placeholder: str = REDACTED_TEXT,
    redacted_field_names=(),
    redacted_field_patterns=(),
) -> FunctionSnapshot:
    source_text = source or get_function_source(func)
    call_kwargs = kwargs or {}
    state_summary, state_schema = summarize_state(
        func,
        args,
        call_kwargs,
        state_fields,
        max_chars,
        serializer_registry=serializer_registry,
        max_depth=max_depth,
        max_items=max_items,
        max_mapping_items=max_mapping_items,
        max_sequence_items=max_sequence_items,
        max_set_items=max_set_items,
        redacted_field_paths=redacted_field_paths,
        redaction_placeholder=redaction_placeholder,
        redacted_field_names=redacted_field_names,
        redacted_field_patterns=redacted_field_patterns,
    )
    return FunctionSnapshot(
        module=func.__module__,
        qualname=func.__qualname__,
        name=func.__name__,
        signature=str(inspect.signature(func)),
        contract=contract,
        source=source_text,
        state_fields=tuple(state_fields),
        state_summary=state_summary,
        state_schema=state_schema,
        globals_summary=summarize_globals(
            func,
            source_text,
            max_chars,
            redacted_field_paths=redacted_field_paths,
            redaction_placeholder=redaction_placeholder,
            redacted_field_names=redacted_field_names,
            redacted_field_patterns=redacted_field_patterns,
        ),
        closure_summary=summarize_closure(
            func,
            max_chars,
            redacted_field_paths=redacted_field_paths,
            redaction_placeholder=redaction_placeholder,
            redacted_field_names=redacted_field_names,
            redacted_field_patterns=redacted_field_patterns,
        ),
    )
