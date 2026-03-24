from dataclasses import dataclass

from paithon.context import (
    bind_call_arguments,
    build_call_summary,
    build_snapshot,
    get_function_source,
    is_placeholder_function,
    safe_repr,
    strip_leading_decorators,
    summarize_closure,
    summarize_globals,
)
from paithon.serializers import StateSerializerRegistry


class BrokenRepr:
    def __repr__(self):
        raise RuntimeError("boom")


@dataclass
class Profile:
    name: str
    score: int


GLOBAL_LABEL = "global-value"
HELPER_MAPPING = {"enabled": True}


def test_safe_repr_handles_unrepresentable_values_and_truncates():
    assert safe_repr(BrokenRepr(), 200) == "<unrepresentable BrokenRepr: boom>"
    assert safe_repr("x" * 12, 8) == "'xxxx..."


def test_strip_leading_decorators_returns_function_body():
    source = "@first\n@second(value=1)\ndef slugify(text):\n    return text.lower()\n"

    assert strip_leading_decorators(source) == "def slugify(text):\n    return text.lower()\n"


def test_get_function_source_prefers_generated_paithon_source():
    def slugify(text):
        return text

    slugify.__paithon_source__ = "def slugify(text):\n    return text.lower()\n"

    assert get_function_source(slugify) == "def slugify(text):\n    return text.lower()\n"


def test_is_placeholder_function_detects_supported_placeholder_shapes():
    def with_pass():
        pass

    def with_ellipsis():
        ...

    def with_not_implemented():
        raise NotImplementedError

    def real_impl():
        return 1

    assert is_placeholder_function(with_pass) is True
    assert is_placeholder_function(with_ellipsis) is True
    assert is_placeholder_function(with_not_implemented) is True
    assert is_placeholder_function(real_impl) is False


def test_summarize_globals_ignores_builtins_and_dunder_names():
    def render(values):
        return GLOBAL_LABEL, HELPER_MAPPING, len(values), __name__

    summary = summarize_globals(render, get_function_source(render), max_chars=200)

    assert summary["GLOBAL_LABEL"] == "'global-value'"
    assert summary["HELPER_MAPPING"] == "{'enabled': True}"
    assert "len" not in summary
    assert "__name__" not in summary


def test_summarize_closure_and_bind_call_arguments():
    prefix = "pfx"

    def render(value, *, suffix="!"):
        return "{0}:{1}:{2}".format(prefix, value, suffix)

    closure_summary = summarize_closure(render, max_chars=200)
    bound = bind_call_arguments(render, args=(3,), kwargs={"suffix": "?"})
    invalid = bind_call_arguments(render, args=(), kwargs={"unknown": 1})

    assert closure_summary == {"prefix": "'pfx'"}
    assert bound.arguments["value"] == 3
    assert bound.arguments["suffix"] == "?"
    assert invalid is None


def test_build_call_summary_and_snapshot_include_state_globals_and_closure():
    registry = StateSerializerRegistry()
    prefix = "pre"

    class Box:
        def __init__(self):
            self.profile = Profile("Ada", 9)

    def render(self, value):
        return "{0}:{1}:{2}:{3}".format(prefix, GLOBAL_LABEL, self.profile.name, value)

    box = Box()
    call_summary = build_call_summary(render, args=(box, 7), kwargs={}, max_chars=200)
    snapshot = build_snapshot(
        render,
        contract="Render a profile label.",
        max_chars=200,
        args=(box, 7),
        kwargs={},
        state_fields=["profile"],
        serializer_registry=registry,
    )

    assert call_summary["value"] == "7"
    assert snapshot.contract == "Render a profile label."
    assert snapshot.globals_summary["GLOBAL_LABEL"] == "'global-value'"
    assert snapshot.closure_summary["prefix"] == "'pre'"
    assert snapshot.state_schema["profile"] == "dataclass:Profile(name, score)"
    assert snapshot.state_summary["profile"] == "{'name': 'Ada', 'score': 9}"


def test_build_snapshot_and_call_summary_apply_redaction_policy():
    registry = StateSerializerRegistry()
    secret_token = "top-secret"
    api_token = "global-secret"

    class Box:
        def __init__(self):
            self.credentials = {"token": "abc123", "profile": {"password": "pw"}}

    def render(self, token):
        return "{0}:{1}:{2}".format(secret_token, api_token, token)

    box = Box()
    call_summary = build_call_summary(
        render,
        args=(box, "visible-token"),
        kwargs={},
        max_chars=200,
        serializer_registry=registry,
        redacted_field_names=("token",),
        redacted_field_patterns=("secret", "password", "token"),
    )
    snapshot = build_snapshot(
        render,
        contract="Render credentials safely.",
        max_chars=200,
        args=(box, "visible-token"),
        kwargs={},
        state_fields=["credentials"],
        serializer_registry=registry,
        redacted_field_names=("token",),
        redacted_field_patterns=("secret", "password", "token"),
    )

    assert call_summary["token"] == "<redacted>"
    assert "<redacted>" in snapshot.state_summary["credentials"]
    assert "abc123" not in snapshot.state_summary["credentials"]
    assert snapshot.closure_summary["api_token"] == "<redacted>"
    assert snapshot.closure_summary["secret_token"] == "<redacted>"


def test_build_snapshot_supports_path_based_redaction_and_custom_placeholder():
    registry = StateSerializerRegistry()

    class Box:
        def __init__(self):
            self.payload = {"headers": {"authorization": "Bearer secret", "x-request-id": "abc"}}

    def render(self, payload):
        return payload

    box = Box()
    call_summary = build_call_summary(
        render,
        args=(box, {"headers": {"authorization": "Bearer secret", "x-request-id": "abc"}}),
        kwargs={},
        max_chars=200,
        serializer_registry=registry,
        redacted_field_paths=("payload.headers.authorization", "payload.headers.x-request-id"),
        redaction_placeholder="<hidden>",
    )
    snapshot = build_snapshot(
        render,
        contract="Render payload safely.",
        max_chars=200,
        args=(box, {"headers": {"authorization": "Bearer secret", "x-request-id": "abc"}}),
        kwargs={},
        state_fields=["payload"],
        serializer_registry=registry,
        redacted_field_paths=("payload.headers.authorization", "payload.headers.x-request-id"),
        redaction_placeholder="<hidden>",
    )

    assert "<hidden>" in call_summary["payload"]
    assert "Bearer secret" not in call_summary["payload"]
    assert "<hidden>" in snapshot.state_summary["payload"]
    assert "abc" not in snapshot.state_summary["payload"]
