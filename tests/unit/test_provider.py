import pytest

from paithon.exceptions import CodeGenerationError
from paithon.models import FunctionSnapshot, ImplementationRequest
from paithon.provider import OpenAIProvider, extract_python_source


class _FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class _FakeResponsesAPI:
    def __init__(self, output_text="```python\ndef add(x, y):\n    return x + y\n```"):
        self.calls = []
        self.output_text = output_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.output_text)


class _FakeClient:
    def __init__(self, output_text="```python\ndef add(x, y):\n    return x + y\n```"):
        self.responses = _FakeResponsesAPI(output_text=output_text)


def test_openai_provider_uses_broadly_supported_response_params():
    client = _FakeClient()
    provider = OpenAIProvider(client=client)
    request = ImplementationRequest(
        snapshot=FunctionSnapshot(
            module="demo",
            qualname="add",
            name="add",
            signature="(x, y)",
            contract="Return the sum of two numbers.",
            source="def add(x, y):\n    raise NotImplementedError\n",
            state_fields=(),
            state_summary={},
            state_schema={},
            globals_summary={},
            closure_summary={},
        )
    )

    source = provider.implement_function(request, "gpt-5-mini")

    assert source == "def add(x, y):\n    return x + y\n"
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0] == {
        "model": "gpt-5-mini",
        "instructions": client.responses.calls[0]["instructions"],
        "input": client.responses.calls[0]["input"],
    }
    assert "Do not import blocked modules" in client.responses.calls[0]["instructions"]
    assert "os" in client.responses.calls[0]["instructions"]
    assert "Do not call blocked builtins" in client.responses.calls[0]["instructions"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("```python\ndef add(x, y):\n    return x + y\n```", "def add(x, y):\n    return x + y\n"),
        ("def add(x, y):\n    return x + y", "def add(x, y):\n    return x + y\n"),
    ],
)
def test_extract_python_source_handles_fenced_and_plain_outputs(raw, expected):
    assert extract_python_source(raw) == expected


def test_openai_provider_raises_when_model_returns_no_text():
    client = _FakeClient(output_text="   ")
    provider = OpenAIProvider(client=client)
    request = ImplementationRequest(
        snapshot=FunctionSnapshot(
            module="demo",
            qualname="add",
            name="add",
            signature="(x, y)",
            contract="Return the sum of two numbers.",
            source="def add(x, y):\n    raise NotImplementedError\n",
            state_fields=(),
            state_summary={},
            state_schema={},
            globals_summary={},
            closure_summary={},
        )
    )

    with pytest.raises(CodeGenerationError):
        provider.implement_function(request, "gpt-5-mini")
