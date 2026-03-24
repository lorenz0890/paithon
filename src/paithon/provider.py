import re
from abc import ABC, abstractmethod
from typing import Dict

from openai import OpenAI

from .exceptions import CodeGenerationError, CodeRepairError, ProviderConfigurationError
from .models import ImplementationRequest, RepairRequest

IMPLEMENT_SYSTEM_PROMPT = """You write production Python functions.
Return only valid Python source code.
Return exactly one function definition with the requested function name and signature.
Do not include markdown, explanations, or tests.
Preserve the given contract.
Prefer deterministic code with standard-library-only dependencies unless the provided source already relies on other modules.
Do not import blocked modules: ctypes, marshal, multiprocessing, os, pickle, socket, subprocess.
Do not call blocked builtins: eval, exec, compile, __import__, open, input.
"""

REPAIR_SYSTEM_PROMPT = """You repair production Python functions.
Return only valid Python source code.
Return exactly one repaired function definition with the same name and signature.
Fix the observed failure while preserving the stated contract.
Do not include markdown, explanations, or tests.
Prefer minimal changes and standard-library-only dependencies unless the provided source already relies on other modules.
Do not import blocked modules: ctypes, marshal, multiprocessing, os, pickle, socket, subprocess.
Do not call blocked builtins: eval, exec, compile, __import__, open, input.
"""


def extract_python_source(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    return stripped + "\n"


def _format_mapping(title: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return "{0}: <none>".format(title)
    lines = ["{0}:".format(title)]
    for key, value in sorted(mapping.items()):
        lines.append("- {0} = {1}".format(key, value))
    return "\n".join(lines)


def _format_sequence(title: str, values) -> str:
    if not values:
        return "{0}: <none>".format(title)
    return "{0}: {1}".format(title, ", ".join(values))


class LLMProvider(ABC):
    @abstractmethod
    def implement_function(self, request: ImplementationRequest, model: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def repair_function(self, request: RepairRequest, model: str) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, client: OpenAI = None):
        if client is not None:
            self._client = client
            return
        try:
            self._client = OpenAI()
        except Exception as exc:
            raise ProviderConfigurationError("failed to initialize OpenAI client: {0}".format(exc))

    def implement_function(self, request: ImplementationRequest, model: str) -> str:
        snapshot = request.snapshot
        prompt = "\n\n".join(
            [
                "Implement this Python function.",
                "Module: {0}".format(snapshot.module),
                "Qualified name: {0}".format(snapshot.qualname),
                "Function name: {0}".format(snapshot.name),
                "Signature: {0}{1}".format(snapshot.name, snapshot.signature),
                "Contract:\n{0}".format(snapshot.contract or "<missing contract>"),
                "Current source:\n{0}".format(snapshot.source),
                _format_sequence("Declared state fields", snapshot.state_fields),
                _format_mapping("Observed object state", snapshot.state_summary),
                _format_mapping("Observed object state schema", snapshot.state_schema),
                _format_mapping("Referenced globals", snapshot.globals_summary),
                _format_mapping("Closure summary", snapshot.closure_summary),
            ]
        )
        return self._complete(prompt, model, IMPLEMENT_SYSTEM_PROMPT, CodeGenerationError)

    def repair_function(self, request: RepairRequest, model: str) -> str:
        snapshot = request.snapshot
        prompt = "\n\n".join(
            [
                "Repair this failing Python function.",
                "Module: {0}".format(snapshot.module),
                "Qualified name: {0}".format(snapshot.qualname),
                "Function name: {0}".format(snapshot.name),
                "Signature: {0}{1}".format(snapshot.name, snapshot.signature),
                "Contract:\n{0}".format(snapshot.contract or "<missing contract>"),
                "Current source:\n{0}".format(snapshot.source),
                _format_sequence("Declared state fields", snapshot.state_fields),
                _format_mapping("Observed object state", snapshot.state_summary),
                _format_mapping("Observed object state schema", snapshot.state_schema),
                _format_mapping("Referenced globals", snapshot.globals_summary),
                _format_mapping("Closure summary", snapshot.closure_summary),
                _format_mapping("Call summary", request.call_summary),
                "Error type: {0}".format(request.error_type),
                "Error message: {0}".format(request.error_message),
                "Traceback:\n{0}".format(request.traceback_text),
            ]
        )
        return self._complete(prompt, model, REPAIR_SYSTEM_PROMPT, CodeRepairError)

    def _complete(self, prompt: str, model: str, instructions: str, error_cls):
        response = self._client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise error_cls("Model response did not contain text output")
        return extract_python_source(text)
