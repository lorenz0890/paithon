import ast
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from typing import Any, Callable, Optional

from ..exceptions import SafetyViolationError
from ..provider import extract_python_source
from .constants import BLOCKED_CALLS, BLOCKED_IMPORTS


class RuntimeSourceMixin:
    def _probe_source_if_needed(self, source: str, expected_name: str) -> None:
        if self.config.execution_mode not in {"subprocess_probe", "subprocess_restricted"}:
            return
        payload = json.dumps({"name": expected_name, "source": source})
        script = textwrap.dedent(
            """
            import json
            import sys

            payload = json.loads(sys.stdin.read())
            namespace = {"__name__": "__paithon_probe__"}
            exec(compile(payload["source"], "<paithon-probe>", "exec"), namespace, namespace)
            if not callable(namespace.get(payload["name"])):
                raise RuntimeError("generated source did not define the expected callable")
            """
        )
        executable = self.config.sandbox_python_executable or sys.executable
        temporary_dir = None
        cwd = self._sandbox_working_dir()
        if cwd is None and self.config.execution_mode == "subprocess_restricted":
            temporary_dir = tempfile.TemporaryDirectory(prefix="paithon-sandbox-")
            cwd = Path(temporary_dir.name)
        preexec_fn = self._sandbox_preexec_fn()
        try:
            run_kwargs = {
                "input": payload,
                "text": True,
                "capture_output": True,
                "timeout": self.config.sandbox_timeout_seconds,
                "check": True,
                "env": self._sandbox_environment(cwd),
            }
            if cwd is not None:
                run_kwargs["cwd"] = str(cwd)
            if preexec_fn is not None:
                run_kwargs["preexec_fn"] = preexec_fn
            subprocess.run(
                [executable, "-I", "-B", "-c", script],
                **run_kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            raise SafetyViolationError("generated source timed out in subprocess probe") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise SafetyViolationError(
                "generated source failed subprocess probe: {0}".format(detail or "unknown subprocess error")
            ) from exc
        finally:
            if temporary_dir is not None:
                temporary_dir.cleanup()

    @staticmethod
    def _validate_generated_source(source: str, expected_name: Optional[str] = None) -> None:
        tree = ast.parse(source)
        top_level_nodes = list(tree.body)
        allowed_prelude = 0
        if top_level_nodes and isinstance(top_level_nodes[0], ast.Expr):
            value = getattr(top_level_nodes[0], "value", None)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                allowed_prelude = 1
        body_nodes = top_level_nodes[allowed_prelude:]
        if len(body_nodes) != 1 or not isinstance(body_nodes[0], ast.FunctionDef):
            raise SafetyViolationError("generated source must contain exactly one top-level function definition")
        if expected_name is not None and body_nodes[0].name != expected_name:
            raise SafetyViolationError(
                "generated source defined {0} instead of {1}".format(body_nodes[0].name, expected_name)
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in BLOCKED_IMPORTS:
                        raise SafetyViolationError("blocked import in generated code: {0}".format(root))
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in BLOCKED_IMPORTS:
                    raise SafetyViolationError("blocked import in generated code: {0}".format(root))
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                    raise SafetyViolationError("blocked call in generated code: {0}".format(node.func.id))

    @staticmethod
    def _build_placeholder_source(name: str, signature: str, contract: str) -> str:
        if not signature.startswith("("):
            raise ValueError("signature must start with '('")
        doc_literal = repr(contract or "Runtime-generated function contract.")
        return (
            "def {0}{1}:\n"
            "    {2}\n"
            "    raise NotImplementedError\n"
        ).format(name, signature, doc_literal)

    def _make_placeholder_function(
        self,
        name: str,
        signature: str,
        contract: str,
        globals_dict: dict,
    ) -> Callable[..., Any]:
        source = self._build_placeholder_source(name, signature, contract)
        return self._compile_template_function(source, name, globals_dict)

    def _make_placeholder_clone(self, func: Callable[..., Any], contract: str) -> Callable[..., Any]:
        source = self._build_placeholder_source(func.__name__, str(inspect.signature(func)), contract)
        placeholder = self._compile_template_function(source, func.__name__, func.__globals__)
        placeholder.__module__ = func.__module__
        placeholder.__qualname__ = func.__qualname__
        placeholder.__annotations__ = dict(getattr(func, "__annotations__", {}))
        placeholder.__defaults__ = func.__defaults__
        placeholder.__kwdefaults__ = func.__kwdefaults__
        placeholder.__dict__.update(getattr(func, "__dict__", {}))
        placeholder.__paithon_source__ = source
        return placeholder

    @staticmethod
    def _compile_template_function(source: str, name: str, globals_dict: dict) -> Callable[..., Any]:
        globals_dict.setdefault("__builtins__", __builtins__)
        module_name = globals_dict.get("__name__", "__paithon_dynamic__")
        temp_namespace = {"__name__": module_name}
        filename = "<paithon-template:{0}.{1}>".format(module_name, name)
        exec(compile(source, filename, "exec"), temp_namespace, temp_namespace)
        template = temp_namespace[name]
        function = types.FunctionType(template.__code__, globals_dict, name, template.__defaults__, None)
        function.__kwdefaults__ = template.__kwdefaults__
        function.__annotations__ = dict(getattr(template, "__annotations__", {}))
        function.__doc__ = template.__doc__
        function.__module__ = module_name
        function.__qualname__ = getattr(template, "__qualname__", name)
        function.__paithon_source__ = source
        return function

    def _sandbox_environment(self, cwd: Optional[Path]) -> dict:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        system_root = os.environ.get("SYSTEMROOT")
        if system_root:
            env["SYSTEMROOT"] = system_root
        if cwd is not None:
            env["HOME"] = str(cwd)
            env["TMPDIR"] = str(cwd)
        return env

    def _sandbox_working_dir(self) -> Optional[Path]:
        if self.config.sandbox_working_dir is None:
            return None
        path = Path(self.config.sandbox_working_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _sandbox_preexec_fn(self):
        if self.config.execution_mode != "subprocess_restricted" or os.name == "nt":
            return None
        try:
            import resource
        except ImportError:  # pragma: no cover - platform dependent
            return None

        cpu_limit = max(1, int(math.ceil(self.config.sandbox_timeout_seconds)))
        memory_limit = (
            None
            if self.config.sandbox_memory_limit_mb is None
            else int(self.config.sandbox_memory_limit_mb) * 1024 * 1024
        )
        file_limit = self.config.sandbox_file_size_limit_bytes
        open_file_limit = self.config.sandbox_max_open_files

        def apply_limits() -> None:
            try:
                os.setsid()
            except Exception:
                pass
            for resource_name, limit in (
                ("RLIMIT_CORE", 0),
                ("RLIMIT_CPU", cpu_limit),
                ("RLIMIT_FSIZE", file_limit),
                ("RLIMIT_NOFILE", open_file_limit),
            ):
                if limit is None:
                    continue
                resource_id = getattr(resource, resource_name, None)
                if resource_id is None:
                    continue
                try:
                    resource.setrlimit(resource_id, (limit, limit))
                except (OSError, ValueError):
                    continue
            if memory_limit is not None:
                for resource_name in ("RLIMIT_AS", "RLIMIT_DATA"):
                    resource_id = getattr(resource, resource_name, None)
                    if resource_id is None:
                        continue
                    try:
                        resource.setrlimit(resource_id, (memory_limit, memory_limit))
                    except (OSError, ValueError):
                        continue

        return apply_limits
