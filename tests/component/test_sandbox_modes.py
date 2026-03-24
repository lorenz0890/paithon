import subprocess
from pathlib import Path

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import SafetyViolationError
from paithon.provider import LLMProvider


class SandboxProvider(LLMProvider):
    def __init__(self, implementations=None):
        self.implementations = implementations or {}

    def implement_function(self, request, model):
        return self.implementations[request.snapshot.name]

    def repair_function(self, request, model):
        raise AssertionError("unexpected repair_function call")


def test_subprocess_restricted_probe_passes_isolated_cwd_and_limits(tmp_path, monkeypatch):
    observed = {}
    engine = RuntimeEngine(
        provider=SandboxProvider(),
        config=RuntimeConfig(
            cache_dir=tmp_path,
            execution_mode="subprocess_restricted",
            sandbox_timeout_seconds=2.0,
            sandbox_memory_limit_mb=64,
            sandbox_file_size_limit_bytes=4096,
            sandbox_max_open_files=16,
        ),
    )

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine._probe_source_if_needed("def slugify(text):\n    return text.lower()\n", "slugify")

    assert observed["cmd"][1:3] == ["-I", "-B"]
    assert observed["kwargs"]["check"] is True
    assert observed["kwargs"]["cwd"]
    assert observed["kwargs"]["env"]["HOME"] == observed["kwargs"]["cwd"]
    assert observed["kwargs"]["env"]["TMPDIR"] == observed["kwargs"]["cwd"]
    assert callable(observed["kwargs"]["preexec_fn"])


def test_subprocess_probe_uses_configured_working_directory_without_restriction(tmp_path, monkeypatch):
    observed = {}
    sandbox_dir = tmp_path / "sandbox"
    engine = RuntimeEngine(
        provider=SandboxProvider(),
        config=RuntimeConfig(
            cache_dir=tmp_path,
            execution_mode="subprocess_probe",
            sandbox_working_dir=sandbox_dir,
        ),
    )

    def fake_run(cmd, **kwargs):
        observed["cmd"] = cmd
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    engine._probe_source_if_needed("def slugify(text):\n    return text.lower()\n", "slugify")

    assert observed["kwargs"]["cwd"] == str(sandbox_dir)
    assert "preexec_fn" not in observed["kwargs"]
    assert Path(observed["kwargs"]["cwd"]).exists()


def test_subprocess_restricted_blocks_definition_time_failures(tmp_path):
    provider = SandboxProvider(
        implementations={
            "boom": (
                "def boom(x=(1 / 0)):\n"
                "    return x\n"
            )
        }
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path, execution_mode="subprocess_restricted"),
    )
    boom = engine.create_function("boom", "()", "Return a value.")

    with pytest.raises(SafetyViolationError):
        boom()
