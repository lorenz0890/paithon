import json

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import SafetyViolationError
from paithon.provider import LLMProvider


class NullProvider(LLMProvider):
    def implement_function(self, request, model):
        raise AssertionError("unexpected implement_function call")

    def repair_function(self, request, model):
        raise AssertionError("unexpected repair_function call")


def build_engine(tmp_path, **config_overrides):
    config = RuntimeConfig(cache_dir=tmp_path, **config_overrides)
    return RuntimeEngine(provider=NullProvider(), config=config)


def test_validate_generated_source_allows_single_function_with_docstring(tmp_path):
    engine = build_engine(tmp_path)

    engine._validate_generated_source(
        '"""module prelude"""\n'
        "def add(x, y):\n"
        "    return x + y\n",
        expected_name="add",
    )


def test_validate_generated_source_rejects_wrong_function_name(tmp_path):
    engine = build_engine(tmp_path)

    with pytest.raises(SafetyViolationError):
        engine._validate_generated_source("def wrong_name(x):\n    return x\n", expected_name="expected_name")


def test_validate_generated_source_rejects_extra_top_level_statements(tmp_path):
    engine = build_engine(tmp_path)

    with pytest.raises(SafetyViolationError):
        engine._validate_generated_source(
            "def add(x, y):\n"
            "    return x + y\n"
            "value = 3\n",
            expected_name="add",
        )


def test_validate_generated_source_blocks_import_from_blocked_module(tmp_path):
    engine = build_engine(tmp_path)

    with pytest.raises(SafetyViolationError):
        engine._validate_generated_source(
            "def read_env(name):\n"
            "    from os import environ\n"
            "    return environ.get(name)\n",
            expected_name="read_env",
        )


def test_validate_generated_source_blocks_blocked_builtin_call(tmp_path):
    engine = build_engine(tmp_path)

    with pytest.raises(SafetyViolationError):
        engine._validate_generated_source(
            "def read_text(path):\n"
            "    return open(path).read()\n",
            expected_name="read_text",
        )


def test_build_placeholder_source_requires_parenthesized_signature(tmp_path):
    engine = build_engine(tmp_path)

    with pytest.raises(ValueError):
        engine._build_placeholder_source("slugify", "text", "Convert text into a slug.")


@pytest.mark.parametrize(
    ("operating_mode", "expected_status"),
    [
        ("review_first", "pending_review"),
        ("development", "development"),
        ("production_locked", "approved"),
    ],
)
def test_default_approval_status_depends_on_operating_mode(tmp_path, operating_mode, expected_status):
    engine = build_engine(tmp_path, operating_mode=operating_mode)

    assert engine._default_approval_status() == expected_status


def test_approval_status_falls_back_to_legacy_untracked(tmp_path):
    engine = build_engine(tmp_path)

    assert engine._approval_status({}) == "legacy_untracked"


def test_normalize_cache_payload_backfills_legacy_metadata_and_persists_it(tmp_path):
    engine = build_engine(tmp_path)
    cache_key = "legacy"
    payload = {
        "qualname": "demo_fn",
        "source": "def demo_fn():\n    return 1\n",
    }

    normalized = engine._normalize_cache_payload(payload, cache_key=cache_key)

    assert normalized["approval_status"] == "legacy_untracked"
    assert normalized["operating_mode"] == "legacy"
    persisted = json.loads((tmp_path / "legacy.json").read_text(encoding="utf-8"))
    assert persisted["approval_status"] == "legacy_untracked"
    assert persisted["operating_mode"] == "legacy"


def test_needs_polyfill_detects_missing_dependency_and_platform_mismatch(tmp_path):
    engine = build_engine(tmp_path)

    assert engine._needs_polyfill(dependency="json", platforms=None) is False
    assert engine._needs_polyfill(dependency="paithon_missing_dependency", platforms=None) is True
    assert engine._needs_polyfill(dependency=None, platforms=("definitely-not-this-platform",)) is True
