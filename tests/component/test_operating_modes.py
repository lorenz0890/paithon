import json
from pathlib import Path

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import RuntimePolicyError
from paithon.provider import LLMProvider
from tests.support.files import load_module_from_path, read_jsonl


class ModeProvider(LLMProvider):
    def __init__(self, implementations=None, repairs=None):
        self.implementations = implementations or {}
        self.repairs = repairs or {}
        self.calls = []

    def implement_function(self, request, model):
        self.calls.append(("implement", request.snapshot.name, model))
        return self.implementations[request.snapshot.name]

    def repair_function(self, request, model):
        self.calls.append(("repair", request.snapshot.name, request.error_type, model))
        return self.repairs[request.snapshot.name]

def test_review_first_is_default_and_marks_cache_pending_review(tmp_path):
    provider = ModeProvider(implementations={"add": "def add(x, y):\n    return x + y\n"})
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    add = engine.create_function("add", "(x, y)", "Return the sum of two numbers.")

    assert add(2, 3) == 5

    cache_files = list(tmp_path.glob("*.json"))
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["approval_status"] == "pending_review"
    audit_events = read_jsonl(tmp_path / "audit.jsonl")
    assert any(event["event"] == "generated" for event in audit_events)
    assert any(event["event"] == "cache_saved" and event["approval_status"] == "pending_review" for event in audit_events)


def test_approved_cache_entry_can_be_loaded_in_production_locked_mode(tmp_path):
    provider = ModeProvider(implementations={"add": "def add(x, y):\n    return x + y\n"})
    review_engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path, operating_mode="review_first"))
    add = review_engine.create_function("add", "(x, y)", "Return the sum of two numbers.")

    assert add(4, 5) == 9
    cache_key = next(path.stem for path in tmp_path.glob("*.json") if path.name != "audit.jsonl")
    review_engine.approve_cache_entry(cache_key, reviewer="test")

    locked_provider = ModeProvider()
    locked_engine = RuntimeEngine(
        provider=locked_provider,
        config=RuntimeConfig(cache_dir=tmp_path, operating_mode="production_locked"),
    )
    locked_add = locked_engine.create_function("add", "(x, y)", "Return the sum of two numbers.")

    assert locked_add(1, 2) == 3
    assert locked_provider.calls == []


def test_production_locked_blocks_fresh_runtime_generation(tmp_path):
    engine = RuntimeEngine(
        provider=ModeProvider(implementations={"slugify": "def slugify(text):\n    return text.lower()\n"}),
        config=RuntimeConfig(cache_dir=tmp_path, operating_mode="production_locked"),
    )
    slugify = engine.create_function("slugify", "(text)", "Convert text to lowercase.")

    with pytest.raises(RuntimePolicyError):
        slugify("Hello")


def test_production_locked_disables_runtime_healing_and_reraises_original_error(tmp_path):
    provider = ModeProvider(repairs={"mean": "def mean(values):\n    return sum(values) / len(values)\n"})
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path, operating_mode="production_locked"),
    )

    @engine.self_healing
    def mean(values):
        """Return the arithmetic mean."""
        return sum(values) / len(value)

    with pytest.raises(NameError):
        mean([1, 2, 3])
    assert provider.calls == []


def test_promoted_runtime_implemented_source_strips_runtime_decorator(tmp_path):
    source_path = tmp_path / "impl_module.py"
    source_path.write_text(
        "def runtime_implemented(*args, **kwargs):\n"
        "    def decorate(func):\n"
        "        return func\n"
        "    return decorate\n\n"
        "@runtime_implemented()\n"
        "def slug(text):\n"
        "    \"\"\"Convert text into a lowercase slug.\"\"\"\n"
        "    raise NotImplementedError\n",
        encoding="utf-8",
    )
    module = load_module_from_path(source_path)
    provider = ModeProvider(
        implementations={
            "slug": (
                "def slug(text):\n"
                "    return text.lower().replace(' ', '-')\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path / "cache"))
    slug = engine.runtime_implemented(module.slug)

    assert slug("Hello World") == "hello-world"
    cache_key = next(path.stem for path in (tmp_path / "cache").glob("*.json") if path.name != "audit.jsonl")
    promoted_path = engine.promote_cache_entry(cache_key)
    promoted_text = promoted_path.read_text(encoding="utf-8")

    assert "@runtime_implemented" not in promoted_text
    assert "return text.lower().replace(' ', '-')" in promoted_text

    promoted_module = load_module_from_path(source_path)
    assert promoted_module.slug("Hello World") == "hello-world"


def test_legacy_cache_entries_are_normalized_to_legacy_untracked(tmp_path):
    provider = ModeProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path, operating_mode="review_first"))
    contract = "Return the sum of two numbers."
    template = engine._make_placeholder_function(
        name="legacy_add",
        signature="(x, y)",
        contract=contract,
        globals_dict={"__name__": "__paithon_dynamic__"},
    )
    state = engine._build_state(
        template,
        contract,
        mode="implement",
        cache_by_class=False,
    )
    cache_key = engine._build_key(
        state.template,
        state.contract,
        state.template_source,
        state.mode,
        state.state_fields,
        state.mutable_state_fields,
        state.rollback_fields,
        state.strict_rollback,
        {},
        state.contract_revision,
    )
    cache_payload = {
        "mode": "implement",
        "module": "__paithon_dynamic__",
        "qualname": "legacy_add",
        "source": "def legacy_add(x, y):\n    return x + y\n",
        "model": "gpt-5-mini",
        "state_fields": [],
        "mutable_state_fields": [],
        "rollback_fields": [],
        "contract": contract,
        "contract_revision": None,
        "template_source": state.template_source,
        "source_path": None,
        "source_lineno": None,
        "context": {},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "{0}.json".format(cache_key)).write_text(json.dumps(cache_payload), encoding="utf-8")

    legacy_add = engine.create_function("legacy_add", "(x, y)", contract, cache_by_class=False)

    assert legacy_add(2, 5) == 7
    assert provider.calls == []

    normalized_payload = json.loads((tmp_path / "{0}.json".format(cache_key)).read_text(encoding="utf-8"))
    assert normalized_payload["approval_status"] == "legacy_untracked"

    audit_events = read_jsonl(tmp_path / "audit.jsonl")
    assert any(event["event"] == "cache_metadata_normalized" and event["approval_status"] == "legacy_untracked" for event in audit_events)
    assert any(event["event"] == "cache_loaded" and event["approval_status"] == "legacy_untracked" for event in audit_events)
