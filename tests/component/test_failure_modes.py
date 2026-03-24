import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import CodeRepairError, RuntimePolicyError, StateMutationError
from paithon.provider import LLMProvider
from tests.support.files import read_jsonl


class FailureModeProvider(LLMProvider):
    def __init__(self, implementations=None, repairs=None):
        self.implementations = implementations or {}
        self.repairs = repairs or {}
        self.calls = []

    def implement_function(self, request, model):
        self.calls.append(("implement", request.snapshot.name, model))
        return self.implementations[request.snapshot.name]

    def repair_function(self, request, model):
        self.calls.append(("repair", request.snapshot.name, request.error_type, model))
        repair = self.repairs[request.snapshot.name]
        if isinstance(repair, Exception):
            raise repair
        return repair


def test_rollback_removes_new_attributes_added_before_failed_heal(tmp_path):
    provider = FailureModeProvider(
        repairs={
            "annotate": (
                "def annotate(self):\n"
                "    self.counter += 1\n"
                "    return self.counter\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Counter:
        def __init__(self):
            self.counter = 0

        @engine.self_healing(
            mutable_state_fields=["counter"],
            rollback_on_failure=True,
        )
        def annotate(self):
            """Increment counter without leaving failed scratch attributes behind."""
            self.scratch = "broken"
            self.counter += 5
            return missing_name

    counter = Counter()
    assert counter.annotate() == 1
    assert counter.counter == 1
    assert not hasattr(counter, "scratch")


def test_runtime_implemented_without_healing_reraises_generated_failure(tmp_path):
    provider = FailureModeProvider(
        implementations={
            "mean": (
                "def mean(values):\n"
                "    return sum(values) / len(value)\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    mean = engine.create_function(
        "mean",
        "(values)",
        "Return the arithmetic mean of a non-empty iterable of numbers.",
        heal_errors=False,
    )

    with pytest.raises(NameError):
        mean([1, 2, 3])
    assert provider.calls == [("implement", "mean", "gpt-5-mini")]


def test_self_healing_respects_max_attempts_when_repaired_source_keeps_failing(tmp_path):
    provider = FailureModeProvider(
        repairs={
            "mean": (
                "def mean(values):\n"
                "    return sum(values) / len(value)\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path, max_heal_attempts=1))

    @engine.self_healing(max_attempts=1)
    def mean(values):
        """Return the arithmetic mean."""
        return sum(values) / len(value)

    with pytest.raises(NameError):
        mean([1, 2, 3])
    assert provider.calls == [("repair", "mean", "NameError", "gpt-5-mini")]


def test_mutation_policy_error_can_trigger_a_repair_when_allowed(tmp_path):
    provider = FailureModeProvider(
        implementations={
            "activate": (
                "def activate(self):\n"
                "    self.status = 'active'\n"
                "    self.history = ['activated']\n"
                "    return self.status\n"
            )
        },
        repairs={
            "activate": (
                "def activate(self):\n"
                "    self.status = 'active'\n"
                "    return self.status\n"
            )
        },
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Wallet:
        def __init__(self):
            self.status = "new"

        @engine.runtime_implemented(
            state_fields=["status"],
            mutable_state_fields=["status"],
            rollback_on_failure=True,
        )
        def activate(self):
            """Set the wallet status to active without mutating unrelated fields."""
            raise NotImplementedError

    wallet = Wallet()
    assert wallet.activate() == "active"
    assert wallet.status == "active"
    assert not hasattr(wallet, "history")
    assert ("repair", "activate", "StateMutationError", "gpt-5-mini") in provider.calls


def test_mutation_policy_error_does_not_trigger_repair_when_filtered_out(tmp_path):
    provider = FailureModeProvider(
        implementations={
            "activate": (
                "def activate(self):\n"
                "    self.status = 'active'\n"
                "    self.history = ['activated']\n"
                "    return self.status\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Wallet:
        def __init__(self):
            self.status = "new"

        @engine.runtime_implemented(
            state_fields=["status"],
            mutable_state_fields=["status"],
            rollback_on_failure=True,
            heal_on=(ValueError,),
        )
        def activate(self):
            """Set the wallet status to active without mutating unrelated fields."""
            raise NotImplementedError

    wallet = Wallet()
    with pytest.raises(StateMutationError):
        wallet.activate()
    assert wallet.status == "new"
    assert provider.calls == [("implement", "activate", "gpt-5-mini")]


def test_review_first_regenerates_when_matching_cache_file_is_corrupted(tmp_path):
    provider = FailureModeProvider(
        implementations={
            "slugify": (
                "def slugify(text):\n"
                "    return text.lower()\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    template = engine._make_placeholder_function(
        name="slugify",
        signature="(text)",
        contract="Convert text to lowercase.",
        globals_dict={"__name__": "__paithon_dynamic__"},
    )
    state = engine._build_state(template, "Convert text to lowercase.", mode="implement", cache_by_class=False)
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
    (tmp_path / "{0}.json".format(cache_key)).write_text("{broken", encoding="utf-8")

    slugify = engine.create_function("slugify", "(text)", "Convert text to lowercase.", cache_by_class=False)

    assert slugify("Hello") == "hello"
    assert provider.calls == [("implement", "slugify", "gpt-5-mini")]


def test_production_locked_with_corrupted_cache_still_blocks_fresh_generation(tmp_path):
    build_engine = RuntimeEngine(
        provider=FailureModeProvider(),
        config=RuntimeConfig(cache_dir=tmp_path, operating_mode="production_locked"),
    )
    template = build_engine._make_placeholder_function(
        name="slugify",
        signature="(text)",
        contract="Convert text to lowercase.",
        globals_dict={"__name__": "__paithon_dynamic__"},
    )
    state = build_engine._build_state(template, "Convert text to lowercase.", mode="implement", cache_by_class=False)
    cache_key = build_engine._build_key(
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
    (tmp_path / "{0}.json".format(cache_key)).write_text("{broken", encoding="utf-8")

    slugify = build_engine.create_function("slugify", "(text)", "Convert text to lowercase.", cache_by_class=False)

    with pytest.raises(RuntimePolicyError):
        slugify("Hello")


def test_rollback_audit_includes_field_level_state_diff(tmp_path):
    provider = FailureModeProvider(
        repairs={
            "step": (
                "def step(self):\n"
                "    self.counter += 1\n"
                "    self.history.append('healed')\n"
                "    return self.counter\n"
            )
        }
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Counter:
        def __init__(self):
            self.counter = 0
            self.history = []

        @engine.self_healing(
            rollback_on_failure=True,
            rollback_fields=["counter", "history"],
        )
        def step(self):
            """Increment a counter and append history."""
            self.counter += 5
            self.history.append("broken")
            return missing_name

    counter = Counter()
    assert counter.step() == 1

    audit_events = read_jsonl(tmp_path / "audit.jsonl")
    rollback_event = next(event for event in audit_events if event["event"] == "rollback_applied")
    assert rollback_event["error_type"] == "NameError"
    assert rollback_event["state_diff"]["counter"]["before"] == 0
    assert rollback_event["state_diff"]["counter"]["after"] == 5
    assert rollback_event["state_diff"]["history"]["before"] == []
    assert rollback_event["state_diff"]["history"]["after"] == ["broken"]
