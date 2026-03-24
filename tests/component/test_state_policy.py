from dataclasses import dataclass
from pathlib import Path

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import SafetyViolationError, StateMutationError
from paithon.provider import LLMProvider


@dataclass
class Profile:
    name: str
    score: int


@dataclass
class Address:
    city: str
    zip_code: str


class PolicyProvider(LLMProvider):
    def __init__(self):
        self.implement_requests = []
        self.repair_requests = []
        self.implementations = {}
        self.repairs = {}

    def implement_function(self, request, model):
        self.implement_requests.append((request, model))
        return self.implementations[request.snapshot.name]

    def repair_function(self, request, model):
        self.repair_requests.append((request, model))
        return self.repairs[request.snapshot.name]


def test_state_serializers_capture_summary_and_schema(tmp_path):
    provider = PolicyProvider()
    provider.implementations["render_profile"] = (
        "def render_profile(self):\n"
        "    return self.profile.name.upper()\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    engine.register_state_serializer(
        Address,
        lambda value: ({"city": value.city, "zip": value.zip_code}, "address:{0}".format(value.city)),
        first=True,
    )

    class User:
        def __init__(self):
            self.profile = Profile(name="Ada", score=9)
            self.address = Address(city="Vienna", zip_code="1010")

        @engine.runtime_implemented(state_fields=["profile", "address"])
        def render_profile(self):
            """Render a short profile summary."""
            raise NotImplementedError

    user = User()
    assert user.render_profile() == "ADA"
    request, _ = provider.implement_requests[0]
    assert request.snapshot.state_schema["profile"] == "dataclass:Profile(name, score)"
    assert request.snapshot.state_schema["address"] == "address:Vienna"
    assert request.snapshot.state_summary["profile"] == "{'name': 'Ada', 'score': 9}"
    assert request.snapshot.state_summary["address"] == "{'city': 'Vienna', 'zip': '1010'}"


def test_rollback_on_failure_restores_state_before_healed_retry(tmp_path):
    provider = PolicyProvider()
    provider.repairs["withdraw"] = (
        "def withdraw(self, amount):\n"
        "    self.balance -= amount\n"
        "    return self.balance\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Wallet:
        def __init__(self):
            self.balance = 20
            self.status = "new"

        @engine.self_healing(
            state_fields=["balance", "status"],
            mutable_state_fields=["balance"],
            rollback_on_failure=True,
        )
        def withdraw(self, amount):
            """Decrease balance by amount and return the new balance."""
            self.status = "corrupted"
            self.balance -= amunt
            return self.balance

    wallet = Wallet()
    assert wallet.withdraw(5) == 15
    assert wallet.balance == 15
    assert wallet.status == "new"


def test_mutation_allowlist_rejects_disallowed_successful_mutation(tmp_path):
    provider = PolicyProvider()
    provider.implementations["activate"] = (
        "def activate(self):\n"
        "    self.status = 'active'\n"
        "    return self.status\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Wallet:
        def __init__(self):
            self.status = "new"

        @engine.runtime_implemented(
            state_fields=["status"],
            mutable_state_fields=(),
            rollback_on_failure=True,
        )
        def activate(self):
            """Return the active status."""
            raise NotImplementedError

    wallet = Wallet()
    with pytest.raises(StateMutationError):
        wallet.activate()
    assert wallet.status == "new"


def test_cache_variants_change_with_class_version_and_state_schema(tmp_path):
    provider = PolicyProvider()
    provider.implementations["render"] = (
        "def render(self):\n"
        "    return 'ok'\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Base:
        __paithon_version__ = "v1"

        def __init__(self, payload):
            self.payload = payload

        @engine.runtime_implemented(state_fields=["payload"], cache_by_class=True, contract_revision="r1")
        def render(self):
            """Render the payload."""
            raise NotImplementedError

    class Child(Base):
        __paithon_version__ = "v2"

    assert Base(Profile("Ada", 9)).render() == "ok"
    assert Child(Address("Vienna", "1010")).render() == "ok"
    assert len(provider.implement_requests) == 2
    first_request, _ = provider.implement_requests[0]
    second_request, _ = provider.implement_requests[1]
    assert first_request.snapshot.state_schema["payload"] == "dataclass:Profile(name, score)"
    assert second_request.snapshot.state_schema["payload"] == "dataclass:Address(city, zip_code)"


def test_review_export_writes_generated_source_and_patch(tmp_path):
    provider = PolicyProvider()
    provider.implementations["slug"] = (
        "def slug(text):\n"
        "    return text.lower().replace(' ', '-')\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path / "cache"))
    slug = engine.create_function("slug", "(text)", "Convert text into a lowercase slug.")

    assert slug("Hello World") == "hello-world"
    manifest_path = engine.export_review_artifacts(tmp_path / "review")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    review_dir = manifest_path.parent
    patch_files = list(review_dir.glob("*.patch"))
    source_files = list(review_dir.glob("*.py"))

    assert "slug" in manifest_text
    assert patch_files
    assert source_files
    assert "return text.lower().replace(' ', '-')" in source_files[0].read_text(encoding="utf-8")


def test_basic_safety_guard_blocks_unsafe_generated_code(tmp_path):
    provider = PolicyProvider()
    provider.implementations["unsafe_reader"] = (
        "def unsafe_reader():\n"
        "    import os\n"
        "    return os.listdir('.')\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    unsafe_reader = engine.create_function("unsafe_reader", "()", "Return a directory listing.")

    with pytest.raises(SafetyViolationError):
        unsafe_reader()


def test_unsafe_generated_source_can_be_repaired_against_safety_policy(tmp_path):
    provider = PolicyProvider()
    provider.implementations["configure_reproducibility"] = (
        "def configure_reproducibility(seed=7):\n"
        "    import os\n"
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
        "    return seed\n"
    )
    provider.repairs["configure_reproducibility"] = (
        "def configure_reproducibility(seed=7):\n"
        "    import random\n"
        "    random.seed(seed)\n"
        "    return seed\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    configure_reproducibility = engine.create_function(
        "configure_reproducibility",
        "(seed=7)",
        "Seed Python RNGs without using blocked system imports or mutating process environment variables.",
    )

    assert configure_reproducibility(11) == 11
    assert len(provider.implement_requests) == 1
    assert len(provider.repair_requests) == 1
    repair_request, _ = provider.repair_requests[0]
    assert repair_request.error_type == "SafetyViolationError"
    assert "blocked import in generated code: os" in repair_request.error_message
    assert "import os" in repair_request.snapshot.source
