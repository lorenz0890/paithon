import json
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import ReviewPromotionError, SafetyViolationError, StateRollbackError
from paithon.provider import LLMProvider
from tests.support.files import load_module_from_path


class ImprovementProvider(LLMProvider):
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


class Status(Enum):
    ACTIVE = "active"


Coordinates = namedtuple("Coordinates", ["lat", "lon"])


@dataclass
class NestedProfile:
    name: str
    created_at: datetime
    home: Path
    score: Decimal
    tag_ids: list


class FakeColumn:
    def __init__(self, name):
        self.name = name


class FakeTable:
    def __init__(self, names):
        self.columns = [FakeColumn(name) for name in names]


class FakeSqlAlchemyRow:
    __table__ = FakeTable(["id", "name", "status"])

    def __init__(self, identifier, name, status):
        self.id = identifier
        self.name = name
        self.status = status


class FakeStateful:
    def __init__(self, value):
        self.value = value
        self.training = True

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, payload):
        self.value = payload["value"]

    def train(self, mode=True):
        self.training = bool(mode)
        return self


class Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("cannot deepcopy")


class CustomCounter:
    def __init__(self, count):
        self.count = count

    def __deepcopy__(self, memo):
        raise RuntimeError("use custom snapshot strategy")


class FakeDataFrame:
    __module__ = "pandas.core.frame"

    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.columns = list(self._rows[0].keys()) if self._rows else []
        self.shape = (len(self._rows), len(self.columns))
        self.dtypes = {column: type(self._rows[0][column]).__name__ for column in self.columns} if self._rows else {}

    def copy(self, deep=True):
        return FakeDataFrame(self._rows)

    def equals(self, other):
        return self._rows == other._rows

    def head(self, limit):
        return FakeDataFrame(self._rows[:limit])

    def to_dict(self, orient="records"):
        assert orient == "records"
        return [dict(row) for row in self._rows]


class FakeKerasModel:
    __module__ = "keras.engine.training"

    def __init__(self, weights, *, trainable=True):
        self._weights = list(weights)
        self.trainable = trainable

    def get_weights(self):
        return list(self._weights)

    def set_weights(self, weights):
        self._weights = list(weights)

def test_richer_serializers_capture_nested_domain_objects_and_orm_state(tmp_path):
    provider = ImprovementProvider()
    provider.implementations["render"] = (
        "def render(self):\n"
        "    return self.record.name.upper()\n"
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path, max_state_depth=3, max_collection_items=2),
    )

    class View:
        def __init__(self):
            self.profile = NestedProfile(
                name="Ada",
                created_at=datetime(2025, 1, 2, 3, 4, 5),
                home=Path("/tmp/demo"),
                score=Decimal("9.5"),
                tag_ids=[11, 12, 13],
            )
            self.record = FakeSqlAlchemyRow(1, "ada", Status.ACTIVE)
            self.location = Coordinates(48.2, 16.3)
            self.identifier = UUID("12345678-1234-5678-1234-567812345678")

        @engine.runtime_implemented(state_fields=["profile", "record", "location", "identifier"])
        def render(self):
            """Render the record name."""
            raise NotImplementedError

    view = View()
    assert view.render() == "ADA"

    snapshot = provider.implement_requests[0][0].snapshot
    assert snapshot.state_schema["profile"] == "dataclass:NestedProfile(name, created_at, home, score, tag_ids)"
    assert "created_at" in snapshot.state_summary["profile"]
    assert "<truncated_items>" in snapshot.state_summary["profile"]
    assert "<total_items>" in snapshot.state_summary["profile"]
    assert snapshot.state_schema["record"] == "sqlalchemy:FakeSqlAlchemyRow(id, name, status)"
    assert "<truncated_items>" in snapshot.state_summary["record"]
    assert snapshot.state_schema["location"] == "namedtuple:Coordinates(lat, lon)"
    assert snapshot.state_schema["identifier"] == "uuid:UUID"


def test_rollback_fields_restore_state_dict_objects_before_healed_retry(tmp_path):
    provider = ImprovementProvider()
    provider.repairs["step"] = (
        "def step(self):\n"
        "    self.model.value += 1\n"
        "    self.optimizer.value += 1\n"
        "    self.history.append('healed')\n"
        "    return self.model.value\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Trainer:
        def __init__(self):
            self.model = FakeStateful(10)
            self.optimizer = FakeStateful(100)
            self.history = []

        @engine.self_healing(
            state_fields=["history"],
            rollback_on_failure=True,
            rollback_fields=["model", "optimizer", "history"],
        )
        def step(self):
            """Advance the model by one step and record history."""
            self.model.value += 5
            self.optimizer.value += 7
            self.history.append("broken")
            return missing_name

    trainer = Trainer()
    assert trainer.step() == 11
    assert trainer.model.value == 11
    assert trainer.optimizer.value == 101
    assert trainer.history == ["healed"]


def test_strict_rollback_raises_for_unsupported_snapshot_fields(tmp_path):
    provider = ImprovementProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Container:
        def __init__(self):
            self.payload = Uncopyable()

        @engine.self_healing(
            rollback_on_failure=True,
            rollback_fields=["payload"],
            strict_rollback=True,
            heal_on=(),
        )
        def run(self):
            """Touch the payload."""
            return 1

    container = Container()
    with pytest.raises(StateRollbackError):
        container.run()


def test_custom_snapshot_strategy_restores_domain_object(tmp_path):
    provider = ImprovementProvider()
    provider.repairs["step"] = (
        "def step(self):\n"
        "    self.counter.count += 1\n"
        "    return self.counter.count\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    engine.register_snapshot_strategy(
        CustomCounter,
        capture=lambda value: ({"count": value.count}, {"count": value.count}),
        restore=lambda current, snapshot: _restore_custom_counter(current, snapshot),
        compare=lambda current, snapshot: current.count == snapshot.compare_payload["count"],
        name="custom-counter",
        first=True,
    )

    class CounterBox:
        def __init__(self):
            self.counter = CustomCounter(10)

        @engine.self_healing(rollback_on_failure=True, rollback_fields=["counter"])
        def step(self):
            """Increment a counter."""
            self.counter.count += 7
            return broken_name

    box = CounterBox()
    assert box.step() == 11
    assert box.counter.count == 11


def test_redaction_policy_hides_secret_state_and_prompt_context(tmp_path):
    provider = ImprovementProvider()
    provider.repairs["render"] = (
        "def render(self, token):\n"
        "    return self.profile['name'] + ':' + token\n"
    )
    api_secret = "global-secret"
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(
            cache_dir=tmp_path,
            redacted_field_names=("token",),
            redacted_field_patterns=("secret", "password"),
        ),
    )

    class View:
        def __init__(self):
            self.profile = {"name": "Ada", "password_hash": "pw"}

        @engine.self_healing(state_fields=["profile"])
        def render(self, token):
            """Render a name with a token and access surrounding secrets."""
            return "{0}:{1}:{2}".format(api_secret, self.profile["password_hash"], missing_name)

    view = View()
    assert view.render("visible") == "Ada:visible"

    snapshot = provider.repair_requests[0][0].snapshot
    assert "<redacted>" in snapshot.state_summary["profile"]
    assert "pw" not in snapshot.state_summary["profile"]
    assert snapshot.closure_summary["api_secret"] == "<redacted>"
    assert provider.repair_requests[0][0].call_summary["token"] == "<redacted>"


def test_rollback_fields_restore_pandas_like_objects_before_healed_retry(tmp_path):
    provider = ImprovementProvider()
    provider.repairs["step"] = (
        "def step(self):\n"
        "    self.frame._rows[0]['score'] += 1\n"
        "    self.events.append('healed')\n"
        "    return self.frame._rows[0]['score']\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Trainer:
        def __init__(self):
            self.frame = FakeDataFrame([{"name": "Ada", "score": 9}])
            self.events = []

        @engine.self_healing(
            rollback_on_failure=True,
            rollback_fields=["frame", "events"],
        )
        def step(self):
            """Update a frame-backed metric."""
            self.frame._rows[0]["score"] = 20
            self.events.append("broken")
            return missing_name

    trainer = Trainer()
    assert trainer.step() == 10
    assert trainer.frame._rows == [{"name": "Ada", "score": 10}]
    assert trainer.events == ["healed"]


def test_redaction_policy_changes_cache_lineage(tmp_path):
    provider_a = ImprovementProvider()
    provider_a.implementations["slugify"] = (
        "def slugify(text):\n"
        "    return text.lower()\n"
    )
    engine_a = RuntimeEngine(provider=provider_a, config=RuntimeConfig(cache_dir=tmp_path))
    slugify_a = engine_a.create_function("slugify", "(text)", "Convert text to lowercase.", cache_by_class=False)

    assert slugify_a("Hello") == "hello"
    assert len(provider_a.implement_requests) == 1

    provider_b = ImprovementProvider()
    provider_b.implementations["slugify"] = (
        "def slugify(text):\n"
        "    return text.upper()\n"
    )
    engine_b = RuntimeEngine(
        provider=provider_b,
        config=RuntimeConfig(cache_dir=tmp_path, redacted_field_names=("token",)),
    )
    slugify_b = engine_b.create_function("slugify", "(text)", "Convert text to lowercase.", cache_by_class=False)

    assert slugify_b("Hello") == "HELLO"
    assert len(provider_b.implement_requests) == 1


def test_path_redaction_and_custom_placeholder_flow_into_runtime_requests(tmp_path):
    provider = ImprovementProvider()
    provider.repairs["render"] = (
        "def render(self, payload):\n"
        "    return payload['headers']['authorization']\n"
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(
            cache_dir=tmp_path,
            redacted_field_paths=("profile.credentials.token", "payload.headers.authorization"),
            redaction_placeholder="<hidden>",
        ),
    )

    class View:
        def __init__(self):
            self.profile = {"credentials": {"token": "abc", "name": "Ada"}}

        @engine.self_healing(state_fields=["profile"])
        def render(self, payload):
            """Use profile credentials and payload headers."""
            return payload["headers"]["authorization"] + missing_name

    view = View()
    assert view.render({"headers": {"authorization": "Bearer secret"}}) == "Bearer secret"

    request = provider.repair_requests[0][0]
    assert "<hidden>" in request.snapshot.state_summary["profile"]
    assert "abc" not in request.snapshot.state_summary["profile"]
    assert request.call_summary["payload"].count("<hidden>") >= 1


def test_rollback_fields_restore_keras_like_objects_before_healed_retry(tmp_path):
    provider = ImprovementProvider()
    provider.repairs["step"] = (
        "def step(self):\n"
        "    weights = self.model.get_weights()\n"
        "    weights[0] += 1.0\n"
        "    self.model.set_weights(weights)\n"
        "    return self.model.get_weights()[0]\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    class Trainer:
        def __init__(self):
            self.model = FakeKerasModel([1.0, 2.0], trainable=True)

        @engine.self_healing(
            rollback_on_failure=True,
            rollback_fields=["model"],
        )
        def step(self):
            """Update model weights safely."""
            self.model.set_weights([9.0, 10.0])
            self.model.trainable = False
            return missing_name

    trainer = Trainer()
    assert trainer.step() == 2.0
    assert trainer.model.get_weights() == [2.0, 2.0]
    assert trainer.model.trainable is True


def test_review_promotion_writes_reviewed_source_back_to_file(tmp_path):
    source_path = tmp_path / "review_module.py"
    source_path.write_text(
        "def slug(text):\n"
        "    return text.lowr()\n",
        encoding="utf-8",
    )
    module = load_module_from_path(source_path)
    provider = ImprovementProvider()
    provider.repairs["slug"] = (
        "def slug(text):\n"
        "    return text.lower().replace(' ', '-')\n"
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path / "cache"))
    wrapped = engine.self_healing(module.slug, contract="Convert text into a lowercase slug.")

    assert wrapped("Hello World") == "hello-world"
    manifest_path = engine.export_review_artifacts(tmp_path / "review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_source_path = Path(manifest[0]["source_file"])
    review_source_path.write_text(
        "def slug(text):\n"
        "    return text.lower().replace(' ', '_')\n",
        encoding="utf-8",
    )

    promoted = engine.promote_review_artifacts(manifest_path)

    assert manifest[0]["cache_key"] in promoted
    assert "return text.lower().replace(' ', '_')" in source_path.read_text(encoding="utf-8")


def test_promote_review_artifacts_requires_manifest(tmp_path):
    engine = RuntimeEngine(provider=ImprovementProvider(), config=RuntimeConfig(cache_dir=tmp_path))
    with pytest.raises(ReviewPromotionError):
        engine.promote_review_artifacts(tmp_path / "missing-review-dir")


def test_subprocess_probe_blocks_definition_time_failures(tmp_path):
    provider = ImprovementProvider()
    provider.implementations["boom"] = (
        "def boom(x=(1 / 0)):\n"
        "    return x\n"
    )
    engine = RuntimeEngine(
        provider=provider,
        config=RuntimeConfig(cache_dir=tmp_path, execution_mode="subprocess_probe"),
    )
    boom = engine.create_function("boom", "()", "Return a value.")

    with pytest.raises(SafetyViolationError):
        boom()


def _restore_custom_counter(current, snapshot):
    target = current if current is not None else snapshot.original_value
    target.count = snapshot.restore_payload["count"]
    return target
