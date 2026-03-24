import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import SafetyViolationError
from paithon.provider import LLMProvider


class DemoProvider(LLMProvider):
    def __init__(self):
        self.implement_requests = []
        self.repair_requests = []

    def implement_function(self, request, model):
        self.implement_requests.append((request, model))
        name = request.snapshot.name
        print("[llm] implement {0} via {1}".format(name, model))
        if name == "render":
            return (
                "def render(self):\n"
                "    return '{0}:{1}'.format(self.record.name.upper(), self.location.city)\n"
            )
        if name == "summarize_job":
            return (
                "def summarize_job(self):\n"
                "    return {\n"
                "        'history': list(self.history),\n"
                "        'model_weight': self.model.weight,\n"
                "        'optimizer_step': self.optimizer.step_count,\n"
                "    }\n"
            )
        if name == "boom":
            return (
                "def boom(x=(1 / 0)):\n"
                "    return x\n"
            )
        raise KeyError(name)

    def repair_function(self, request, model):
        self.repair_requests.append((request, model))
        name = request.snapshot.name
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        if name == "step":
            return (
                "def step(self):\n"
                "    self.model.weight += 1\n"
                "    self.optimizer.step_count += 1\n"
                "    self.history.append('healed')\n"
                "    return self.model.weight\n"
            )
        if name == "slug":
            return (
                "def slug(text):\n"
                "    return text.lower().replace(' ', '-')\n"
            )
        raise KeyError(name)


class Status(Enum):
    ACTIVE = "active"


@dataclass
class Address:
    city: str
    zip_code: str


@dataclass
class Profile:
    name: str
    created_at: datetime
    home: Path
    score: Decimal
    tags: list


class FakeColumn:
    def __init__(self, name):
        self.name = name


class FakeTable:
    def __init__(self, names):
        self.columns = [FakeColumn(name) for name in names]


class FakeRow:
    __table__ = FakeTable(["id", "name", "status"])

    def __init__(self):
        self.id = 1
        self.name = "ada"
        self.status = Status.ACTIVE


class FakeStateful:
    def __init__(self, **state):
        self.__dict__.update(state)
        self.training = True

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, payload):
        self.__dict__.clear()
        self.__dict__.update(payload)

    def train(self, mode=True):
        self.training = bool(mode)
        return self


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_demo():
    with tempfile.TemporaryDirectory(prefix="paithon-controls-") as tmp_dir:
        root = Path(tmp_dir)
        provider = DemoProvider()
        engine = RuntimeEngine(
            provider=provider,
            config=RuntimeConfig(
                cache_dir=root / "cache",
                max_state_depth=3,
                max_collection_items=2,
            ),
        )

        print("PAIthon review / sandbox demo")
        print("root =", root)

        class Dashboard:
            def __init__(self):
                self.profile = Profile(
                    name="Ada",
                    created_at=datetime(2025, 1, 2, 3, 4, 5),
                    home=Path("/tmp/demo"),
                    score=Decimal("9.5"),
                    tags=["gnn", "mutag", "jit"],
                )
                self.record = FakeRow()
                self.location = Address("Vienna", "1010")
                self.identifier = UUID("12345678-1234-5678-1234-567812345678")

            @engine.runtime_implemented(state_fields=["profile", "record", "location", "identifier"])
            def render(self):
                """Render a compact dashboard header."""
                raise NotImplementedError

        print("\n1. Rich built-in serializers")
        dashboard = Dashboard()
        print("render() ->", dashboard.render())
        render_request = provider.implement_requests[0][0]
        print("state schema ->", json.dumps(render_request.snapshot.state_schema, sort_keys=True))
        print("state summary ->", json.dumps(render_request.snapshot.state_summary, sort_keys=True))

        class Trainer:
            def __init__(self):
                self.model = FakeStateful(weight=10)
                self.optimizer = FakeStateful(step_count=0)
                self.history = []

            @engine.self_healing(
                state_fields=["history"],
                rollback_on_failure=True,
                rollback_fields=["model", "optimizer", "history"],
            )
            def step(self):
                """Advance a toy training job and record progress."""
                self.model.weight += 5
                self.optimizer.step_count += 7
                self.history.append("broken")
                return missing_name

            @engine.runtime_implemented(state_fields=["history", "model", "optimizer"])
            def summarize_job(self):
                """Return a compact summary of the toy training job."""
                raise NotImplementedError

        print("\n2. Strategy-based rollback")
        trainer = Trainer()
        print("step() ->", trainer.step())
        print("trainer summary ->", trainer.summarize_job())

        print("\n3. Review promotion")
        module_path = root / "review_target.py"
        module_path.write_text(
            "def slug(text):\n"
            "    return text.lowr()\n",
            encoding="utf-8",
        )
        module = load_module(module_path)
        wrapped_slug = engine.self_healing(module.slug, contract="Convert text into a lowercase slug.")
        print("wrapped_slug('Hello Review') ->", wrapped_slug("Hello Review"))
        manifest_path = engine.export_review_artifacts(root / "review")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        slug_entry = next(entry for entry in manifest if entry["qualname"] == "slug")
        review_source_path = Path(slug_entry["source_file"])
        review_source_path.write_text(
            "def slug(text):\n"
            "    return text.lower().replace(' ', '_')\n",
            encoding="utf-8",
        )
        promoted = {
            slug_entry["cache_key"]: str(
                engine.promote_cache_entry(
                    slug_entry["cache_key"],
                    source_text=review_source_path.read_text(encoding="utf-8"),
                )
            )
        }
        print("promoted entries ->", promoted)
        print("updated source file ->")
        print(module_path.read_text(encoding="utf-8").strip())

        print("\n4. Subprocess probe sandbox")
        sandbox_engine = RuntimeEngine(
            provider=provider,
            config=RuntimeConfig(
                cache_dir=root / "sandbox-cache",
                execution_mode="subprocess_probe",
                sandbox_timeout_seconds=2.0,
            ),
        )
        boom = sandbox_engine.create_function("boom", "()", "Return a value.")
        try:
            boom()
        except SafetyViolationError as exc:
            print("boom() blocked ->", exc)

        print("\n5. Restricted subprocess sandbox")
        restricted_engine = RuntimeEngine(
            provider=provider,
            config=RuntimeConfig(
                cache_dir=root / "restricted-sandbox-cache",
                execution_mode="subprocess_restricted",
                sandbox_timeout_seconds=2.0,
                sandbox_memory_limit_mb=128,
                sandbox_file_size_limit_bytes=4096,
                sandbox_max_open_files=16,
            ),
        )
        restricted_boom = restricted_engine.create_function("boom", "()", "Return a value.")
        try:
            restricted_boom()
        except SafetyViolationError as exc:
            print("restricted boom() blocked ->", exc)


if __name__ == "__main__":
    run_demo()
