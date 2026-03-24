import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class DemoProvider(LLMProvider):
    def __init__(self):
        self.repair_requests = []

    def implement_function(self, request, model):
        raise KeyError(request.snapshot.name)

    def repair_function(self, request, model):
        self.repair_requests.append((request, model))
        name = request.snapshot.name
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        if name == "render_report":
            return (
                "def render_report(self, payload):\n"
                "    return {\n"
                "        'status': self.response.status_code,\n"
                "        'features': self.estimator.n_features_in_,\n"
                "        'model': self.network.name,\n"
                "        'authorization': payload['headers']['authorization'],\n"
                "    }\n"
            )
        if name == "step":
            return (
                "def step(self):\n"
                "    weights = self.model.get_weights()\n"
                "    weights[0] += 1.0\n"
                "    self.model.set_weights(weights)\n"
                "    state = self.scheduler.get_state()\n"
                "    state['epoch'] += 1\n"
                "    self.scheduler.set_state(state)\n"
                "    self.events.append('healed')\n"
                "    return {\n"
                "        'weights': self.model.get_weights(),\n"
                "        'scheduler': self.scheduler.get_state(),\n"
                "        'events': list(self.events),\n"
                "    }\n"
            )
        raise KeyError(name)


class FakeHttpRequest:
    def __init__(self, method):
        self.method = method


class FakeHttpResponse:
    __module__ = "requests.models"

    def __init__(self):
        self.status_code = 200
        self.url = "https://api.example.com/users"
        self.request = FakeHttpRequest("GET")
        self.headers = {"content-type": "application/json", "authorization": "Bearer secret"}

    def json(self):
        return {"user": {"id": 7, "token": "abc"}}


class FakeSklearnEstimator:
    __module__ = "sklearn.linear_model"

    def __init__(self):
        self.n_features_in_ = 3
        self.classes_ = ["a", "b"]

    def get_params(self, deep=False):
        assert deep is False
        return {"alpha": 0.1, "fit_intercept": True}


class FakeKerasModel:
    __module__ = "keras.engine.training"

    def __init__(self, weights, *, trainable=True, name="demo-net"):
        self._weights = list(weights)
        self.trainable = trainable
        self.name = name
        self.built = True

    def get_config(self):
        return {"units": 16, "activation": "relu"}

    def count_params(self):
        return 128

    def get_weights(self):
        return list(self._weights)

    def set_weights(self, weights):
        self._weights = list(weights)


class FakeStateAccessor:
    def __init__(self, state):
        self._state = dict(state)

    def get_state(self):
        return dict(self._state)

    def set_state(self, state):
        self._state = dict(state)


def read_audit_log(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_demo():
    with tempfile.TemporaryDirectory(prefix="paithon-redaction-rollback-") as tmp_dir:
        root = Path(tmp_dir)
        provider = DemoProvider()
        engine = RuntimeEngine(
            provider=provider,
            config=RuntimeConfig(
                cache_dir=root / "cache",
                redacted_field_paths=(
                    "response.headers.authorization",
                    "response.body_preview.user.token",
                    "payload.headers.authorization",
                    "profile.credentials.token",
                ),
                redaction_placeholder="<hidden>",
            ),
        )

        print("PAIthon redaction / rollback demo")
        print("root =", root)

        class PromptContext:
            def __init__(self):
                self.response = FakeHttpResponse()
                self.estimator = FakeSklearnEstimator()
                self.network = FakeKerasModel([1.0, 2.0], trainable=True)
                self.profile = {"credentials": {"token": "abc", "name": "Ada"}}

            @engine.self_healing(state_fields=["response", "estimator", "network", "profile"])
            def render_report(self, payload):
                """Build a report from response state, estimator metadata, model metadata, and payload headers."""
                return missing_name

        print("\n1. Path-based redaction and framework serializers")
        prompt_context = PromptContext()
        report = prompt_context.render_report({"headers": {"authorization": "Bearer secret"}})
        request = next(request for request, _ in provider.repair_requests if request.snapshot.name == "render_report")
        print("state schema ->", json.dumps(request.snapshot.state_schema, sort_keys=True))
        print("state summary ->", json.dumps(request.snapshot.state_summary, sort_keys=True))
        print("call summary ->", json.dumps(request.call_summary, sort_keys=True))
        print("report ->", json.dumps(report, sort_keys=True))

        class Trainer:
            def __init__(self):
                self.model = FakeKerasModel([1.0, 2.0], trainable=True, name="trainer-net")
                self.scheduler = FakeStateAccessor({"epoch": 0, "lr": 0.1})
                self.events = []

            @engine.self_healing(
                rollback_on_failure=True,
                rollback_fields=["model", "scheduler", "events"],
            )
            def step(self):
                """Advance model weights and scheduler state by one step."""
                self.model.set_weights([9.0, 10.0])
                self.model.trainable = False
                self.scheduler.set_state({"epoch": 1, "lr": 0.01})
                self.events.append("broken")
                return missing_name

        print("\n2. Rollback summaries")
        trainer = Trainer()
        result = trainer.step()
        rollback_event = next(
            event
            for event in read_audit_log(root / "cache" / "audit.jsonl")
            if event["event"] == "rollback_applied" and event["qualname"].endswith("Trainer.step")
        )
        print("result ->", json.dumps(result, sort_keys=True))
        print("model rollback summary ->", rollback_event["state_diff"]["model"].get("summary"))
        print("scheduler rollback summary ->", rollback_event["state_diff"]["scheduler"].get("summary"))
        print("events rollback summary ->", rollback_event["state_diff"]["events"].get("summary"))


if __name__ == "__main__":
    run_demo()
