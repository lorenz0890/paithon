import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import RuntimePolicyError
from paithon.provider import LLMProvider


class ReviewFirstProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    def implement_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("implement", name, model))
        print("[llm] implement {0} via {1}".format(name, model))
        if name == "summarize_scores":
            return (
                "def summarize_scores(values):\n"
                "    return {\n"
                "        'min': min(values),\n"
                "        'max': max(values),\n"
                "        'avg': sum(values) / len(values),\n"
                "    }\n"
            )
        raise KeyError(name)

    def repair_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("repair", name, request.error_type, model))
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        if name == "mean":
            return (
                "def mean(values):\n"
                "    if not values:\n"
                "        raise ValueError('values must not be empty')\n"
                "    return sum(values) / len(values)\n"
            )
        raise KeyError(name)


def read_audit_log(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_demo():
    with tempfile.TemporaryDirectory(prefix="paithon-review-first-") as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        provider = ReviewFirstProvider()
        engine = RuntimeEngine(
            provider=provider,
            config=RuntimeConfig(cache_dir=cache_dir, operating_mode="review_first"),
        )

        print("PAIthon review-first workflow")
        print("cache_dir =", cache_dir)
        print("operating_mode =", engine.config.operating_mode)

        summarize_scores = engine.create_function(
            "summarize_scores",
            "(values)",
            "Return a dict with min, max, and avg for a non-empty iterable of numbers.",
        )

        @engine.self_healing
        def mean(values):
            """Return the arithmetic mean of a non-empty iterable of numbers."""
            return sum(values) / len(value)

        print("\n1. Review-first generation and healing")
        print("summarize_scores([2, 4, 6]) ->", summarize_scores([2, 4, 6]))
        print("mean([2, 4, 6]) ->", mean([2, 4, 6]))

        manifest_path = engine.export_review_artifacts(cache_dir / "review")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("\n2. Exported review artifacts")
        for entry in manifest:
            print(" ", entry["qualname"], "approval_status =", entry["approval_status"])

        print("\n3. Explicit approval before production use")
        for entry in manifest:
            engine.approve_cache_entry(entry["cache_key"], reviewer="demo")
        for entry in manifest:
            payload = json.loads((cache_dir / "{0}.json".format(entry["cache_key"])).read_text(encoding="utf-8"))
            print(" ", entry["qualname"], "->", payload["approval_status"])

        print("\n4. Production-locked engine")
        locked_engine = RuntimeEngine(
            provider=ReviewFirstProvider(),
            config=RuntimeConfig(cache_dir=cache_dir, operating_mode="production_locked"),
        )
        locked_summarize_scores = locked_engine.create_function(
            "summarize_scores",
            "(values)",
            "Return a dict with min, max, and avg for a non-empty iterable of numbers.",
        )
        print("locked_summarize_scores([1, 3, 5]) ->", locked_summarize_scores([1, 3, 5]))

        blocked_slugify = locked_engine.create_function(
            "slugify",
            "(text)",
            "Convert text into a lowercase slug.",
        )
        try:
            blocked_slugify("Hello")
        except RuntimePolicyError as exc:
            print("locked slugify generation blocked ->", exc)

        print("\n5. Audit log")
        for event in read_audit_log(cache_dir / "audit.jsonl"):
            print(" ", event["event"], event.get("qualname"), event.get("approval_status"))


if __name__ == "__main__":
    run_demo()
