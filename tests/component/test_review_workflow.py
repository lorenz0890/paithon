import json
import shutil
import subprocess
from pathlib import Path

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from paithon.exceptions import CodeRepairError, ReviewPromotionError, RuntimePolicyError, SafetyViolationError
from paithon.provider import LLMProvider
from tests.support.files import load_module_from_path, read_jsonl


class WorkflowProvider(LLMProvider):
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


def init_git_repo(path: Path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "PAIthon Tests"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "paithon-tests@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "checkout", "-b", "main"], cwd=path, check=True, capture_output=True, text=True)

def test_production_locked_blocks_pending_review_cache_entries(tmp_path):
    provider = WorkflowProvider(implementations={"slugify": "def slugify(text):\n    return text.lower()\n"})
    review_engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path, operating_mode="review_first"))
    slugify = review_engine.create_function("slugify", "(text)", "Convert text to lowercase.")

    assert slugify("Hello") == "hello"

    locked_engine = RuntimeEngine(
        provider=WorkflowProvider(),
        config=RuntimeConfig(cache_dir=tmp_path, operating_mode="production_locked"),
    )
    locked_slugify = locked_engine.create_function("slugify", "(text)", "Convert text to lowercase.")

    with pytest.raises(RuntimePolicyError):
        locked_slugify("Hello")

    audit_events = read_jsonl(tmp_path / "audit.jsonl")
    assert any(event["event"] == "cache_load_blocked" and event["approval_status"] == "pending_review" for event in audit_events)


def test_approve_cache_entry_requires_existing_cache_key(tmp_path):
    engine = RuntimeEngine(provider=WorkflowProvider(), config=RuntimeConfig(cache_dir=tmp_path))

    with pytest.raises(ReviewPromotionError):
        engine.approve_cache_entry("missing-cache-key")


def test_promote_cache_entry_requires_source_path(tmp_path):
    provider = WorkflowProvider(implementations={"dynamic_fn": "def dynamic_fn(value):\n    return value * 2\n"})
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    dynamic_fn = engine.create_function("dynamic_fn", "(value)", "Double a number.", cache_by_class=False)

    assert dynamic_fn(4) == 8
    cache_key = next(path.stem for path in tmp_path.glob("*.json") if path.name != "audit.jsonl")

    with pytest.raises(ReviewPromotionError):
        engine.promote_cache_entry(cache_key)


def test_promote_cache_entry_requires_existing_target_file(tmp_path):
    source_path = tmp_path / "promote_missing_target.py"
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
    provider = WorkflowProvider(
        implementations={"slug": "def slug(text):\n    return text.lower().replace(' ', '-')\n"}
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path / "cache"))
    slug = engine.runtime_implemented(module.slug)

    assert slug("Hello World") == "hello-world"
    cache_key = next(path.stem for path in (tmp_path / "cache").glob("*.json") if path.name != "audit.jsonl")
    source_path.unlink()

    with pytest.raises(ReviewPromotionError):
        engine.promote_cache_entry(cache_key)


def test_promoted_runtime_implemented_source_preserves_non_paithon_decorators(tmp_path):
    source_path = tmp_path / "decorated_module.py"
    source_path.write_text(
        "def runtime_implemented(*args, **kwargs):\n"
        "    def decorate(func):\n"
        "        return func\n"
        "    return decorate\n\n"
        "class Slugger:\n"
        "    @staticmethod\n"
        "    @runtime_implemented()\n"
        "    def slug(text):\n"
        "        \"\"\"Convert text into a lowercase slug.\"\"\"\n"
        "        raise NotImplementedError\n",
        encoding="utf-8",
    )
    module = load_module_from_path(source_path)
    provider = WorkflowProvider(
        implementations={"slug": "def slug(text):\n    return text.lower().replace(' ', '-')\n"}
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path / "cache"))
    slug = engine.runtime_implemented(module.Slugger.slug)

    assert slug("Hello World") == "hello-world"
    cache_key = next(path.stem for path in (tmp_path / "cache").glob("*.json") if path.name != "audit.jsonl")
    engine.promote_cache_entry(cache_key)
    promoted_text = source_path.read_text(encoding="utf-8")

    assert "@runtime_implemented" not in promoted_text
    assert "@staticmethod" in promoted_text

    promoted_module = load_module_from_path(source_path)
    assert promoted_module.Slugger.slug("Hello World") == "hello-world"


def test_safety_repair_failure_reraises_original_safety_violation(tmp_path):
    provider = WorkflowProvider(
        implementations={
            "unsafe_reader": (
                "def unsafe_reader():\n"
                "    import os\n"
                "    return os.listdir('.')\n"
            )
        },
        repairs={"unsafe_reader": RuntimeError("provider failed to repair unsafe source")},
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    unsafe_reader = engine.create_function("unsafe_reader", "()", "Return a directory listing.")

    with pytest.raises(SafetyViolationError) as exc_info:
        unsafe_reader()

    assert "blocked import in generated code: os" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, CodeRepairError)
    assert getattr(exc_info.value, "__paithon_repair_error__", None) is exc_info.value.__cause__


def test_interactive_review_can_approve_entries(tmp_path):
    provider = WorkflowProvider(implementations={"slugify": "def slugify(text):\n    return text.lower()\n"})
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))
    slugify = engine.create_function("slugify", "(text)", "Convert text to lowercase.")

    assert slugify("Hello") == "hello"
    manifest_path = engine.export_review_artifacts(tmp_path / "review")
    answers = iter(["approve"])

    results = engine.interactive_review(
        manifest_path,
        reviewer="tester",
        input_func=lambda prompt: next(answers),
        output_func=lambda message: None,
    )

    cache_key = next(iter(results))
    assert results[cache_key] == "approve"
    payload = json.loads((tmp_path / "{0}.json".format(cache_key)).read_text(encoding="utf-8"))
    assert payload["approval_status"] == "approved"
    assert payload["approved_by"] == "tester"


def test_interactive_review_can_promote_entries(tmp_path):
    source_path = tmp_path / "interactive_promote.py"
    source_path.write_text(
        "def slug(text):\n"
        "    return text.lowr()\n",
        encoding="utf-8",
    )
    module = load_module_from_path(source_path)
    provider = WorkflowProvider(
        repairs={"slug": "def slug(text):\n    return text.lower().replace(' ', '_')\n"}
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path / "cache"))
    wrapped = engine.self_healing(module.slug, contract="Convert text into a lowercase slug.")

    assert wrapped("Hello Review") == "hello_review"
    manifest_path = engine.export_review_artifacts(tmp_path / "review")
    answers = iter(["promote"])

    results = engine.interactive_review(
        manifest_path,
        input_func=lambda prompt: next(answers),
        output_func=lambda message: None,
    )

    assert next(iter(results.values())) == "promote"
    assert "return text.lower().replace(' ', '_')" in source_path.read_text(encoding="utf-8")


def test_export_git_review_bundle_writes_patch_and_apply_script(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    init_git_repo(repo_root)
    source_path = repo_root / "review_target.py"
    source_path.write_text(
        "def slug(text):\n"
        "    return text.lowr()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "review_target.py"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "Add review target"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    module = load_module_from_path(source_path)
    provider = WorkflowProvider(
        repairs={"slug": "def slug(text):\n    return text.lower().replace(' ', '-')\n"}
    )
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=repo_root / ".paithon_cache"))
    wrapped = engine.self_healing(module.slug, contract="Convert text into a lowercase slug.")

    assert wrapped("Hello Review") == "hello-review"
    manifest_path = engine.export_review_artifacts(repo_root / ".paithon_review")
    bundle_path = engine.export_git_review_bundle(
        repo_root / ".paithon_git_review",
        manifest_path,
        branch_name="paithon/review-demo",
    )

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    patch_path = Path(bundle["patch_file"])
    apply_script_path = Path(bundle["apply_script"])
    patch_text = patch_path.read_text(encoding="utf-8")
    apply_script = apply_script_path.read_text(encoding="utf-8")

    assert bundle["branch_name"] == "paithon/review-demo"
    assert bundle["current_branch"] == "main"
    assert "--- a/review_target.py" in patch_text
    assert "+++ b/review_target.py" in patch_text
    assert "return text.lowr()" in patch_text
    assert "return text.lower().replace(' ', '-')" in patch_text
    assert "git switch -c paithon/review-demo HEAD" in apply_script
    subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=repo_root, check=True, capture_output=True, text=True)
