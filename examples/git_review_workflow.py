import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


class GitReviewProvider(LLMProvider):
    def repair_function(self, request, model):
        print("[llm] repair {0} via {1}".format(request.snapshot.name, model))
        if request.snapshot.name == "slug":
            return (
                "def slug(text):\n"
                "    return text.lower().replace(' ', '-')\n"
            )
        raise KeyError(request.snapshot.name)

    def implement_function(self, request, model):
        raise KeyError(request.snapshot.name)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def run_demo():
    if shutil.which("git") is None:
        print("git is required for this demo.")
        return

    with tempfile.TemporaryDirectory(prefix="paithon-git-review-") as tmp_dir:
        repo_root = Path(tmp_dir) / "repo"
        repo_root.mkdir()
        run_git(repo_root, "init")
        run_git(repo_root, "config", "user.name", "PAIthon Demo")
        run_git(repo_root, "config", "user.email", "demo@example.com")
        run_git(repo_root, "checkout", "-b", "main")

        source_path = repo_root / "review_target.py"
        source_path.write_text(
            "def slug(text):\n"
            "    return text.lowr()\n",
            encoding="utf-8",
        )
        run_git(repo_root, "add", "review_target.py")
        run_git(repo_root, "commit", "-m", "Add review target")

        provider = GitReviewProvider()
        engine = RuntimeEngine(
            provider=provider,
            config=RuntimeConfig(cache_dir=repo_root / ".paithon_cache", operating_mode="review_first"),
        )
        module = load_module(source_path)
        wrapped_slug = engine.self_healing(module.slug, contract="Convert text into a lowercase slug.")

        print("PAIthon git review workflow")
        print("repo_root =", repo_root)
        print("current_branch =", run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"))
        print("\n1. Heal locally in review_first")
        print("wrapped_slug('Hello Review') ->", wrapped_slug("Hello Review"))

        review_dir = repo_root / ".paithon_review"
        manifest_path = engine.export_review_artifacts(review_dir)
        print("\n2. Interactive review loop")
        results = engine.interactive_review(
            manifest_path,
            reviewer="demo",
            input_func=lambda prompt: "approve",
            output_func=lambda message: print(" ", message),
        )
        print("interactive results ->", results)

        print("\n3. Git review bundle")
        bundle_path = engine.export_git_review_bundle(
            repo_root / ".paithon_git_review",
            manifest_path,
            branch_name="paithon/review-demo",
        )
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        print("bundle ->", bundle_path)
        print("patch_file ->", bundle["patch_file"])
        print("apply_script ->", bundle["apply_script"])
        print("branch_name ->", bundle["branch_name"])
        patch_preview = Path(bundle["patch_file"]).read_text(encoding="utf-8").strip().splitlines()[:6]
        print("patch preview ->")
        for line in patch_preview:
            print(" ", line)


if __name__ == "__main__":
    run_demo()
