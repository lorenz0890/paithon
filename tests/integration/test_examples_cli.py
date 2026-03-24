import subprocess
import sys

import pytest

from tests.support.files import PROJECT_ROOT


@pytest.mark.parametrize(
    ("script_path", "expected_snippets"),
    [
        ("examples/review_first_workflow.py", ("PAIthon review-first workflow", "cache_approved", "production_locked")),
        ("examples/demo.py", ("PAIthon demo", "Runtime implementation", "Cache reuse across a fresh engine")),
        ("examples/helper_use_cases.py", ("PAIthon helper use cases", "Schema adapter", "Runtime polyfill")),
        ("examples/advanced_state_policy.py", ("PAIthon advanced state/policy demo", "review manifest", "Mutation allowlist enforcement")),
        ("examples/review_and_sandbox_demo.py", ("PAIthon review / sandbox demo", "promoted entries", "Restricted subprocess sandbox")),
        ("examples/git_review_workflow.py", ("PAIthon git review workflow", "interactive results", "Git review bundle")),
        ("examples/redaction_and_rollback_demo.py", ("PAIthon redaction / rollback demo", "Path-based redaction and framework serializers", "Rollback summaries")),
    ],
)
def test_example_script_runs_successfully(script_path, expected_snippets):
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    output = completed.stdout
    for snippet in expected_snippets:
        assert snippet in output


def test_ml_demo_script_runs_with_scripted_provider_when_ml_stack_is_available():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "examples" / "ml_mutag_demo.py"), "--provider", "scripted", "--epochs", "1"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    output = completed.stdout
    assert "PAIthon ML demo: MUTAG graph classification" in output
    assert "Self-healing training loop" in output
    assert "Runtime-generated summary" in output
