# Real Project Starter

This is a concrete starter skeleton for using `paithon` in a normal Python application with a review-first workflow.

## Layout

```text
templates/real_project/
  .gitignore
  pyproject.toml
  src/myproj/
    __init__.py
    paithon_runtime.py
    adapters.py
    pricing.py
  tools/
    export_paithon_review.py
    interactive_paithon_review.py
    approve_paithon_review.py
    promote_paithon_review.py
    export_paithon_git_review.py
```

## What It Shows

- one explicit shared engine in `src/myproj/paithon_runtime.py`
- `@runtime_implemented` for contract-driven adapters
- `@response_adapter` for external response-shape drift
- review artifact export
- interactive review decisions
- explicit cache approval
- Git-ready review bundle export
- promotion of reviewed code back into source files

## Local Workflow

1. Run the app or tests in `review_first` so code is generated or healed.
2. Export artifacts:

```bash
python tools/export_paithon_review.py
```

3. Review the files under `.paithon_review/`.
4. Optionally drive the same flow interactively:

```bash
python tools/interactive_paithon_review.py --reviewer your-name
```

5. Either promote reviewed code back into source:

```bash
python tools/promote_paithon_review.py
```

6. Or explicitly approve the cache entries:

```bash
python tools/approve_paithon_review.py --reviewer your-name
```

7. Or export a Git-ready patch bundle for branch-based review:

```bash
python tools/export_paithon_git_review.py --branch-name your-feature-branch
```

8. For deployment environments that should refuse fresh runtime mutation, run with:

```bash
PAITHON_OPERATING_MODE=production_locked
```

## Notes

- The template uses `src/` layout and helper scripts that add `src` to `sys.path` so the review tools can run without requiring an editable install first.
- The `pyproject.toml` dependency entry for `paithon-jig` is intentionally minimal. If you consume `paithon` from Git or from a local checkout, replace that dependency with your actual source of truth.
- Promotion is the preferred path for normal repo code. Approval is mainly for dynamic or cache-native workflows.
- The Git review bundle is useful when you want branch-based PR review without applying the promoted source directly in your working tree.
