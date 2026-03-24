# Contributing

This project now treats `paithon` as a review-first JIG library, not as uncontrolled runtime self-modification.

## Default Workflow

Use this operating model unless there is a specific reason not to:

1. Run locally in `review_first`.
2. Let `paithon` generate or heal code during development.
3. Export review artifacts.
4. Review the generated `.py` and `.patch` files.
5. Prefer promoting reviewed source back into normal Python files.
6. Commit promoted source to Git and open a normal PR.
7. Use `production_locked` where fresh runtime generation or healing is not acceptable.

## What To Use PAIthon For

Good fits:

- adapters and parsers
- schema normalization
- external API response handling
- repetitive helper methods
- exploratory stubs that will later be promoted to normal source

Avoid using it first for:

- auth and security boundaries
- payment logic
- core business invariants
- hot paths with tight latency budgets

## Contracts

Contracts should be precise enough that a reviewer can judge correctness from the generated code:

- describe inputs and outputs explicitly
- document important invariants and failure behavior
- describe permitted side effects for methods
- keep one function focused on one responsibility

For methods, prefer `state_fields`, `mutable_state_fields`, and `rollback_fields` so object context and mutation scope stay bounded.

## Review Loop

The concrete starter workflow lives in [templates/real_project/README.md](/media/lorenz/Volume/code/PAIthon/templates/real_project/README.md#L1).

At a high level:

1. Run the code locally so generation or healing happens.
2. Export review artifacts with `engine.export_review_artifacts(...)`.
3. Review `.paithon_review/*.py` and `.paithon_review/*.patch`, or drive the same decisions with `engine.interactive_review(...)`.
4. Choose one of:
   - `engine.promote_review_artifacts(...)` for normal application code
   - `engine.approve_cache_entry(...)` when the cache artifact itself is the intended deployed artifact
   - `engine.export_git_review_bundle(...)` when you want a Git patch plus helper script for branch-based PR flow

Promotion is the default recommendation for code that belongs in the repo.

## Git And PR Policy

- Do not commit `.paithon_cache/`.
- Do not treat `.paithon_review/` as canonical source.
- Do commit promoted source changes.
- If you keep reviewed artifacts out of tree, prefer exporting a Git review bundle instead of manually copying generated code around.
- PRs should be reviewed as normal source diffs, not as "AI output".

Reviewers should check:

- whether the contract is precise enough
- whether the generated code matches the contract exactly
- whether error handling is narrower than the previous failure, not broader and vague
- whether stateful methods mutate only intended fields
- whether tests cover the promoted behavior

## Approval Vs Promotion

Use approval when:

- the function was created dynamically with `create_function(...)`
- there is no natural source file to patch
- you intentionally want a trusted cached artifact without rewriting source

Use promotion when:

- the function lives in a normal Python module
- the behavior should become maintainable repo code
- the change should flow through Git, CI, and PR review

## Environment Defaults

- Local development: `review_first`
- CI validation: promoted source, optionally under `production_locked`
- Production: `production_locked`

`production_locked` only trusts cache entries marked `approved` or `promoted`.

## Repo Hygiene

Recommended ignore rules:

```gitignore
.venv/
.pytest_cache/
build/
dist/
*.egg-info/
.paithon_cache/
.paithon_review/
.paithon_datasets/
```

If you add new demos or examples, keep the README aligned with the actual command-line flags and actual runtime behavior.

## Release Hygiene

- Keep the distribution metadata in [pyproject.toml](/media/lorenz/Volume/code/PAIthon/pyproject.toml#L1) aligned with the actual release name and license.
- Keep [RELEASING.md](/media/lorenz/Volume/code/PAIthon/RELEASING.md#L1) aligned with the GitHub workflows under [.github/workflows](/media/lorenz/Volume/code/PAIthon/.github/workflows).
- The install name is `paithon-jig`, while the import surface remains `paithon`.

## Maintainer Layout

The public runtime import surface remains `paithon.runtime`, but the implementation is intentionally split into smaller modules under [src/paithon/_runtime](/media/lorenz/Volume/code/PAIthon/src/paithon/_runtime).

Current internal responsibilities:

- `engine.py`: `RuntimeEngine` construction and registry setup
- `decorators.py`: decorator APIs and retry wrapper logic
- `execution.py`: generation, repair, install, and initialization flow
- `state.py`: cache keying, variant context, and state snapshot / mutation policy logic
- `review.py`: export, approval, and promotion workflows
- `source.py`: placeholder creation and static safety / subprocess probing
- `policy.py`: operating-mode and audit behavior

If you change runtime behavior, add or update tests for the relevant module-level responsibility rather than only relying on high-level demos.

## Test Layout

The test suite is intentionally split by scope:

- `tests/unit`: isolated helper and policy tests
- `tests/component`: `RuntimeEngine` behavior tests with fake providers, cache directories, and temporary source files
- `tests/integration`: end-to-end example and demo execution tests

Available marker subsets:

```bash
pytest -q -m unit
pytest -q -m component
pytest -q -m integration
```

When fixing a bug, prefer adding the narrowest test that reproduces it first, then add a broader component or integration test only if the failure mode crosses subsystem boundaries.
