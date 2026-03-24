# PAIthon

Install from PyPI with `pip install paithon-jig`, then import `paithon`.

`paithon` is a Python library for guide coding: Just-In-Time Generated code, or `JIG`.

The pitch is simple: do not let a model freestyle across your repo. Give it a function signature, a contract, a bounded view of state, and clear runtime policy. Then let it implement or repair code exactly when execution needs it, while keeping review and promotion to normal source as the default path.

- `@runtime_implemented` turns a stub into working code on first call, caches the generated source, and, by default, heals it later if it fails.
- `@self_healing` turns a crash into a constrained repair loop: capture traceback and context, ask for a fix, hot-swap the function, retry the call.
- state policy controls such as `state_fields`, `mutable_state_fields`, `rollback_fields`, `rollback_on_failure`, `strict_rollback`, `cache_by_class`, `contract_revision`, and `heal_on` keep generation and repair inside an explicit envelope.
- `operating_mode="review_first"` is the default. It allows local generation and healing, but marks artifacts as `pending_review` so they can be exported, approved, or promoted before production use.
- review export, interactive approval, Git-ready patch bundles, and promotion let you inspect cached generated or healed code and move it into normal source-control workflows.
- `operating_mode="production_locked"` refuses fresh generation and fresh healing, and only loads approved or promoted cache artifacts.
- optional `execution_mode="subprocess_probe"` and `execution_mode="subprocess_restricted"` add constrained preflight steps before generated code is trusted in-process.

This is intentionally opinionated. The library favors deterministic prompts, local cache persistence, explicit state policy, and reviewability over trying to make AI coding feel magical.

## Guide Coding, Not Vibe Coding

`JIG` means the implementation is generated just in time, at the moment execution actually needs it. But the important word is not "generated." It is "guided."

Vibe coding asks the model to improvise. Guide coding defines the lane first and only then lets the model write inside it.

In `paithon`, you guide the runtime with:

- a function signature and contract
- an explicit state surface via `state_fields`
- a mutation policy via `mutable_state_fields`
- rollback scope via `rollback_fields` and `rollback_on_failure`
- cache lineage controls such as `contract_revision` and class-aware caching
- repair scope via `heal_on`
- review export and source promotion
- optional subprocess probing before in-process installation

That is the opposite of "just write something plausible." The model is being asked to implement or repair code inside a constrained runtime envelope.

This makes `paithon` most useful when the code is cheap to specify precisely but annoying to hand-write repeatedly: adapters, data normalization, external-shape glue, fallback implementations, exploratory stubs, and policy-heavy method bodies.

That does not mean it is only for small problems. Large tasks can be decomposed into smaller functions and methods with precise local contracts, then generated and healed one unit at a time. In that sense, `paithon` fits best with practices such as smaller units of responsibility, explicit interfaces, separation of concerns, and localized failure boundaries.

That decomposition can also help with limited LLM context windows in large codebases. `paithon` does not make whole-system understanding unnecessary, but it does localize generation and repair prompts to the relevant function source, contract, selected state, and summarized globals or closures. For agent-written code, that can preserve a clearer high-level overview in the calling code while delegating local implementation detail to JIG at runtime. Furthermore, the decomposition of complex, large tasks into simple one allows the implementation with cheaper, simple LLM models.

The recommended model is therefore review-first JIG: generate locally, test locally, export review artifacts, promote or approve the result, and use `production_locked` where you want runtime behavior to stay inside trusted artifacts.

## Starter Template

The repo now includes a concrete starter skeleton at [templates/real_project/README.md](/media/lorenz/Volume/code/PAIthon/templates/real_project/README.md#L1).

That template includes:

- one explicit shared engine in [paithon_runtime.py](/media/lorenz/Volume/code/PAIthon/templates/real_project/src/myproj/paithon_runtime.py#L1)
- example contract-driven modules in [adapters.py](/media/lorenz/Volume/code/PAIthon/templates/real_project/src/myproj/adapters.py#L1) and [pricing.py](/media/lorenz/Volume/code/PAIthon/templates/real_project/src/myproj/pricing.py#L1)
- review helper scripts in [export_paithon_review.py](/media/lorenz/Volume/code/PAIthon/templates/real_project/tools/export_paithon_review.py#L1), [approve_paithon_review.py](/media/lorenz/Volume/code/PAIthon/templates/real_project/tools/approve_paithon_review.py#L1), and [promote_paithon_review.py](/media/lorenz/Volume/code/PAIthon/templates/real_project/tools/promote_paithon_review.py#L1)

For team usage with Git and PRs, see [CONTRIBUTING.md](/media/lorenz/Volume/code/PAIthon/CONTRIBUTING.md#L1).

For maintainers: the public [runtime.py](/media/lorenz/Volume/code/PAIthon/src/paithon/runtime.py#L1) module is now a compatibility facade. The implementation is split across smaller modules under [src/paithon/_runtime](/media/lorenz/Volume/code/PAIthon/src/paithon/_runtime).

## Installation

```bash
python -m pip install paithon-jig
```

The PyPI distribution is `paithon-jig`. The import remains:

```python
import paithon
```

## Quick start

```python
from paithon import RuntimeConfig, RuntimeEngine
from paithon import self_healing, runtime_implemented


@self_healing
def mean(values):
    """Return the arithmetic mean of a non-empty iterable of numbers."""
    return sum(values) / len(value)


@runtime_implemented
def slugify(text: str) -> str:
    """Convert text into a lowercase ASCII slug separated by hyphens."""
    raise NotImplementedError
```

Stateful methods can declare which attributes are relevant:

```python
class Wallet:
    def __init__(self, balance: int):
        self.balance = balance
        self.status = "new"
        self.history = []

    @runtime_implemented(state_fields=["balance", "status", "history"])
    def deposit(self, amount: int) -> int:
        """Increase balance, append a history entry, and return the new balance."""
        raise NotImplementedError
```

`state_fields` are read from the first bound argument of the call. For normal instance methods, that means `self`.

You can also add policy controls:

```python
@runtime_implemented(
    state_fields=["balance", "status"],
    mutable_state_fields=["balance"],
    rollback_on_failure=True,
    rollback_fields=["balance", "status"],
    cache_by_class=True,
    contract_revision="r1",
)
def withdraw(self, amount: int) -> int:
    """Decrease balance by amount and return the new balance."""
    raise NotImplementedError
```

## Review Loop

The implemented review loop is:

1. run code locally in `review_first`
2. export artifacts with `engine.export_review_artifacts(...)`
3. inspect the generated `.py` and `.patch` files, or drive the same flow with `engine.interactive_review(...)`
4. either approve cache with `engine.approve_cache_entry(...)`, export a Git-ready patch bundle with `engine.export_git_review_bundle(...)`, or promote reviewed code with `engine.promote_review_artifacts(...)`
5. use `production_locked` where only approved or promoted artifacts should be trusted

For ordinary application code, promotion is the preferred end state. Approval is mainly useful for dynamic or cache-native workflows where there is no natural source file to patch.

## GitHub And PyPI

This repo is prepared for:

- GitHub CI via [ci.yml](/media/lorenz/Volume/code/PAIthon/.github/workflows/ci.yml)
- manual TestPyPI publishing via [publish-testpypi.yml](/media/lorenz/Volume/code/PAIthon/.github/workflows/publish-testpypi.yml)
- tag-driven PyPI publishing via [publish.yml](/media/lorenz/Volume/code/PAIthon/.github/workflows/publish.yml)

For the full release checklist, see [RELEASING.md](/media/lorenz/Volume/code/PAIthon/RELEASING.md#L1).

## Demo

Run the review-first workflow demo:

```bash
python examples/review_first_workflow.py
```

That demo shows the default mode in practice: generate and heal locally, export review artifacts, explicitly approve them, and then load the approved cache from a `production_locked` engine.

Run the Git-aware review demo:

```bash
python examples/git_review_workflow.py
```

That demo creates a temporary Git repo, heals code locally, runs the interactive review loop, and exports a review bundle containing a patch plus an `apply_review.sh` helper for branch-based PR flow.

Run the offline demo:

```bash
python examples/demo.py
```

The offline demo includes a nested scenario where one `@self_healing` function calls another, the inner repair fails locally, and the original exception is handed to the outer function for healing.
It also includes a class-based example where `state_fields` exposes selected object attributes so generated and healed methods can modify `self` coherently.

Run the OpenAI-backed variant:

```bash
OPENAI_API_KEY=... python examples/demo.py --provider openai
```

The current demo script only runs the basic generation, healing, and cache-reuse flow against `--provider openai`. The nested escalation and stateful-method sections are intentionally skipped in that mode because the fake provider keeps those scenarios deterministic.

Run the helper-use-case demo:

```bash
python examples/helper_use_cases.py
```

That demo covers `create_function`, `@schema_adapter`, `@response_adapter`, and `@polyfill` with a deterministic fake provider.

Run the advanced state/policy demo:

```bash
python examples/advanced_state_policy.py
```

That demo covers custom state serializers, rollback-on-failure, mutation allowlists, OOP-aware cache variants, and review export.

Run the review / sandbox demo:

```bash
python examples/review_and_sandbox_demo.py
```

That demo covers richer built-in serializers, strategy-based rollback for `state_dict()` objects, promotion of reviewed source back into the original file, `execution_mode="subprocess_probe"`, and the stronger `execution_mode="subprocess_restricted"` mode with isolated working directories and OS-level resource limits.

Run the redaction / rollback detail demo:

```bash
python examples/redaction_and_rollback_demo.py
```

That demo focuses on the newest state-policy additions: path-based redaction with a custom placeholder, framework-aware serializers for HTTP response / sklearn / keras-like objects, and rollback summaries for keras-weight and `get_state` / `set_state` style objects.

Run the ML demo on `MUTAG`:

```bash
python -m pip install -r requirements-ml-cpu.txt
python examples/ml_mutag_demo.py --provider scripted --epochs 2
```

For Linux CPU-only environments, [requirements-ml-cpu.txt](/media/lorenz/Volume/code/PAIthon/requirements-ml-cpu.txt#L1) installs the PyTorch CPU wheels plus `torch_geometric`. Those versions are intentionally kept separate from the core package dependencies because the ML stack is optional. On its first run, PyG will download `MUTAG` into the dataset cache directory.

The ML demo uses `@runtime_implemented` for experiment setup pieces such as seeding, device resolution, dataset loading, splitting, loader construction, model creation, optimizer creation, parameter counting, and run summarization. It uses `@self_healing` on a stateful training object for `train_epoch()` and `evaluate()`, with `rollback_fields` covering the model, optimizer, and metric state so failed attempts can be rewound before retry.
In `review_first` and `development`, if a first draft violates the static safety policy, the runtime will feed that safety error back through a repair pass once before giving up. That is especially useful in the OpenAI-backed ML demo, where an initial draft might otherwise reach for blocked system imports such as `os`.

You can also try the same flow against the real OpenAI provider:

```bash
OPENAI_API_KEY=... python examples/ml_mutag_demo.py --provider openai --epochs 2
```

You can also inject a custom engine:

```python
from pathlib import Path
from paithon import OpenAIProvider, RuntimeConfig, RuntimeEngine

engine = RuntimeEngine(
    provider=OpenAIProvider(),
    config=RuntimeConfig(
        cache_dir=Path(".paithon_cache"),
        max_heal_attempts=2,
        operating_mode="review_first",
    ),
)


@engine.runtime_implemented
def parse_flag(value: str) -> bool:
    """Parse common boolean strings. Accept yes/no, true/false, 1/0."""
    raise NotImplementedError
```

## How JIG Works

## Operating Modes

`paithon` now has explicit operating modes:

- `review_first`: the default. Runtime generation and healing are allowed, but cached artifacts are marked `pending_review`. This is the recommended local workflow.
- `development`: similar to `review_first`, but the cache is treated as a looser development sandbox and artifacts are marked `development`.
- `production_locked`: fresh runtime generation and fresh runtime healing are disabled. Only approved or promoted cache artifacts are loaded.

In practical terms, the intended path is:

1. run in `review_first`
2. generate or heal locally
3. export review artifacts
4. promote or approve what you trust
5. run `production_locked` where fresh runtime mutation is not acceptable

### Generate

`@runtime_implemented` expects a placeholder body:

- `pass`
- `...`
- `raise NotImplementedError`

On first call it:

1. collects the function signature, contract, source, referenced globals, closure summary, and any declared object state fields from the first bound argument
2. uses registered state serializers when summarizing object state, with built-in support for dataclasses, pydantic-like models, attrs-style objects, namedtuples, enums, datetime/path/decimal/UUID values, SQLAlchemy models and sessions, Django models and querysets, HTTP response objects, sklearn estimators, keras-like models, pandas-like dataframes and series, and numpy/tensor/state summaries
3. asks the provider for a full Python function definition with the same name and signature
4. validates the generated AST against a blocked-call / blocked-import policy and can optionally probe the generated definition in an isolated subprocess before installing it in-process
5. compiles that function into the original global namespace
6. stores the generated source plus review metadata and approval status in a local disk cache
7. runs the call

The cache key is derived from the module, qualified name, signature, resolved contract text, source, decorator mode, declared state and policy fields such as `state_fields`, `mutable_state_fields`, `rollback_fields`, and `strict_rollback`, optional `contract_revision`, and a runtime context that can include first-bound-argument class metadata and serialized state schema. If the contract, source, declared state or rollback policy, revision, class version, or state schema change, a different cache entry is used.

### Repair

`@self_healing` wraps an existing implementation. On failure it:

1. captures the traceback and call arguments
2. summarizes the function's globals, closures, and declared object state from the first bound argument to keep prompts compact
3. optionally snapshots object state before the call so failed mutations can be rolled back, with built-in snapshot support for deep-copy values, state accessor objects (`get_state` / `set_state` style), keras weight containers, pandas-like dataframes and series, numpy-like arrays, tensor-like values, and `state_dict()` / `load_state_dict()` objects
4. asks the provider for a corrected implementation
5. validates, compiles, and caches the fix with review metadata
6. retries the original call

The default is one healing attempt per failing invocation.

If a nested self-healing function cannot repair itself, the original exception is re-raised with the failed repair attempt chained as the cause. That lets an outer self-healing caller make a higher-level fix such as fallback policy, retries, or alternate orchestration.

If `mutable_state_fields` is declared, successful calls are checked against that allowlist. Disallowed mutations raise `StateMutationError`. When `rollback_on_failure=True`, failed calls restore the snapshotted rollback scope before retry or re-raise, and audit logs include a structured field-level diff of the rolled-back state. `rollback_fields=[...]` lets you roll back state that is important for correctness even if you do not want it exposed in prompts, and `strict_rollback=True` turns unsupported snapshot targets into an explicit failure instead of best-effort skipping. If healing is enabled for that wrapper and the exception matches its `heal_on` filter, a `StateMutationError` can itself trigger a repair attempt.

## State Policy

The state/policy features currently implemented are:

- `state_fields=[...]` to expose selected attributes from the first bound argument
- `mutable_state_fields=[...]` to restrict which selected object attributes may change
- `rollback_on_failure=True` to restore object state after a failed call before healing or re-raising
- `rollback_fields=[...]` to snapshot and restore additional fields, including `state_dict()`-style model and optimizer objects
- `strict_rollback=True` to fail fast when the declared rollback scope cannot be snapshotted safely
- `cache_by_class=True` to vary cache entries by first-bound-argument class metadata
- `contract_revision="..."` to force a new cache lineage without rewriting the contract text
- `RuntimeConfig(redacted_field_names=..., redacted_field_patterns=..., redacted_field_paths=..., redaction_placeholder=...)` to redact sensitive state, call-summary, global, and closure entries in runtime prompts
- `engine.register_state_serializer(...)` to provide compact summaries and schema tokens for domain objects
- `engine.register_snapshot_strategy(...)` to teach the engine how to capture, compare, and restore domain-specific mutable objects
- `RuntimeConfig(max_mapping_items=..., max_sequence_items=..., max_set_items=...)` to bound nested state summaries more precisely than the global collection limit
- `engine.export_review_artifacts(path)` to write generated/healed source and unified diff suggestions for review
- `engine.interactive_review(path_or_manifest, ...)` to drive approve/promote/skip decisions from an interactive loop
- `engine.approve_cache_entry(cache_key, reviewer="...")` to mark a cache artifact as production-usable without rewriting the source file
- `engine.export_git_review_bundle(path, path_or_manifest, ...)` to emit a Git-ready patch bundle plus an `apply_review.sh` helper script for branch-based review
- `engine.promote_review_artifacts(path_or_manifest)` and `engine.promote_cache_entry(cache_key)` to write reviewed/generated code back into source files
- `RuntimeConfig(operating_mode="review_first" | "development" | "production_locked")` to control generation, healing, and cache trust rules
- `RuntimeConfig(execution_mode="subprocess_probe" | "subprocess_restricted")` to probe generated code in an isolated subprocess before trusting it in-process

Legacy cache files created before review metadata existed are normalized to `approval_status="legacy_untracked"` on load. In `review_first` they remain usable but clearly marked in audit logs; in `production_locked` they are still rejected until explicitly approved or promoted.
Malformed cache files are treated as missing. In `review_first` or `development`, that can trigger regeneration; in `production_locked`, the runtime still fails closed and refuses fresh generation.
If generated or repaired code violates the static safety policy, the runtime raises `SafetyViolationError`. In `review_first` and `development`, it will first try one bounded repair pass per attempt so the model can rewrite the function without blocked imports or calls; in `production_locked`, fresh generation and repair remain disabled.

## Helper APIs

The engine also exposes a few higher-level helpers built on top of the two core decorators:

- `create_function(name, signature, contract, ...)` creates a runtime-implemented callable directly from contract text. This is useful for agent-authored plugins, REPL sessions, and exploratory stubs.
- `@schema_adapter(...)` is a thin wrapper around `@runtime_implemented` that prefixes the contract for one-off parsing and normalization tasks.
- `@response_adapter(...)` is a thin wrapper around `@self_healing` for external response parsing. By default it only heals `KeyError`, `TypeError`, `ValueError`, `AttributeError`, and `IndexError`, and it retries up to two repairs per call.
- `@polyfill(...)` generates a runtime fallback only when a dependency is missing or the current `sys.platform` is not in the declared platform allowlist. Otherwise the original implementation is used unchanged.
- `engine.register_state_serializer(...)` registers custom serializers used for prompt summaries and state-schema-aware caching.
- `engine.register_snapshot_strategy(...)` registers custom rollback strategies for mutable domain objects.
- `RuntimeConfig(redacted_field_names=..., redacted_field_patterns=..., redacted_field_paths=..., redaction_placeholder=...)` redacts matching field names, nested paths, and runtime variables in state snapshots, call summaries, globals, and closures.
- `engine.export_review_artifacts(path)` exports cached generated/healed source plus patch suggestions into a review directory.
- `engine.interactive_review(path_or_manifest, ...)` runs an interactive approve/promote/skip loop over exported entries.
- `engine.approve_cache_entry(cache_key, reviewer="...")` marks a cache entry as approved for `production_locked` use.
- `engine.export_git_review_bundle(path, path_or_manifest, ...)` emits a Git patch plus helper script for branch-based review flow.
- `engine.promote_review_artifacts(path_or_manifest)` applies reviewed export artifacts back into source files.
- `engine.promote_cache_entry(cache_key)` writes a single cached implementation back into its original source file.

If you call `create_function(...)` without a `namespace=...`, it uses an isolated dynamic module namespace. Pass `namespace=globals()` if the generated function needs access to your current module globals.

## What this version does not pretend to solve

- Async functions are not supported yet.
- Arbitrary nested closures cannot be reconstructed exactly. Closure values are summarized for prompting and injected back as globals when possible, which works for many read-only cases but not full `nonlocal` semantics.
- Full container- or VM-grade sandboxed execution is not implemented. The current safety boundary is an AST policy plus optional subprocess probing, and `subprocess_restricted` adds isolated working directories plus OS-level resource limits on supported platforms. Generated code still ultimately runs in-process after that probe.
- The cache stores generated source code, not formal proofs of correctness.
- `review_first` does not stop you from reusing pending-review cache locally; it records review status so you can distinguish local iteration from approved production artifacts.

Those are deliberate boundaries for a first version.

## Good Fits Today

- schema adapters that write one-off parsers from contracts or docstrings via `@schema_adapter`
- runtime polyfills for optional dependencies or platform-specific behavior via `@polyfill`
- agent-authored plugin functions in REPL or notebook sessions via `create_function`
- contract-driven stubs in exploratory prototyping via `create_function` and `@runtime_implemented`
- automatic retry through repair when external API response shapes drift via `@response_adapter`
- hierarchical repair, where low-level functions escalate caller-policy decisions upward via nested `@self_healing`
- experimental ML glue such as dataset setup, small model builders, and fragile train/eval loops, as shown in the `MUTAG` demo
- reviewable runtime fixes that can be promoted back into normal source files after human inspection
- review-first development flows where agent- or human-authored contracts generate code locally and production only trusts approved artifacts

## Still Missing

- test-time flake healing for data pipelines and notebooks

## What JIG Changes

This pattern is powerful, but it changes how software behaves:

- The build artifact is no longer the whole program; part of the program is generated at runtime.
- Reproducibility requires cache control, prompt versioning, and model pinning.
- Security posture changes because generated code is executable code.
- Debugging shifts from "what did the developer write?" to "what contract, prompt, cache entry, and model produced this function?"
- Teams will need policy: which modules may self-modify, which environments allow healing, and whether healed source should be promoted back into static source control.

In practice, the right deployment model is usually:

- allow runtime generation in local development and exploratory systems
- restrict or require review in production
- log every generated or healed source artifact with metadata

## Next Improvements

The next serious improvements should focus on state, policy, and observability:

- Serializer breadth: built-ins now cover many common Python, ORM-like, HTTP-like, dataframe-like, and ML-like shapes, including Django querysets, SQLAlchemy sessions, HTTP response objects, sklearn estimators, keras-like models, pandas-like dataframes and series, plus name-, pattern-, and path-based redaction controls. The next step is more first-party serializers for specific frameworks and even tighter policy controls around partial-field exposure.
- Rollback fidelity: snapshot strategies now cover deep-copy values, state accessor objects, keras-like weight containers, pandas-like dataframes and series, tensor-like values, numpy-like arrays, and `state_dict()` objects, and rollback audits include structured field diffs, kind-specific diff details, and short human-facing summaries. More framework-specific strategies and still richer diff presentation are still worth adding.
- Review workflows: approval, promotion, interactive review, and Git patch export now exist. The next step is forge-aware automation for hosted PR systems and branch lifecycle management.
- Sandboxing: `subprocess_restricted` now adds isolated working directories and OS-level resource limits before in-process installation. Container- or VM-level isolation would still be a stronger safety boundary.

## Running tests

```bash
pytest -q
```

The suite is split into:

- `tests/unit` for isolated helper and policy tests
- `tests/component` for `RuntimeEngine` behavior with fake providers and local files
- `tests/integration` for example scripts and end-to-end flows

You can also run subsets with markers:

```bash
pytest -q -m unit
pytest -q -m component
pytest -q -m integration
```
