# Woof Woof: Autonomous ML Research for KuaiRand-Pure

This repository contains a validation-first autonomous research agent for the
TikTok TechJam KuaiRand-Pure recommender benchmark. It reproduces the official
Factorization Machine (FM), proposes or selects one focused improvement at a
time, executes it in a guarded subprocess, independently evaluates its
predictions, records a search tree, repairs failures, checks small gains on a
second seed, and stops on the official convergence/budget rules.

The frozen task is **within-user ranking of logged video impressions** with the
native binary `long_view` label. Metrics are **GAUC** and **nDCG@5**;
`primary = mean(GAUC, nDCG@5)`. [evaluate.py](evaluate.py) is not modified.

One supplied problem-statement screenshot contains a contradictory legacy row
that says `click` with nDCG@10/Recall@50. The repeated task, benchmark,
evaluation, and judging sections instead specify `long_view`, GAUC, and
nDCG@5, as do the starter kit's pinned `evaluate.py` and submission format.
This repository follows that repeated, executable contract. If the organizer's
submission portal publishes a newer evaluator, confirm the contract with them
before changing any metric or label; do not silently mix the two definitions.

## Current validation result

The first knowledge-guided operator uses the textbook BPR pairwise loss on the
existing FM, with benchmark-adapted observed-negative sampling. It pairs each
eligible logged `long_view=1` training impression with a uniformly sampled
logged `long_view=0` impression from the same user. Positives from all-positive
users (0.40% of training positives) cannot form a pair and are skipped; the
operator never treats an unexposed catalogue item as a negative.

| Validation, seed 0 | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Reproduced pointwise FM | 0.667133 | 0.535805 | 0.601469 |
| Observed-negative BPR-style FM | **0.669711** | **0.537081** | **0.603396** |

| Seed-0 BPR delta | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Versus reproduced FM | **+0.002578** | **+0.001276** | **+0.001927** |
| Versus published validation baseline | **+0.002311** | **+0.001381** | **+0.001796** |

- Paired seed-1 gain: **+0.000460**.
- Two-seed mean paired gain: **+0.001194**.

This is a modest, preliminary validation result, not a held-out or statistical
significance claim. Because the gain is below the `0.002` convergence/noise
threshold, the agent automatically ran a paired second-seed check before
promoting BPR as its current best node. Seed 1 improved primary but reduced
nDCG@5 slightly. Two seeds are a promotion heuristic, not enough to establish
significance. No held-out metric was evaluated.

## What changed from the starter kit

- Candidate code emits `scores.npy`; it cannot declare its own winning metric.
  The orchestrator checks exact shape, row count, numeric dtype, and finiteness,
  then calls the pinned evaluator itself.
- Research runs receive a generated development data view containing train and
  validation rows only. The random-exposure log and test rows are absent.
- A one-time final view preserves test features/order but masks every test
  outcome (`long_view`, engagement fields, play/stay times). Finalization writes
  predictions and a submission but never computes held-out metrics.
- Iterations form a tree with `parent_id`, operation, direction, hypothesis,
  sources, code diff, metrics, failures, recovery attempts, token usage, and
  timing in `run_log.jsonl`.
- Small gains are compared on a second seed. Three non-improving rounds trigger
  convergence; the hard caps are 50 total nodes and six hours by default.
- Failed candidates can be repaired twice. The actual repaired artifact is
  promoted into the canonical iteration directory, fixing the starter loop's
  `_fixN` finalization bug.
- Runs are resumable. Partial iteration directories are preserved as
  `.orphaned-*`, and a log entry written just before a crash can rebuild state.
- Source data, generated views, runtime modules, and selected pipeline code are
  hash-pinned. A per-run lock rejects concurrent resumes, and a durable marker
  is written before one-time finalization can access held-out rows.
- Model-authored subprocesses receive no API key or other secret environment
  variables. AST checks reject aliased process/network/destructive APIs; a
  timeout kills the process group. Linux additionally gets CPU/file/memory
  limits.
- The supplied research notes are bundled in
  [agent/knowledge_base.md](agent/knowledge_base.md) as explicitly
  reference-only material. They do not override the task contract.
- `data.load(..., split_names=...)` supports selective loading, and
  `data.iter_interactions()` exposes timestamp/auxiliary signals for future
  causal-history and training-only multitask experiments.
- Baseline and ablation CLIs are validation-only by default. Local held-out
  evaluation requires an explicit flag; `submit.py` refuses test scoring.

## Setup

Python 3.9+ is supported. An isolated environment is strongly recommended.

With `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

Or with the standard library:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The archived starter instructions identify the official KuaiRand download at
[kuairand.com](https://kuairand.com) and provide this registration-free Zenodo
artifact. From the repository root:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

These commands should produce `./KuaiRand-Pure/`. The archive and extracted
dataset are ignored by Git; do not commit or redistribute their CSV files.

The data directory should contain at least:

```text
KuaiRand-Pure/data/
├── log_standard_4_08_to_4_21_pure.csv
├── log_standard_4_22_to_5_08_pure.csv
├── user_features_pure.csv
├── video_features_basic_pure.csv
└── video_features_statistic_pure.csv
```

Run the fast readiness check:

```bash
python agent/preflight.py
```

Iterations 0 and 1—the pointwise FM baseline and built-in BPR-style
candidate—need no API key. Nodes after those use the OpenAI Responses API.
Create your local configuration from the committed template:

```bash
cp .env.example .env
chmod 600 .env
```

Open `.env` and replace the API-key placeholder. The second line is the
recommended quality/cost-balanced default for this iterative coding workload:

```dotenv
OPENAI_API_KEY=replace-with-your-real-openai-api-key
OPENAI_MODEL=gpt-5.6-terra
```

Put the real API key only in `.env`. Never paste it into Python source, chat,
terminal transcripts, run logs, screenshots, or a commit. `.env` is ignored by
Git. `OPENAI_MODEL` must be a model ID that the OpenAI project attached to your
key can access. If `gpt-5.6-terra` is unavailable to that project, replace it
with an available Responses API model. OpenAI describes Terra as its balance
of intelligence and cost and documents function calling support on its
[official model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra).
Check that the local SDK, key setting, model setting, and data files are ready
without making an API request or printing the secret:

```bash
python agent/preflight.py --require_llm
```

The runner loads repository-root `.env` automatically. You can instead set
`OPENAI_API_KEY` and `OPENAI_MODEL` in the process environment. Passing
`--model <model-id>` overrides `OPENAI_MODEL` for that command. The integration
uses the [official OpenAI Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

## Test step by step

### 1. Run all contract tests

```bash
python -m unittest discover -s tests -v
```

The suite covers the pinned evaluator, split isolation, submission alignment,
BPR pairing/gradient direction, AST safety, secret scrubbing, exact score
validation, repair promotion, data-view masking, resume reconciliation, and
convergence. The current suite has 70 tests.

### 2. Reproduce the official baseline on validation only

```bash
python baseline.py --model fm
```

Expected primary is approximately `0.6016`. This command loads only train and
validation unless `--evaluate-local-heldout` is explicitly supplied.

### 3. Exercise the trusted harness with only iteration 0

```bash
python agent/orchestrator.py \
  --max_iterations 1 \
  --run_dir /tmp/woof-baseline-run
```

Check `/tmp/woof-baseline-run/summary.json`. `test_metrics_evaluated` must be
`false`, and baseline reproduction must be within the configured tolerance.

### 4. Run the autonomous baseline → BPR experiment without an API key

```bash
python agent/orchestrator.py \
  --max_iterations 2 \
  --run_dir /tmp/woof-bpr-run
```

This performs the full autonomous cycle for the built-in observed-negative
BPR-style operator. If its seed-0 gain is below `0.002`, the harness
automatically runs both that candidate and its current-best parent with seed 1
before deciding whether to promote it.

Inspect:

```bash
python -m json.tool /tmp/woof-bpr-run/summary.json
python -m json.tool /tmp/woof-bpr-run/search_tree.json
```

### 5. Exercise one OpenAI-guided research node

After `python agent/preflight.py --require_llm` reports `"ok": true`, run:

```bash
python agent/orchestrator.py \
  --max_iterations 3 \
  --run_dir /tmp/woof-openai-run
```

Iterations 0 and 1 are the trusted FM and BPR nodes. Iteration 2 is the first
OpenAI-proposed node, so this step makes a paid API request and may make
additional retry/repair requests if the provider or generated code fails.
Inspect `/tmp/woof-openai-run/summary.json`: `provider` should be
`builtin+openai`, and the token counts should be nonzero.

### 6. Run the full LLM-guided search

```bash
python agent/orchestrator.py \
  --max_iterations 50 \
  --wall_clock_hours 6
```

This command uses `OPENAI_MODEL` from `.env`. To override it for one run:

```bash
python agent/orchestrator.py \
  --model '<model ID enabled for account>' \
  --max_iterations 50 \
  --wall_clock_hours 6
```

After iteration 1, the OpenAI model receives the current parent pipeline,
complete experiment registry, search results, and bundled knowledge base. It
proposes one focused change, debugs failures, and changes direction after
plateaus. Once started with valid credentials, no human choice is required
during the run. A run capped at `--max_iterations 2` never calls the API;
later nodes require a valid OpenAI key and model.

If the process is interrupted, resume it explicitly:

```bash
python agent/orchestrator.py --resume agent/runs/<run-id>
```

If a human changed the code/configuration between attempts, record that in the
deliverable count:

```bash
python agent/orchestrator.py --resume agent/runs/<run-id> \
  --manual_interventions 1
```

### 7. Inspect the converged run before finalization

Each run contains:

```text
agent/runs/<run-id>/
├── run_log.jsonl          # one complete record per research node
├── search_tree.json       # convenient tree-shaped export
├── state.json             # durable resume state
├── summary.json           # best validation result and resource totals
├── best_pipeline.py       # exact validation-selected winner
├── runtime/               # pinned candidate-visible data/evaluator modules
├── data_views/development # no test rows or random-exposure log
└── iterations/iter_XXXX/  # code, attempts, scores, diagnostics, logs
```

Verify that the best iteration and its code/diff are sensible, every accepted
score came from the trusted evaluator, and convergence/resource fields are
present.

### 8. Generate the final submission once

Only after the run is frozen:

```bash
python agent/orchestrator.py \
  --resume agent/runs/<run-id> \
  --finalize
```

This executes the exact saved winner against the masked final feature view and
writes `agent/runs/<run-id>/submission.csv`. It records a SHA-256 digest and
refuses a second finalization for the same run. It does **not** calculate test
metrics.

The saved research artifact is executable source, configuration, seed, and
data order—not serialized learned weights. Finalization deterministically
retrains that source on the training split, uses validation for early stopping,
and then scores the masked final view. Re-running with the same data and
environment is intended to reproduce it; NumPy/platform changes can still
affect bit-level results. If the event requires a binary checkpoint rather than
reproducible code plus final scores, add checkpoint serialization before the
submission deadline.

Validate the output schema/alignment:

```bash
python submit.py --check --split test \
  agent/runs/<run-id>/submission.csv
```

Validation scoring remains available for a validation-format file:

```bash
python submit.py --make --split valid /tmp/woof-validation-fm.csv
python submit.py --score --split valid /tmp/woof-validation-fm.csv
```

`python submit.py --score --split test ...` is deliberately rejected.

## Autonomous search policy

The implementation follows the bundled knowledge pack's inexpensive-first
policy:

1. Loss alignment: observed-negative BPR-style FM, then listwise/refinements.
2. Causal user history: simple pooling before DIN/SIM-style attention.
3. Training-only auxiliary outcomes; no same-impression outcome inputs.
4. Censored watch-time modeling.
5. Temporal/drift diagnostics, then capacity-heavy architectures.

The harness normally branches from the validation best. Every fifth node may
branch from a near-best underexplored node. It records all parents and tried
ideas, asks for one change at a time, switches direction after three misses,
and recommends pairwise combination once independent directions have produced
wins that passed the configured promotion checks.

Primary references in the bundled notes include Rendle et al. (BPR, UAI 2009),
Zhou et al. (DIN, KDD 2018), Pi et al. (SIM, CIKM 2020), Ma et al. (ESMM,
SIGIR 2018), Kang and McAuley (SASRec, ICDM 2018), Zhao et al. (CWM, KDD
2024), and AIDE's tree-search framing. No live web source was needed for the
implemented BPR operator.

## Validation evidence and artifact plan

The compact [hardened BPR bootstrap evidence](results/validation_bpr/README.md)
contains sanitized metrics, two compact run-log records, and the exact executed
code diff. It deliberately excludes raw data, per-row scores, secrets, local
user paths, and submissions. This two-node bootstrap stopped at its requested
iteration cap and must not be described as a converged full search.

Before the final write-up, archive the eventual converged run separately with
its hypothesis, exact diff, GAUC/nDCG@5/primary, error and recovery events,
manual-intervention count, token usage, wall-clock, iterations, environment,
and best-source hash. Keep large runtime directories and score arrays out of
Git. The organizer-generated hidden-test result should be reported only after
the one permitted evaluation; it should never be backfilled into a development
log.

## Project disclosure checklist

Use this section as the source of truth for the Devpost description, then
replace every explicit team placeholder before submission.

- **Development tools:** a VS Code-compatible IDE/Codex workspace, terminal,
  and Python CLI. GitHub is the intended delivery host, but this extracted
  working directory is not yet initialized as a standalone repository.
- **APIs and agents:** OpenAI Codex assisted repository development. Autonomous
  post-bootstrap proposals and repairs use the OpenAI Responses API through
  its Python SDK when an OpenAI key and model are supplied. The verified
  built-in FM/BPR run made no OpenAI API calls and used zero LLM API tokens.
- **Libraries/frameworks:** NumPy is the numerical runtime; the optional
  OpenAI SDK is the only agent API dependency. Tests use Python's standard
  `unittest`. The implemented path does not require pandas, scikit-learn,
  PyTorch, RecBole, or pretrained weights.
- **Datasets/assets:** only the organizer-provided KuaiRand-Pure standard logs
  and user/video feature files are training inputs. The random-exposure log is
  excluded from the development view. `agent/knowledge_base.md` is a bundled,
  human-reviewed research reference, not training data or an instruction
  override. No external training data or pretrained model weights are used.
- **Team contributions — TODO:** replace the placeholders below with real names
  and concrete work; do not submit the template as attribution.

| Team member | Contribution |
|---|---|
| `<name 1>` | `<agent architecture / modeling / evaluation / write-up>` |
| `<name 2>` | `<agent architecture / modeling / evaluation / write-up>` |
| `<name 3, if applicable>` | `<specific contribution>` |

The dataset retains its own terms in `KuaiRand-Pure/LICENSE`. This repository
currently has **no top-level project-code license**, so third-party reuse rights
have not been granted. Choose and add an appropriate `LICENSE` before presenting
the repository as open source; do not assume the dataset license covers this
code.

## Safety and scientific limitations

- The AST checker and subprocess controls are defense-in-depth, not a hard
  container or VM boundary. Candidate code is passed only the generated data
  view, but ordinary Python file reads cannot be perfectly confined by AST
  analysis; use a container/VM that mounts only that view for a hard boundary.
- macOS gets environment scrubbing, an isolated working directory, and
  process-group timeouts. Portable kernel memory limits are enabled only on
  Linux because macOS does not reliably enforce `RLIMIT_AS` for Python/NumPy.
- Validation feedback can still be overfit over many iterations. The paired
  second-seed promotion check and convergence rules reduce—not eliminate—that
  risk; neither is a statistical confirmation procedure.
- BPR's two-seed mean improvement is encouraging but small. The final claim is
  determined only by the organizer's one-time hidden evaluation.
- This is a pipeline comparison, not a clean loss-only ablation: pointwise FM
  and pairwise FM perform different numbers of optimizer steps per epoch, so
  their effective update schedule and regularization also differ.
- The simplified knowledge pack is a research reference, not proof that every
  method will improve this dataset. Validation is the ground-truth check.

## Project layout

| Path | Purpose |
|---|---|
| `data.py` | Selective split loader, five-field encoder, streaming research signals |
| `baseline.py` | Random, popularity, and official FM baselines |
| `evaluate.py` | Frozen GAUC/nDCG@5 implementation |
| `submit.py` | Strict submission writer/checker; validation-only scoring |
| `agent/orchestrator.py` | Autonomous search, execution, scoring, recovery, resume/finalization |
| `agent/seed_pipeline.py` | Immutable iteration-0 pointwise FM |
| `agent/bpr_pipeline.py` | Built-in observed-negative BPR-style bootstrap operator |
| `agent/context.py` | Candidate contract and proposal policy |
| `agent/knowledge_base.md` | Bundled reference-only research notes |
| `agent/safety.py` | AST safety/contract tripwire |
| `results/validation_bpr/` | Sanitized hardened bootstrap evidence and exact executed BPR diff |
| `tests/` | Unit and integration contract tests |
| `README.md.bak` | Preserved, Git-ignored upstream starter notes; historical reference only |

The KuaiRand data retains its own license under `KuaiRand-Pure/LICENSE` and is
ignored by Git. Check the dataset's terms before publishing or redistributing
the CSV files. The current `README.md` governs this project; `README.md.bak` is
kept unchanged only as an archived copy of the original starter instructions.
