# Woof Woof: Autonomous ML Research for KuaiRand-Pure

Woof Woof is a validation-first autonomous ML research agent for the TikTok
TechJam KuaiRand-Pure recommender benchmark. It reproduces the official
Factorization Machine (FM), proposes one focused improvement at a time, runs
candidate code under defense-in-depth controls, evaluates predictions with the
frozen organizer metric, records every decision, and retains only validated
improvements.

The task is **within-user ranking of logged video impressions** using the native
binary `long_view` label. The metrics are **GAUC** and **nDCG@5**, with
`primary = mean(GAUC, nDCG@5)`. The organizer-provided [evaluate.py](evaluate.py)
is unchanged.

> One legacy row in the supplied problem statement mentions `click` with
> nDCG@10/Recall@50. The repeated benchmark sections, starter code, evaluator,
> and submission format specify `long_view`, GAUC, and nDCG@5, so this project
> follows that executable contract.

## Results

All results below are from the fixed public validation split: 124,909 rows and
22,377 users. No hidden/test metric was evaluated.

| Pipeline | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Published validation baseline | 0.667400 | 0.535700 | 0.601600 |
| Reproduced pointwise FM | 0.667133 | 0.535805 | 0.601469 |
| Earlier observed-negative BPR | 0.669711 | 0.537081 | 0.603396 |
| Promoted single model, seed 0 | 0.671301 | 0.538250 | 0.604776 |
| **Promoted three-seed ensemble** | **0.672333** | **0.538648** | **0.605490** |

| Absolute improvement | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Versus reproduced FM | +0.005201 | +0.002843 | **+0.004022** |
| Versus published validation baseline | +0.004933 | +0.002948 | **+0.003890** |

The selected model is an ensemble of three observed-negative, same-user BPR
factorization machines. Each component uses:

- learning rate `0.0005`;
- 30 duration quantiles fitted on training data only;
- a four-hour categorical field with a training-only vocabulary;
- smoothed user-by-tab affinity calculated from training outcomes only; and
- within-user score standardization before label-free ensemble averaging.

The three component primary scores were `0.604776`, `0.604900`, and `0.605273`.
The ensemble improved both GAUC and nDCG@5 over every individual component.

### Autonomous campaign resources

The reported breadth campaign is `improvement-campaign-20260901-02`.

| Field | Recorded value |
|---|---:|
| Search mode / stop reason | `campaign` / `campaign_converged` |
| Nodes evaluated | 7 of 50 |
| Official convergence first detected | Iteration 4 |
| Best node | Iteration 1 |
| Agent wall-clock | 1,588.1 s (26m 28.1s) |
| OpenAI input tokens | 51,282 |
| OpenAI output tokens | 31,804 |
| Total OpenAI tokens | 83,086 |
| GPU-hours | 0.0 |
| Failed model nodes | 0 |
| Code-repair recoveries | 0 |
| Automatically retried API timeouts | 6 |
| Manual interventions after launch | 0 |
| Final-output generation | 92.9 s additional |
| Total wall-clock through finalization | 1,681.3 s (28m 1.3s) |

The LLM-proposed post-bootstrap candidates did not displace iteration 1. The
agent's contribution in this run was to autonomously test, reject, and log them
while preserving the accepted winner. Development and model selection remained
validation-only. The selected source was subsequently finalized once against a
masked test-feature view, producing 170,588 aligned scores without computing a
hidden/test metric. KuaiRand-1K and KuaiRand-27K were not attempted.

## How the autonomous loop works

1. **Trust anchor.** Iteration 0 runs an immutable pointwise FM and checks each
   validation metric against the published baseline tolerance.
2. **Ranking-aligned bootstrap.** Iteration 1 runs the promoted three-seed BPR
   ensemble without making an API call.
3. **Research proposal.** From iteration 2 onward, the OpenAI Responses API sees
   the accepted parent, prior experiment registry, and reference-only knowledge
   pack, then proposes exactly one hypothesis and code change.
4. **Guarded execution.** Candidate code receives a generated train/validation
   data view with no test rows or random-exposure log. Its environment is
   secret-scrubbed; AST checks, private working directories, output validation,
   timeouts, and process-group termination provide defense in depth.
5. **Trusted evaluation.** Candidates emit only aligned scores. The orchestrator
   validates their shape/type/finiteness and calls the frozen evaluator itself.
   Candidate code cannot declare its own winning metric.
6. **Reflect and revise.** A gain is promoted; a regression is rejected. Small
   gains receive a second-seed confirmation. Failed code can receive two repair
   attempts, and interrupted runs can resume from durable state.
7. **Convergence.** Official mode follows the organizer's `epsilon=0.002`,
   `N=3` rule. Explicit `--campaign` mode additionally requires one successfully
   evaluated node from every priority family, subject to the same 50-node and
   six-hour caps. Rejected code is never selected as a future parent.

The campaign covered loss alignment, causal history, multitask learning,
censored watch-time modeling, temporal drift, and architecture. The five
post-bootstrap candidates scored between `0.604320` and `0.605295`, below the
accepted `0.605490`, and were rolled back from the production path.

This is not an OS-level security sandbox. For hostile generated code, run the
whole agent in a disposable container or VM that mounts only the generated data
view. Linux receives additional CPU/file/memory limits; macOS does not reliably
enforce the same NumPy memory boundary.

## Evidence and run logs

The public, validation-only evidence is intentionally separated from large
local run artifacts:

- [Devpost Markdown draft](DEVPOST.md): paste-ready project description,
  results table, resource report, and technology disclosures.
- [Full autonomous run log](results/improvement_campaign/run_log.jsonl): seven
  JSONL entries containing each hypothesis, exact diff, metrics, attempts,
  errors/API events, selection decision, and token use.
- [Campaign summary](results/improvement_campaign/summary.json): best metrics,
  deltas, resources, finalization status, and hidden-test status.
- [Decision ledger](results/improvement_campaign/decision_log.jsonl): compact
  accepted/rejected history, including pre-campaign ablations.
- [Campaign report](results/improvement_campaign/README.md): readable result and
  method summary.
- [Selected campaign pipeline](results/improvement_campaign/best_pipeline.py):
  exact validation-selected source archived with the evidence. The production
  [BPR pipeline](agent/bpr_pipeline.py) adds a direct-test masked-view guard
  without changing the validation path.
- [Earlier BPR evidence](results/validation_bpr/README.md): preserved first BPR
  milestone and executed diff.

Local `agent/runs/` directories contain data views, score arrays, and execution
artifacts and are therefore ignored by Git. The sanitized files above contain
the required audit trail without raw data, secrets, or submissions.

## Setup

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and extract the organizer-provided KuaiRand-Pure release from the
repository root:

```bash
curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

The required files are:

```text
KuaiRand-Pure/data/
├── log_standard_4_08_to_4_21_pure.csv
├── log_standard_4_22_to_5_08_pure.csv
├── user_features_pure.csv
├── video_features_basic_pure.csv
└── video_features_statistic_pure.csv
```

The dataset and generated run directories are ignored by Git. Check local
readiness without training:

```bash
python agent/preflight.py
```

Iterations 0–1 need no API key. For later autonomous proposals:

```bash
cp .env.example .env
chmod 600 .env
# Replace only the placeholder in .env; never commit or share the real key.
python agent/preflight.py --require_llm
```

The completed campaign used `gpt-5.6-sol`. `OPENAI_MODEL` can be changed to a
Responses API model available to the user's OpenAI project.

## Verify and reproduce

### 1. Run the test suite

```bash
python -m unittest discover -s tests -v
```

Expected result for this revision: **93 tests pass**.

### 2. Reproduce the official FM

```bash
python baseline.py --model fm
```

Expected validation primary is approximately `0.6015–0.6016`.

### 3. Reproduce the selected model without an API call

Use a new, empty run directory:

```bash
python agent/orchestrator.py \
  --max_iterations 2 \
  --run_dir /tmp/woof-reproduction

python -m json.tool /tmp/woof-reproduction/summary.json
```

Expected validation metrics are approximately GAUC `0.672333`, nDCG@5
`0.538648`, and primary `0.605490`. This two-node reproduction takes about
103 seconds on the reference Mac and stops at the requested iteration cap; it
is not the seven-node converged breadth campaign.

### 4. Run a fresh autonomous campaign (optional and paid)

```bash
python agent/orchestrator.py \
  --campaign \
  --max_iterations 50 \
  --wall_clock_hours 6 \
  --run_dir agent/runs/<new-run-id>
```

Later nodes use the configured OpenAI model and may incur API charges. Resume a
campaign with the same mode:

```bash
python agent/orchestrator.py \
  --campaign \
  --resume agent/runs/<run-id>
```

## Final output

The selected campaign run was finalized exactly once with:

```bash
python agent/orchestrator.py \
  --campaign \
  --resume agent/runs/improvement-campaign-20260901-02 \
  --finalize

python submit.py --check --split test \
  agent/runs/improvement-campaign-20260901-02/submission.csv
```

The checker passed for all **170,588** rows. Submission SHA-256:

```text
03b8bc99cbc6f391ad3fd8a80bc2da2ce3cab8d4bf011fcc8c844a39b3c35de2
```

Finalization executes the exact saved winner against a generated final view in
which every test outcome is masked before candidate code runs. It produces
scores but never computes hidden/test metrics. Do not run `submit.py --score`
on the test split; that operation is rejected by design.

The model artifact is reproducible source plus its fixed configuration/seed and
data order rather than serialized NumPy weights. The generated `submission.csv`
is ignored by Git and must be uploaded separately wherever the organizers
request it. Sanitized finalization metadata is archived in
[results/improvement_campaign/finalization.json](results/improvement_campaign/finalization.json).

## Tools, APIs, libraries, and data

| Category | Used |
|---|---|
| Development tools | VS Code-compatible editor/Codex workspace, terminal, Python CLI, Git and GitHub; Claude Code was also used during development/debugging |
| Runtime API | OpenAI Responses API through the `openai` Python SDK; completed campaign model: `gpt-5.6-sol` |
| Numerical/runtime libraries | NumPy; no PyTorch, pandas, scikit-learn, RecBole, or pretrained weights |
| Tests | Python standard-library `unittest` |
| Training data | Organizer-provided KuaiRand-Pure standard logs and user/video files only |
| Research context | Human-curated [knowledge pack](agent/knowledge_base.md), treated as reference material rather than training data or executable instructions |

No external training data or pretrained model weights are used. The
random-exposure log is deliberately excluded from candidate development views.
Only KuaiRand-Pure was attempted; the 1K and 27K bonus benchmarks were not.

## Team contributions

Work was divided between agent/model development and the research and reporting
needed to explain and support the experiments.

| Team member | Contribution |
|---|---|
| Tamanna Hasan | Co-led development of the autonomous ML agent and recommender pipeline, including implementation, model training, validation runs, and testing. |
| Aaron Mari Santos Solis | Co-led development and model experimentation, including pipeline improvements, model training, validation analysis, and testing. |
| Ooi Ky Shen | Co-authored the project report and researched established recommender-system methods and sources for the team knowledge base. |
| Ranjanaa Baskaran | Co-authored the project report and contributed literature research and synthesis for the team knowledge base. |

## Limitations and future work

- Repeated decisions on one public validation split can overfit that split.
  Seed confirmation and convergence reduce but do not eliminate this risk.
- The winning iteration was a built-in, development-validated operator; the
  later LLM proposals broadened the search but did not improve it.
- The FM/BPR comparison changes both loss and optimizer-update schedules, so it
  is a pipeline comparison rather than a clean loss-only ablation.
- Guarded subprocess execution is defense in depth, not a hard sandbox.
- The current in-memory loader targets KuaiRand-Pure and does not scale directly
  to the 1K/27K bonus datasets.
- The project stores reproducible model source rather than serialized weights.

Given more time, we would add nested or rolling validation to reduce adaptive
overfitting, containerize generated-code execution, serialize model checkpoints,
build streaming/memory-mapped bonus-dataset adapters, and revisit stronger
causal sequence models only after establishing a useful pooled-history signal.

## Project layout

| Path | Purpose |
|---|---|
| `data.py` | Split-aware loading, duration encoding, and research-signal streaming |
| `baseline.py` | Random, popularity, and official FM baselines |
| `evaluate.py` | Frozen organizer GAUC/nDCG@5 evaluator |
| `submit.py` | Submission writer/checker and validation-only scorer |
| `agent/orchestrator.py` | Autonomous search, execution, evaluation, recovery, resume, and finalization |
| `agent/bpr_pipeline.py` | Selected three-seed BPR-FM ensemble |
| `agent/context.py` | Candidate contract and research prompt construction |
| `agent/knowledge_base.md` | Reference-only research notes |
| `agent/experiment_memory.md` | Durable validation-only experiment evidence |
| `agent/experiments/` | Isolated rejected experiment implementations |
| `agent/safety.py` | Candidate AST and output-contract checks |
| `DEVPOST.md` | Paste-ready formatted Devpost report |
| `results/improvement_campaign/` | Sanitized final validation evidence and logs |
| `tests/` | Unit and integration contract tests |

## Data terms and project license

KuaiRand retains its own terms in `KuaiRand-Pure/LICENSE`; review them before
redistributing any dataset files. Raw data is not part of this repository.

This repository currently has no top-level code license. Add an appropriate
`LICENSE` before describing the project code as open source or granting reuse
rights.
