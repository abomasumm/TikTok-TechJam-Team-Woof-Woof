# Woof Woof: Autonomous ML Research for KuaiRand-Pure

**Repository:** [TikTok-TechJam-Team-Woof-Woof](https://github.com/abomasumm/TikTok-TechJam-Team-Woof-Woof)

## How our solution addresses the problem statement

The challenge asks for an autonomous ML research agent that executes the
complete MLE iteration loop: understand the task, inspect data, engineer
features, train, evaluate, reflect, and revise, with minimal human intervention.
Our agent, `agent/orchestrator.py`, implements this process end-to-end.

### The autonomous workflow

1. **Reproduce the baseline.** Iteration 0 is an immutable reproduction of the
   official pointwise Factorization Machine baseline. It acts as a trust anchor:
   the autonomous search cannot modify it, and the harness verifies each
   validation metric against the published baseline before continuing.

2. **Start with a ranking-aligned model.** Iteration 1 runs our promoted
   built-in model without an API call: a three-seed ensemble of
   observed-negative, same-user BPR factorization machines. Each component uses
   a learning rate of `0.0005`, 30 duration quantiles fitted on training data
   only, a training-vocabulary four-hour field, and smoothed user-by-tab
   affinity computed only from training outcomes. Component predictions are
   standardized within each user and averaged without using evaluation labels.

3. **Propose one research change.** From iteration 2 onward, the OpenAI
   Responses API proposes exactly one focused hypothesis and code change per
   iteration. It receives the accepted parent, previous experiment results, and
   a human-curated research knowledge pack.

4. **Run candidates safely.** Candidates run in guarded subprocesses with
   train/validation-only data views, secret-scrubbed environments, AST checks,
   private working directories, output validation, and process timeouts. These
   controls provide defense in depth, although they are not an OS-level
   security sandbox.

5. **Evaluate, retain, and recover.** The orchestrator independently evaluates
   every candidate using the frozen organizer evaluator. Improvements are
   promoted; regressions are rejected and can never become future parents.
   Small gains receive second-seed confirmation. Failed code can receive up to
   two automatic repair attempts. Every iteration records its hypothesis,
   source references, exact code diff, metrics, execution attempts, errors, API
   events, token usage, and selection decision.

6. **Stop autonomously.** Official mode follows the organizer's
   `epsilon=0.002`, `N=3` convergence rule exactly. Our reported `--campaign`
   run additionally required one successfully evaluated candidate from every
   priority family: loss alignment, causal history, multitask learning,
   censored watch-time modeling, temporal drift, and architecture. Official
   convergence first occurred at iteration 4; campaign mode continued through
   iteration 6 to complete coverage and stopped after seven total nodes.

7. **Finalize once.** Finalization is a separate one-time action. It executes
   the exact validation-selected source against a generated final view in which
   all test outcomes are masked before candidate code runs. It produces
   submission scores but never computes a local hidden-test metric.

## Results

All model-selection results are from the fixed public validation split.

| Pipeline | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Published validation baseline | 0.667400 | 0.535700 | 0.601600 |
| Reproduced pointwise FM | 0.667133 | 0.535805 | 0.601469 |
| Earlier observed-negative BPR | 0.669711 | 0.537081 | 0.603396 |
| **Validation-selected three-seed ensemble** | **0.672333** | **0.538648** | **0.605490** |

| Absolute improvement | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Versus reproduced FM | +0.005201 | +0.002843 | **+0.004022** |
| Versus published validation baseline | +0.004933 | +0.002948 | **+0.003890** |

The five post-bootstrap LLM candidates explored causal history, click
pretraining, censored watch-time pretraining, temporal weighting, and a
field-pair-gated FM architecture. None exceeded `0.605490`, so the agent
correctly rejected them and retained iteration 1.

### Resource usage and autonomy

| Resource | Recorded value |
|---|---:|
| Iterations | 7 of 50 |
| Converged campaign wall-clock | 1,588.1 s (26m 28.1s) |
| OpenAI input tokens | 51,282 |
| OpenAI output tokens | 31,804 |
| Total OpenAI tokens | 83,086 |
| GPU-hours | 0.0 |
| Failed model nodes | 0 |
| Candidate-code repair events | 0 |
| Automatically retried API timeouts | 6 |
| Manual interventions after launch | 0 |

### Final output status

- Finalized exactly once from the validation-selected iteration-1 source.
- Generated **170,588** aligned test scores.
- Starter-kit submission schema/alignment check: **passed**.
- Submission SHA-256:
  `03b8bc99cbc6f391ad3fd8a80bc2da2ce3cab8d4bf011fcc8c844a39b3c35de2`.
- Hidden/test metrics computed locally: **no**.
- Bonus KuaiRand-1K and KuaiRand-27K benchmarks attempted: **no**.

## Development tools used

A terminal/CLI workflow with a VS Code-compatible editor, Python, Git, and
GitHub. OpenAI Codex assisted with repository and agent development.
Claude Code/Anthropic was used for development-time debugging and root-cause
analysis of runtime issues; it is not a dependency or runtime API used by the
deployed agent.

## APIs used

The OpenAI Responses API is the only runtime API called by the autonomous
agent. It is used for research proposals and code-repair requests from
iteration 2 onward through the OpenAI Python SDK. The completed breadth
campaign used `gpt-5.6-sol`. Iterations 0–1 make no API calls. The recommendation
model itself is the NumPy BPR-FM ensemble, not an OpenAI model.

## Libraries and frameworks used

Deliberately minimal: NumPy is the only numerical dependency used by the
recommender pipeline. The OpenAI Python SDK is the only agent-API dependency.
Testing uses Python's standard `unittest`; the final revision passes 93 unit and
integration tests. No PyTorch, pandas, scikit-learn, RecBole, or pretrained
weights are used.

## Datasets and assets used

The organizer-provided **KuaiRand-Pure** dataset (Kuaishou, via the official
Zenodo release), specifically the two standard interaction logs and the
user/video feature files, split by date into fixed train/validation/test
windows as specified by the challenge.

The randomized-exposure log is deliberately excluded from the agent's
development view to prevent leakage. No external training data or pretrained
model weights are used anywhere.

The agent also receives a small, human-curated reference document,
`agent/knowledge_base.md`, summarizing relevant published methods—BPR, DIN,
SIM, ESMM, SASRec, CWM, and AIDE's tree-search framing—as background research
context. It is explicitly reference material, not training data, and never
overrides the task's evaluation contract.
