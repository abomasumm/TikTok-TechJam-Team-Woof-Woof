# Autonomous agent internals

See the root [README](../README.md) for setup and the complete testing/runbook.

The agent has three proposal layers:

1. `seed_pipeline.py` reproduces the immutable pointwise FM baseline.
2. `bpr_pipeline.py` is the promoted built-in ranking ensemble.
3. An OpenAI model proposes later single-change candidates through the
   Responses API, using the contract in `context.py` and reference notes in
   `knowledge_base.md` plus validation-only memory in `experiment_memory.md`.

Candidates write only an aligned `scores.npy` plus optional diagnostics. The
orchestrator independently validates and evaluates validation scores, records
the tree and code diff, repairs runtime failures, applies a paired second-seed
promotion check to sub-`0.002` gains, and checkpoints after every durable log
entry. That two-seed check is a search heuristic, not a significance test.

The bootstrap uses textbook BPR loss with benchmark-adapted observed-negative
sampling. Each eligible positive is paired with a uniformly sampled logged
zero-label impression from the same user; unexposed items are never negatives.
The promoted version adds 30 train-fitted duration buckets, a train-vocabulary
four-hour field, and train-only user-by-tab calibration. It trains seeds 0, 1,
and 2 and averages their per-user standardized scores. Its reproduced
validation metrics are GAUC `0.672333`, nDCG@5 `0.538648`, primary `0.605490`.
This remains a validation result, not a hidden-test claim.

Research candidates are passed a generated data view with no test rows.
One-time finalization uses the exact saved winner and a separate view in which
every test outcome is masked. It writes a submission but never calculates test
metrics. The promoted pipeline refuses a direct `--target_split test` run unless
the data directory has the orchestrator's masked final-view manifest. The
AST/subprocess layer is best-effort; use a container or VM that mounts only the
generated view if generated code must be treated as hostile.

Quick validation-only run with no API key (iterations 0 and 1 only):

```bash
python agent/orchestrator.py --max_iterations 2 --run_dir /tmp/woof-bpr-run
```

This now takes roughly 1.5–2 minutes on the reference Mac because iteration 1
trains three ensemble components.

Full autonomous run:

```bash
cp .env.example .env
chmod 600 .env
# In .env, replace OPENAI_API_KEY with your real key.
# OPENAI_MODEL defaults to gpt-5.6-sol; change it if unavailable.
python agent/preflight.py --require_llm
python agent/orchestrator.py --max_iterations 50 --wall_clock_hours 6
```

That command is official mode and stops on the organizer's exact convergence
rule. For deliberately exhaustive development breadth, use a separate run:

```bash
python agent/orchestrator.py --campaign --max_iterations 50 \
  --wall_clock_hours 6 --run_dir agent/runs/campaign-01
```

Campaign mode requires a successfully evaluated node from each priority family
before it may stop for convergence. Failures do not count as coverage; after
two proposal/runtime failures the scheduler moves on. All proposals branch from
the accepted winner, so rejected code cannot silently become a parent.

The repository-root `.env` is loaded automatically, and `--model <model-id>`
overrides `OPENAI_MODEL` for one command. Never put the real key in source,
chat, logs, screenshots, or a commit. Nodes after iteration 1 require OpenAI;
`--max_iterations 2` does not call the API. See the
[official OpenAI Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

Resume:

```bash
python agent/orchestrator.py --resume agent/runs/<run-id>
```

Add `--campaign` when resuming a campaign run. Changing modes for an existing
run is rejected.

Finalize only the run selected for submission, using the same mode it was
created with:

```bash
# Official-mode run
python agent/orchestrator.py --resume agent/runs/<run-id> --finalize

# Campaign-mode run
python agent/orchestrator.py --campaign \
  --resume agent/runs/<run-id> --finalize
```

Finalization creates predictions but does not evaluate hidden/test labels.

This is defense-in-depth execution, not a hard security sandbox. The API key is
removed from child environments, imports/calls are AST-checked, subprocesses
have timeouts and process-group termination, and Linux gets resource limits.
For high-assurance execution, place the whole runner inside a disposable
container or VM.
