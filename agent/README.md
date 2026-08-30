# Autonomous agent loop for KuaiRand-Pure

This is the "robot" that actually decides what to try, writes the code, runs
it, and logs the result — proof of autonomy for the challenge deliverables.

## What it does, one iteration at a time

1. Shows Claude the current best `pipeline.py` plus a short history of what's
   been tried so far.
2. Claude replies with one hypothesis + a complete new `pipeline.py`.
3. The file is scanned for obviously dangerous code (`agent/safety.py` —
   network calls, shelling out, deleting files, unapproved imports), then run
   for real against the KuaiRand-Pure data with a timeout.
4. If it crashes, the error is sent back to Claude for up to 2 fix attempts
   before the iteration is logged as failed and the loop moves on from the
   last good version.
5. Every iteration — success or failure — gets one line appended to
   `run_log.jsonl`: hypothesis, code diff, metrics, errors/recovery, tokens
   used.
6. Stops when the validation score stalls (hasn't improved by more than 0.002
   over 3 iterations in a row), or after 50 iterations, or after 6 hours —
   whichever comes first.
7. Writes `best_pipeline.py`, `summary.json`, and a `submission.csv` (in the
   project root, in the exact format `submit.py` expects) from whichever
   iteration scored best on validation.

Iteration 0 isn't written by Claude — it's `seed_pipeline.py`, a faithful
reimplementation of the official FM baseline through this harness's I/O
contract. It exists to prove the harness reproduces the published baseline
numbers before any LLM-authored changes are layered on top.

## Setup

```bash
pip install -r agent/requirements.txt
```

Get an API key from https://console.anthropic.com and set it as an
environment variable (don't paste it into a chat or commit it anywhere):

```bash
# bash
export ANTHROPIC_API_KEY=sk-...
```
```powershell
# PowerShell
$env:ANTHROPIC_API_KEY = "sk-..."
```

## Run

```bash
python agent/orchestrator.py --max_iterations 50 --wall_clock_hours 6
```

Useful flags:
- `--model claude-sonnet-5` (default) — swap for a different model id if you want.
- `--fresh` — wipe `agent/iterations/` and `run_log.jsonl` and start over.
- `--data_dir path/to/KuaiRand-Pure/data` — if the data isn't in the default location.

Each run's artifacts land in `agent/`:
- `run_log.jsonl` — the deliverable run log (one JSON object per iteration).
- `iterations/iter_XXXX/` — that iteration's exact `pipeline.py`, `metrics.json`,
  and saved `valid_scores.npy` / `test_scores.npy`.
- `best_pipeline.py` — the winning code.
- `summary.json` — final scores, delta vs. baseline, token/wall-clock totals
  (this is what the "resource usage" deliverable is built from).
- `../submission.csv` — generated from the best iteration's saved test scores.

## Known limitations (be upfront about these in your writeup)

- The safety check is a denylist, not a sandbox. It blocks the obvious stuff
  (subprocess, network, file deletion, unapproved imports) but a
  determined/broken model could still write a slow infinite loop, which the
  per-iteration subprocess timeout (20 min) catches, or excessive memory use,
  which it doesn't.
- Only numpy + the Python standard library are importable inside a generated
  `pipeline.py` (matching the starter kit's own numpy-only constraint) — no
  auto `pip install`, so ideas that need torch/lightgbm/etc. won't run as-is
  in this environment. The model is told this in its system prompt so it
  should stick to numpy implementations.
- The convergence rule is implemented against the *running best* validation
  score (has the best-so-far moved by more than 0.002 in the last 3
  iterations), which is one reasonable reading of the official ε/N rule but
  worth double-checking against how the organizers score it if it matters for
  your submission.
