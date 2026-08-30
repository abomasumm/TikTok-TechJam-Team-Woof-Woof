"""Autonomous ML research agent orchestrator for KuaiRand-Pure.

Each iteration:
  1. Ask Claude for ONE hypothesis + a full replacement pipeline.py, built on
     top of the current best-known code.
  2. Static safety check the returned code (agent/safety.py).
  3. Run it as a subprocess against the real data, with a timeout.
  4. If it errors, feed the error back to Claude for up to MAX_FIX_ATTEMPTS
     fix-it turns before giving up on that iteration.
  5. Score it, log everything (hypothesis, diff, metrics, errors/recovery,
     tokens, wall-clock) to run_log.jsonl.
  6. Stop on convergence (running-best hasn't improved by > EPS over the last
     N iterations), the iteration cap, or the wall-clock ceiling.

Usage:
    set ANTHROPIC_API_KEY=sk-...          (Windows cmd)
    $env:ANTHROPIC_API_KEY = "sk-..."     (PowerShell)
    export ANTHROPIC_API_KEY=sk-...       (bash)
    python agent/orchestrator.py --max_iterations 50 --wall_clock_hours 6
"""
import argparse, difflib, json, os, shutil, subprocess, sys, time, traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context import build_system_prompt
from safety import check_code

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
ITER_DIR = os.path.join(AGENT_DIR, 'iterations')
LOG_PATH = os.path.join(AGENT_DIR, 'run_log.jsonl')
SUMMARY_PATH = os.path.join(AGENT_DIR, 'summary.json')

EPS = 0.002
N_CONVERGE = 3
MAX_FIX_ATTEMPTS = 2
PIPELINE_TIMEOUT_SEC = 20 * 60

PROPOSE_TOOL = {
    "name": "propose_iteration",
    "description": "Propose the next pipeline.py iteration.",
    "input_schema": {
        "type": "object",
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": "One or two sentences: what you're changing and why you expect it to help."
            },
            "reasoning": {
                "type": "string",
                "description": "Brief technical reasoning (2-5 sentences)."
            },
            "code": {
                "type": "string",
                "description": "The COMPLETE contents of the new pipeline.py file, satisfying the CODE CONTRACT exactly."
            },
        },
        "required": ["hypothesis", "reasoning", "code"],
    },
}


def log_event(entry):
    with open(LOG_PATH, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_claude(client, model, system, messages, max_tokens=8192):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=[PROPOSE_TOOL],
        tool_choice={"type": "tool", "name": "propose_iteration"},
    )
    tool_block = next(b for b in resp.content if b.type == "tool_use")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return tool_block.input, usage


def build_user_prompt(current_code, history, best_score):
    hist_lines = []
    for h in history[-12:]:
        if h["status"] == "ok":
            hist_lines.append(
                f"- iter {h['iteration']}: \"{h['hypothesis']}\" -> "
                f"valid primary {h['metrics']['valid']['primary']:.4f} "
                f"(GAUC {h['metrics']['valid']['GAUC']:.4f}, nDCG@5 {h['metrics']['valid']['nDCG@5']:.4f})"
                f"{' [NEW BEST]' if h.get('is_best') else ''}"
            )
        else:
            hist_lines.append(f"- iter {h['iteration']}: \"{h['hypothesis']}\" -> FAILED ({h['status']}): {h.get('error','')[:200]}")
    hist_text = "\n".join(hist_lines) if hist_lines else "(no iterations yet -- this is the first one)"

    return f"""CURRENT BEST pipeline.py (valid primary = {best_score:.4f}):
```python
{current_code}
```

ITERATION HISTORY:
{hist_text}

Propose the next iteration via the propose_iteration tool."""


def build_fix_prompt(broken_code, error_text):
    return f"""Your last proposed pipeline.py failed to run. Fix it and resubmit the
COMPLETE corrected file via the propose_iteration tool (keep the same hypothesis
if it's still valid, or note briefly if the failure changes your approach).

BROKEN CODE:
```python
{broken_code}
```

ERROR:
```
{error_text[-4000:]}
```"""


def run_pipeline(code_path, out_dir, data_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, code_path, "--data_dir", data_dir, "--out_dir", out_dir, "--seed", str(seed)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=PIPELINE_TIMEOUT_SEC, env=env,
        )
    except subprocess.TimeoutExpired as e:
        return False, None, f"TIMEOUT after {PIPELINE_TIMEOUT_SEC}s\nstdout so far:\n{e.stdout}\nstderr so far:\n{e.stderr}", time.time() - t0

    elapsed = time.time() - t0
    if proc.returncode != 0:
        return False, None, f"exit code {proc.returncode}\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}", elapsed

    metrics_path = os.path.join(out_dir, "metrics.json")
    valid_scores_path = os.path.join(out_dir, "valid_scores.npy")
    test_scores_path = os.path.join(out_dir, "test_scores.npy")
    for p in (metrics_path, valid_scores_path, test_scores_path):
        if not os.path.exists(p):
            return False, None, f"expected output missing: {p}\nSTDOUT:\n{proc.stdout[-2000:]}", elapsed

    with open(metrics_path) as fh:
        metrics = json.load(fh)

    for split, path in (("valid", valid_scores_path), ("test", test_scores_path)):
        arr = np.load(path)
        if not np.all(np.isfinite(arr)):
            return False, None, f"{split}_scores.npy contains NaN/Inf", elapsed

    return True, metrics, None, elapsed


def try_iteration(client, model, code, out_dir, data_dir, seed, error_feedback_history):
    """Runs `code`; on failure asks Claude to fix it, up to MAX_FIX_ATTEMPTS times.
    Returns (final_code, ok, metrics, error, attempts_log, tokens)."""
    attempts = []
    tokens = {"input_tokens": 0, "output_tokens": 0}
    cur_code = code

    for attempt in range(MAX_FIX_ATTEMPTS + 1):
        ok_safety, reason = check_code(cur_code)
        if not ok_safety:
            attempts.append({"attempt": attempt, "status": "safety_blocked", "reason": reason})
            if attempt >= MAX_FIX_ATTEMPTS:
                return cur_code, False, None, f"safety check failed: {reason}", attempts, tokens
            fix_input, usage = call_claude(
                client, model, build_system_prompt(),
                [{"role": "user", "content": build_fix_prompt(cur_code, f"SAFETY CHECK REJECTED THIS CODE: {reason}")}],
            )
            tokens["input_tokens"] += usage["input_tokens"]; tokens["output_tokens"] += usage["output_tokens"]
            cur_code = fix_input["code"]
            continue

        attempt_out_dir = out_dir if attempt == 0 else f"{out_dir}_fix{attempt}"
        code_path = _write_code(attempt_out_dir, cur_code)
        ok, metrics, error, elapsed = run_pipeline(code_path, attempt_out_dir, data_dir, seed)
        attempts.append({"attempt": attempt, "status": "ok" if ok else "run_failed",
                          "elapsed_sec": elapsed, "error": None if ok else error})
        if ok:
            return cur_code, True, metrics, None, attempts, tokens

        if attempt >= MAX_FIX_ATTEMPTS:
            return cur_code, False, None, error, attempts, tokens

        fix_input, usage = call_claude(
            client, model, build_system_prompt(),
            [{"role": "user", "content": build_fix_prompt(cur_code, error)}],
        )
        tokens["input_tokens"] += usage["input_tokens"]; tokens["output_tokens"] += usage["output_tokens"]
        cur_code = fix_input["code"]

    return cur_code, False, None, "exhausted fix attempts", attempts, tokens


def _write_code(out_dir, code):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pipeline.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_ROOT, "KuaiRand-Pure", "data"))
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max_iterations", type=int, default=50)
    ap.add_argument("--wall_clock_hours", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="wipe agent/iterations and run_log.jsonl before starting")
    a = ap.parse_args()

    if a.fresh and os.path.exists(ITER_DIR):
        shutil.rmtree(ITER_DIR)
    if a.fresh and os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    os.makedirs(ITER_DIR, exist_ok=True)

    try:
        import anthropic
    except ImportError:
        print("Missing dependency: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in your environment first.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    run_start = time.time()
    history = []
    best_score = -1.0
    best_iter = None
    best_code = None
    best_curve = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}

    # --- Iteration 0: seed pipeline (official baseline reproduced through the harness) ---
    with open(os.path.join(AGENT_DIR, "seed_pipeline.py")) as fh:
        seed_code = fh.read()
    iter0_dir = os.path.join(ITER_DIR, "iter_0000")
    code_path = _write_code(iter0_dir, seed_code)
    ok, metrics, error, elapsed = run_pipeline(code_path, iter0_dir, a.data_dir, a.seed)
    if not ok:
        print(f"FATAL: seed pipeline (baseline reproduction) failed: {error}", file=sys.stderr)
        sys.exit(1)

    best_score = metrics["valid"]["primary"]
    best_iter, best_code = 0, seed_code
    best_curve.append(best_score)
    entry = {
        "iteration": 0, "timestamp": time.time(), "hypothesis": "seed: official FM baseline reproduced through harness",
        "reasoning": "establishes the reproduction point every later iteration branches from",
        "diff": None, "status": "ok", "metrics": metrics, "error": None, "attempts": [{"attempt": 0, "status": "ok", "elapsed_sec": elapsed}],
        "tokens_input": 0, "tokens_output": 0, "is_best": True,
    }
    history.append(entry); log_event(entry)
    print(f"[iter 0] seed reproduced: valid primary {best_score:.4f} "
          f"(GAUC {metrics['valid']['GAUC']:.4f}, nDCG@5 {metrics['valid']['nDCG@5']:.4f})")

    i = 1
    while i <= a.max_iterations:
        elapsed_hours = (time.time() - run_start) / 3600.0
        if elapsed_hours >= a.wall_clock_hours:
            print(f"Stopping: wall-clock ceiling of {a.wall_clock_hours}h reached.")
            break
        if len(best_curve) > N_CONVERGE and (best_curve[-1] - best_curve[-1 - N_CONVERGE]) <= EPS:
            print(f"Converged: running-best primary improved by <= {EPS} over last {N_CONVERGE} iterations.")
            break

        user_prompt = build_user_prompt(best_code, history, best_score)
        try:
            proposal, usage = call_claude(client, a.model, build_system_prompt(), [{"role": "user", "content": user_prompt}])
        except Exception as e:
            print(f"[iter {i}] LLM call failed: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        total_tokens["input_tokens"] += usage["input_tokens"]; total_tokens["output_tokens"] += usage["output_tokens"]

        hypothesis, reasoning, proposed_code = proposal["hypothesis"], proposal["reasoning"], proposal["code"]
        iter_dir = os.path.join(ITER_DIR, f"iter_{i:04d}")

        try:
            final_code, ok, metrics, error, attempts, fix_tokens = try_iteration(
                client, a.model, proposed_code, iter_dir, a.data_dir, a.seed, history,
            )
        except Exception:
            ok, metrics, error, attempts, fix_tokens = False, None, traceback.format_exc(), [], {"input_tokens": 0, "output_tokens": 0}
            final_code = proposed_code
        total_tokens["input_tokens"] += fix_tokens["input_tokens"]; total_tokens["output_tokens"] += fix_tokens["output_tokens"]

        diff = "\n".join(difflib.unified_diff(
            best_code.splitlines(), final_code.splitlines(),
            fromfile=f"iter_{best_iter:04d}/pipeline.py", tofile=f"iter_{i:04d}/pipeline.py", lineterm="",
        ))

        is_best = ok and metrics["valid"]["primary"] > best_score
        entry = {
            "iteration": i, "timestamp": time.time(), "hypothesis": hypothesis, "reasoning": reasoning,
            "diff": diff, "status": "ok" if ok else "failed", "metrics": metrics, "error": error,
            "attempts": attempts, "tokens_input": usage["input_tokens"] + fix_tokens["input_tokens"],
            "tokens_output": usage["output_tokens"] + fix_tokens["output_tokens"], "is_best": is_best,
        }
        history.append(entry); log_event(entry)

        if ok:
            print(f"[iter {i}] \"{hypothesis[:80]}\" -> valid primary {metrics['valid']['primary']:.4f}"
                  f"{'  <-- NEW BEST' if is_best else ''}")
            if is_best:
                best_score, best_iter, best_code = metrics["valid"]["primary"], i, final_code
            best_curve.append(best_score)
        else:
            print(f"[iter {i}] \"{hypothesis[:80]}\" -> FAILED after {len(attempts)} attempt(s): {str(error)[:200]}")
            best_curve.append(best_score)

        i += 1

    # --- Finalize: write best pipeline + submission.csv from its saved test scores ---
    best_pipeline_path = os.path.join(AGENT_DIR, "best_pipeline.py")
    with open(best_pipeline_path, "w", encoding="utf-8") as fh:
        fh.write(best_code)

    best_iter_dir = os.path.join(ITER_DIR, f"iter_{best_iter:04d}")
    test_scores = np.load(os.path.join(best_iter_dir, "test_scores.npy"))

    sys.path.insert(0, PROJECT_ROOT)
    from data import load as load_data
    splits = load_data(a.data_dir)
    submission_path = os.path.join(PROJECT_ROOT, "submission.csv")
    with open(submission_path, "w", newline="", encoding="utf-8") as fh:
        import csv
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (x, s) in enumerate(zip(splits["test"], test_scores)):
            w.writerow([row_id, x[1], x[2], f"{float(s):.6g}"])

    with open(os.path.join(best_iter_dir, "metrics.json")) as fh:
        best_metrics = json.load(fh)

    summary = {
        "best_iteration": best_iter,
        "iterations_run": i - 1,
        "wall_clock_seconds": time.time() - run_start,
        "total_tokens": total_tokens,
        "best_valid_metrics": best_metrics["valid"],
        "best_test_metrics_local": best_metrics["test"],
        "baseline_test_primary": 0.5946,
        "delta_vs_baseline_local_test_primary": best_metrics["test"]["primary"] - 0.5946,
        "submission_path": submission_path,
        "best_pipeline_path": best_pipeline_path,
        "run_log_path": LOG_PATH,
    }
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== RUN COMPLETE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
