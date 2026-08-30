"""Autonomous, validation-only research loop for KuaiRand-Pure.

Generated candidates emit scores, never authoritative metrics. This harness
validates the artifact and calls the frozen evaluator itself. Iteration 0 is the
official pointwise FM; iteration 1 is a vetted observed-negative BPR-style
bootstrap unless disabled; later iterations are proposed and repaired by an
OpenAI model through the Responses API.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
RUNS_DIR = AGENT_DIR / "runs"
BASELINE_PIPELINE = AGENT_DIR / "seed_pipeline.py"
BPR_PIPELINE = AGENT_DIR / "bpr_pipeline.py"
BASELINE_SCORES = PROJECT_ROOT / "baseline_scores.json"

sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from context import build_system_prompt  # noqa: E402
try:
    from .config import OpenAIConfig, load_openai_config  # noqa: E402
except ImportError:  # Direct execution: python agent/orchestrator.py
    from config import OpenAIConfig, load_openai_config  # noqa: E402
from safety import check_code  # noqa: E402


PROPOSE_TOOL = {
    "type": "function",
    "name": "propose_iteration",
    "description": "Propose one complete validation-only pipeline iteration.",
    "parameters": {
        "type": "object",
        "properties": {
            "parent_id": {"type": "integer"},
            "operation": {
                "type": "string",
                "enum": ["draft", "improve", "debug", "combine"],
            },
            "direction": {"type": "string"},
            "hypothesis": {"type": "string"},
            "reasoning": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "code": {"type": "string"},
        },
        "additionalProperties": False,
        "required": [
            "parent_id",
            "operation",
            "direction",
            "hypothesis",
            "reasoning",
            "sources",
            "code",
        ],
    },
    "strict": True,
}

DIRECTION_PRIORITY = (
    "loss/ranking",
    "history/causal",
    "multitask/training_only",
    "watch_time/censored",
    "temporal/drift",
    "architecture",
)

MAX_SCORE_BYTES_PER_ROW = 16
MAX_SCORE_HEADER_BYTES = 64 * 1024
RUNTIME_FILES = ("data.py", "evaluate.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_create_json(path, payload):
    """Durably create *path* exactly once, failing if it already exists."""
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # A partial marker is intentionally retained: it still records that the
        # one-time operation began and prevents an unsafe retry.
        raise
    if os.name == "posix":
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass


class RunLock:
    """Non-blocking, process-scoped advisory lock for one run directory."""

    def __init__(self, run_dir):
        self.path = Path(run_dir) / ".run.lock"
        self._handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            elif _msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                _msvcrt.locking(handle.fileno(), _msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - unsupported Python platform
                raise RuntimeError("this platform has no supported file-lock API")
        except (OSError, RuntimeError) as exc:
            handle.close()
            raise RuntimeError(f"run is already active or cannot be locked: {self.path}") from exc
        self._handle = handle
        return self

    def release(self):
        if self._handle is None:
            return
        try:
            if _fcntl is not None:
                _fcntl.flock(self._handle.fileno(), _fcntl.LOCK_UN)
            elif _msvcrt is not None:  # pragma: no cover - Windows
                self._handle.seek(0)
                _msvcrt.locking(self._handle.fileno(), _msvcrt.LK_UNLCK, 1)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.release()


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path):
    entries = []
    if not Path(path).exists():
        return entries
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt run log at line {line_number}: {exc}") from exc
    return entries


def validate_scores(path, expected_rows):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required artifact missing: {path}")
    maximum_bytes = MAX_SCORE_HEADER_BYTES + max(0, int(expected_rows)) * MAX_SCORE_BYTES_PER_ROW
    try:
        artifact_bytes = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot stat scores.npy safely: {exc}") from exc
    if artifact_bytes > maximum_bytes:
        raise ValueError(
            f"scores.npy exceeds size limit ({artifact_bytes:,} > {maximum_bytes:,} bytes)"
        )
    try:
        scores = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot load scores.npy safely: {exc}") from exc
    if scores.ndim != 1:
        raise ValueError(f"scores.npy must be one-dimensional, got shape {scores.shape}")
    if len(scores) != expected_rows:
        raise ValueError(
            f"scores.npy has {len(scores):,} rows; expected exactly {expected_rows:,}"
        )
    if scores.dtype.kind not in "fiu":
        raise ValueError(f"scores.npy must be numeric, got dtype {scores.dtype}")
    scores = scores.astype(np.float64, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores.npy contains NaN or infinite values")
    return scores


def read_diagnostics(path, max_bytes=5_000_000):
    path = Path(path)
    if not path.exists():
        return {}
    if path.stat().st_size > max_bytes:
        raise ValueError("diagnostics.json exceeds the 5 MB limit")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("diagnostics.json must contain a JSON object")
    forbidden = ("test_metric", "heldout_metric", "test_label", "heldout_label")
    lowered = json.dumps(payload, sort_keys=True).lower()
    if any(name in lowered for name in forbidden):
        raise ValueError("diagnostics.json appears to contain held-out labels or metrics")
    return payload


def score_validation(validation_rows, scores):
    from evaluate import evaluate

    labels = np.asarray([row[6] for row in validation_rows], dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if np.array_equal(scores, labels):
        raise ValueError("scores exactly equal validation labels; probable label leakage")
    # Direct label transforms evade an equality-only check (for example
    # ``2 * labels`` or ``labels + 1e-10 * noise``). With enough examples in
    # both classes, reject outputs whose within-class residual is negligible
    # relative to the between-class gap. This is a narrow tripwire; arbitrary
    # leakage cannot be proven or excluded from scores alone.
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) >= 2 and len(negative) >= 2:
        positive_mean = float(np.mean(positive))
        negative_mean = float(np.mean(negative))
        gap = abs(positive_mean - negative_mean)
        residual = max(
            float(np.max(np.abs(positive - positive_mean))),
            float(np.max(np.abs(negative - negative_mean))),
        )
        scale = max(gap, float(np.ptp(scores)), np.finfo(np.float64).eps)
        if gap > 0 and residual <= max(1e-12, scale * 1e-7):
            raise ValueError(
                "scores are an almost-exact transform of validation labels; "
                "probable label leakage"
            )
    metrics = evaluate(
        [row[1] for row in validation_rows],
        labels,
        scores,
    )
    return native_metrics(metrics)


def child_environment(seed, runtime_dir=None, temporary_dir=None):
    allowed_names = (
        "PATH",
        "LANG",
        "LC_ALL",
        "SYSTEMROOT",
        "WINDIR",
    )
    environment = {key: os.environ[key] for key in allowed_names if key in os.environ}
    environment.update(
        {
            "PYTHONPATH": str(Path(runtime_dir or PROJECT_ROOT).resolve()),
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if temporary_dir is not None:
        private_tmp = str(Path(temporary_dir).resolve())
        environment.update({"TMPDIR": private_tmp, "TMP": private_tmp, "TEMP": private_tmp})
    return environment


def resource_limiter(timeout_seconds, memory_limit_mb):
    if os.name != "posix" or _resource is None:
        return None

    def apply_limits():
        cpu = max(1, int(timeout_seconds) + 5)
        for resource_name, limit in (
            ("RLIMIT_CPU", cpu),
            ("RLIMIT_FSIZE", 512 * 1024 * 1024),
        ):
            try:
                _resource.setrlimit(getattr(_resource, resource_name), (limit, limit))
            except (AttributeError, OSError, ValueError):
                pass
        if platform.system() == "Linux" and memory_limit_mb > 0:
            try:
                memory = int(memory_limit_mb) * 1024 * 1024
                _resource.setrlimit(_resource.RLIMIT_AS, (memory, memory))
            except (AttributeError, OSError, ValueError):
                pass

    return apply_limits


def terminate_process_tree(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def tail(path, limit=4000):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_candidate(
    code_path,
    attempt_dir,
    data_dir,
    seed,
    target_split,
    expected_rows,
    timeout_seconds,
    memory_limit_mb,
    runtime_dir=None,
):
    attempt_dir = Path(attempt_dir)
    artifact_dir = attempt_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = attempt_dir / "tmp"
    temporary_dir.mkdir(mode=0o700, exist_ok=True)
    try:
        os.chmod(temporary_dir, 0o700)
    except OSError:
        pass
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    command = [
        sys.executable,
        str(Path(code_path).resolve()),
        "--data_dir",
        str(Path(data_dir).resolve()),
        "--out_dir",
        str(artifact_dir.resolve()),
        "--seed",
        str(seed),
        "--target_split",
        target_split,
    ]
    original_code_hash = sha256_file(code_path)
    started = time.time()
    with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=attempt_dir,
                env=child_environment(seed, runtime_dir, temporary_dir),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=(os.name == "posix"),
                preexec_fn=resource_limiter(timeout_seconds, memory_limit_mb),
                close_fds=True,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"failed to start candidate process: {exc}",
                "elapsed_seconds": time.time() - started,
                "stdout_tail": tail(stdout_path),
                "stderr_tail": tail(stderr_path),
            }
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate_process_tree(process)
            process.wait()
            return {
                "ok": False,
                "error": f"TIMEOUT after {timeout_seconds:.1f}s",
                "elapsed_seconds": time.time() - started,
                "stdout_tail": tail(stdout_path),
                "stderr_tail": tail(stderr_path),
            }

    elapsed = time.time() - started
    if return_code != 0:
        return {
            "ok": False,
            "error": f"candidate exited with code {return_code}",
            "elapsed_seconds": elapsed,
            "stdout_tail": tail(stdout_path),
            "stderr_tail": tail(stderr_path),
        }
    try:
        if sha256_file(code_path) != original_code_hash:
            raise ValueError("candidate modified its own source file")
        allowed_attempt_entries = {
            "pipeline.py",
            "stdout.log",
            "stderr.log",
            "artifacts",
            "tmp",
        }
        unexpected_attempt = sorted(
            path.name for path in attempt_dir.iterdir() if path.name not in allowed_attempt_entries
        )
        if unexpected_attempt:
            raise ValueError(f"candidate wrote unexpected files: {unexpected_attempt}")
        allowed_artifacts = {"scores.npy", "diagnostics.json"}
        unexpected_artifacts = sorted(
            str(path.relative_to(artifact_dir))
            for path in artifact_dir.rglob("*")
            if path.is_file()
            and str(path.relative_to(artifact_dir)) not in allowed_artifacts
        )
        if unexpected_artifacts:
            raise ValueError(f"candidate wrote unexpected artifacts: {unexpected_artifacts}")
        scores = validate_scores(artifact_dir / "scores.npy", expected_rows)
        diagnostics = read_diagnostics(artifact_dir / "diagnostics.json")
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "elapsed_seconds": elapsed,
            "stdout_tail": tail(stdout_path),
            "stderr_tail": tail(stderr_path),
        }
    return {
        "ok": True,
        "scores": scores,
        "diagnostics": diagnostics,
        "artifact_dir": str(artifact_dir),
        "elapsed_seconds": elapsed,
        "stdout_tail": tail(stdout_path),
        "stderr_tail": tail(stderr_path),
    }


def promote_attempt(iteration_dir, code, result, attempt_number):
    iteration_dir = Path(iteration_dir)
    (iteration_dir / "pipeline.py").write_text(code, encoding="utf-8")
    source_dir = Path(result["artifact_dir"])
    shutil.copy2(source_dir / "scores.npy", iteration_dir / "valid_scores.npy")
    if (source_dir / "diagnostics.json").exists():
        shutil.copy2(source_dir / "diagnostics.json", iteration_dir / "diagnostics.json")
    atomic_json(
        iteration_dir / "accepted_attempt.json",
        {"attempt": attempt_number, "promoted_at": utc_now()},
    )


class OpenAIProposer:
    def __init__(self, model=None, api_retries=3, max_tokens=8192, config=None):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is missing; install agent/requirements.txt"
            ) from exc
        config = config or load_openai_config(model=model)
        if model and model != config.model:
            config = OpenAIConfig(api_key=config.api_key, model=model)
        if not config.has_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM iterations")
        if not config.has_model:
            raise RuntimeError("set --model or OPENAI_MODEL for LLM iterations")
        self.client = openai.OpenAI(
            api_key=config.api_key,
            timeout=90.0,
            max_retries=0,
        )
        self.model = config.model
        self.api_retries = api_retries
        self.max_tokens = max_tokens

    def _call(self, messages, deadline=None):
        errors = []
        for attempt in range(self.api_retries + 1):
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("wall-clock budget exhausted before API request")
            try:
                response = self.client.responses.create(
                    model=self.model,
                    max_output_tokens=self.max_tokens,
                    instructions=build_system_prompt(),
                    input=messages,
                    tools=[PROPOSE_TOOL],
                    tool_choice={"type": "function", "name": "propose_iteration"},
                    parallel_tool_calls=False,
                    store=False,
                )
                calls = [
                    item
                    for item in response.output
                    if getattr(item, "type", None) == "function_call"
                    and getattr(item, "name", None) == "propose_iteration"
                ]
                if len(calls) != 1:
                    raise ValueError(
                        "OpenAI response must contain exactly one "
                        "propose_iteration function call"
                    )
                proposal = json.loads(calls[0].arguments)
                if not isinstance(proposal, dict):
                    raise ValueError("propose_iteration arguments must decode to an object")
                usage = {
                    "input_tokens": int(response.usage.input_tokens),
                    "output_tokens": int(response.usage.output_tokens),
                }
                return proposal, usage, errors
            except Exception as exc:
                errors.append({"attempt": attempt, "error": repr(exc), "timestamp": utc_now()})
                if attempt >= self.api_retries:
                    raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
                delay = min(2**attempt, 15)
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.time()))
                if delay <= 0:
                    raise TimeoutError("wall-clock budget exhausted during API retry")
                time.sleep(delay)

    def propose(self, prompt, deadline=None):
        return self._call([{"role": "user", "content": prompt}], deadline)

    def repair(self, code, error, parent_id, deadline=None):
        prompt = f"""The candidate for parent {parent_id} failed. Debug the SAME
hypothesis and return a complete corrected pipeline through the tool. Do not
silently replace it with a different experiment.

ERROR
{error[-5000:]}

BROKEN CODE
```python
{code}
```"""
        return self._call([{"role": "user", "content": prompt}], deadline)


def normalize_proposal(proposal, selected_parent):
    required = ("operation", "direction", "hypothesis", "reasoning", "sources", "code")
    missing = [key for key in required if key not in proposal]
    if missing:
        raise ValueError(f"proposal missing required field(s): {missing}")
    if proposal["operation"] not in {"draft", "improve", "debug", "combine"}:
        raise ValueError(f"invalid operation: {proposal['operation']!r}")
    if not isinstance(proposal["sources"], list):
        raise ValueError("proposal sources must be a list")
    for key in ("direction", "hypothesis", "reasoning", "code"):
        if not isinstance(proposal[key], str) or not proposal[key].strip():
            raise ValueError(f"proposal field {key!r} must be a non-empty string")
    normalized = dict(proposal)
    normalized["requested_parent_id"] = proposal.get("parent_id")
    normalized["parent_id"] = selected_parent
    return normalized


def execute_with_repairs(
    proposal,
    iteration_dir,
    proposer,
    validation_rows,
    data_dir,
    seed,
    timeout_seconds,
    memory_limit_mb,
    max_fix_attempts,
    known_hashes,
    deadline,
    runtime_dir=None,
):
    code = proposal["code"]
    attempts = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    api_events = []
    for attempt_number in range(max_fix_attempts + 1):
        code_hash = sha256_text(code)
        if attempt_number == 0 and code_hash in known_hashes:
            ok, reason = False, "duplicate candidate code already evaluated"
        else:
            ok, reason = check_code(code)
        if not ok:
            result = {"ok": False, "error": f"safety/contract check failed: {reason}"}
        else:
            attempt_dir = Path(iteration_dir) / f"attempt_{attempt_number:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            code_path = attempt_dir / "pipeline.py"
            code_path.write_text(code, encoding="utf-8")
            try:
                os.chmod(code_path, 0o444)
            except OSError:
                pass
            remaining = max(1.0, deadline - time.time())
            result = run_candidate(
                code_path,
                attempt_dir,
                data_dir,
                seed,
                "valid",
                len(validation_rows),
                min(timeout_seconds, remaining),
                memory_limit_mb,
                runtime_dir,
            )
        attempts.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"scores", "diagnostics", "artifact_dir"}
            }
            | {"attempt": attempt_number, "code_sha256": code_hash}
        )
        if result["ok"]:
            metrics = score_validation(validation_rows, result["scores"])
            promote_attempt(iteration_dir, code, result, attempt_number)
            return {
                "ok": True,
                "code": code,
                "code_sha256": code_hash,
                "metrics": metrics,
                "diagnostics": result["diagnostics"],
                "attempts": attempts,
                "tokens": token_usage,
                "api_events": api_events,
                "error": None,
            }
        if proposer is None or attempt_number >= max_fix_attempts:
            break
        repaired, usage, events = proposer.repair(
            code,
            result["error"] + "\n" + result.get("stderr_tail", ""),
            proposal["parent_id"],
            deadline,
        )
        token_usage["input_tokens"] += usage["input_tokens"]
        token_usage["output_tokens"] += usage["output_tokens"]
        api_events.extend(events)
        code = normalize_proposal(repaired, proposal["parent_id"])["code"]
    return {
        "ok": False,
        "code": code,
        "code_sha256": sha256_text(code),
        "metrics": None,
        "diagnostics": {},
        "attempts": attempts,
        "tokens": token_usage,
        "api_events": api_events,
        "error": attempts[-1].get("error", "candidate failed") if attempts else "candidate failed",
    }


def history_by_iteration(history):
    return {entry["iteration"]: entry for entry in history}


def select_parent(history, best_iteration, next_iteration, epsilon):
    if next_iteration % 5:
        return best_iteration
    successful = [
        entry
        for entry in history
        if entry.get("status") == "ok"
        and entry["iteration"] != best_iteration
        and entry.get("metrics", {}).get("valid", {}).get("primary", -1)
        >= history_by_iteration(history)[best_iteration]["metrics"]["valid"]["primary"] - 2 * epsilon
    ]
    if not successful:
        return best_iteration
    child_counts = {
        entry["iteration"]: sum(1 for node in history if node.get("parent_id") == entry["iteration"])
        for entry in successful
    }
    successful.sort(
        key=lambda entry: (
            child_counts[entry["iteration"]],
            -entry["metrics"]["valid"]["primary"],
        )
    )
    return successful[0]["iteration"]


def recommend_direction(history):
    successful_wins = {
        entry.get("direction", "").split("/", 1)[0]
        for entry in history
        if entry.get("is_best") and entry.get("iteration", 0) > 0
    }
    if len(successful_wins) >= 2:
        return "combine/two_confirmed_wins"
    for direction in DIRECTION_PRIORITY:
        root = direction.split("/", 1)[0]
        attempts = [
            entry
            for entry in history
            if entry.get("direction", "").split("/", 1)[0] == root
            and entry.get("iteration", 0) > 0
        ]
        since_win = 0
        for entry in reversed(attempts):
            if entry.get("is_best"):
                break
            since_win += 1
        if len(attempts) < 3 or since_win < 3:
            return direction
    return "loss/listwise_or_combination"


def build_user_prompt(parent_entry, parent_code, history, best_entry, recommended):
    recent = []
    for entry in history[-20:]:
        metric = entry.get("metrics", {}).get("valid", {})
        score = metric.get("primary")
        outcome = f"primary={score:.6f}" if score is not None else f"FAILED: {entry.get('error', '')[:160]}"
        recent.append(
            f"- id={entry['iteration']} parent={entry.get('parent_id')} "
            f"direction={entry.get('direction')} best={entry.get('is_best', False)} "
            f"hypothesis={entry.get('hypothesis')!r} -> {outcome}"
        )
    tried = [
        f"{entry.get('direction')}: {entry.get('hypothesis')}"
        for entry in history
        if entry.get("iteration", 0) > 0
    ]
    return f"""SELECTED PARENT ID: {parent_entry['iteration']}
SELECTED PARENT VALID PRIMARY: {parent_entry['metrics']['valid']['primary']:.6f}
CURRENT BEST ID: {best_entry['iteration']}
CURRENT BEST VALID PRIMARY: {best_entry['metrics']['valid']['primary']:.6f}
HARNESS-RECOMMENDED NEXT DIRECTION: {recommended}

PARENT PIPELINE
```python
{parent_code}
```

RECENT SEARCH TREE RESULTS
{chr(10).join(recent)}

COMPLETE TRIED-IDEA REGISTRY (do not duplicate)
{chr(10).join(tried) if tried else '(none beyond baseline)'}

Use parent_id={parent_entry['iteration']} and propose exactly one focused next
iteration through the tool. The recommendation is a search-policy hint; depart
from it only with a specific evidence-based reason."""


def has_converged(best_curve, epsilon, rounds):
    return len(best_curve) >= rounds + 1 and best_curve[-1] - best_curve[-1 - rounds] <= epsilon


def code_diff(parent_code, candidate_code, parent_id, iteration):
    return "\n".join(
        difflib.unified_diff(
            parent_code.splitlines(),
            candidate_code.splitlines(),
            fromfile=f"iter_{parent_id:04d}/pipeline.py",
            tofile=f"iter_{iteration:04d}/pipeline.py",
            lineterm="",
        )
    )


def create_run_dir(requested):
    if requested:
        run_dir = Path(requested).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"new --run_dir must be empty: {run_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = RUNS_DIR / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iterations").mkdir()
    return run_dir


STATIC_DATA_FILES = (
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
    "user_features_pure.csv",
)
SOURCE_DATA_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    *STATIC_DATA_FILES,
)
OUTCOME_COLUMNS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
)


def _copy_read_only(source, destination):
    shutil.copy2(source, destination)
    try:
        os.chmod(destination, 0o444)
    except OSError:
        pass


def hash_named_files(directory, names):
    directory = Path(directory)
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"integrity check missing files in {directory}: {missing}")
    return {name: sha256_file(directory / name) for name in names}


def verify_named_hashes(directory, expected, label):
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"{label} has no pinned integrity hashes")
    actual = hash_named_files(directory, tuple(expected))
    if actual != expected:
        changed = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise RuntimeError(f"{label} integrity mismatch: {changed}")
    return actual


def _read_manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read integrity manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"integrity manifest must be a JSON object: {path}")
    return manifest


def manifest_integrity_record(directory, manifest_name, hashes_key="files_sha256"):
    directory = Path(directory)
    manifest_path = directory / manifest_name
    manifest = _read_manifest(manifest_path)
    expected = manifest.get(hashes_key)
    verify_named_hashes(directory, expected, str(directory))
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "files_sha256": expected,
    }


def verify_integrity_record(directory, manifest_name, expected_record, label):
    if not isinstance(expected_record, dict):
        raise RuntimeError(f"{label} has no pinned integrity record")
    actual = manifest_integrity_record(directory, manifest_name)
    if actual != expected_record:
        raise RuntimeError(f"{label} manifest or file integrity changed")
    return actual


def read_verified_code(path, expected_sha256):
    path = Path(path)
    if not path.is_file() or not expected_sha256:
        raise RuntimeError(f"pipeline integrity metadata is missing for {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"pipeline integrity mismatch for {path}")
    code = path.read_text(encoding="utf-8")
    ok, reason = check_code(code)
    if not ok:
        raise RuntimeError(f"stored pipeline no longer passes safety validation: {reason}")
    return code


def _write_late_log_view(source, destination, mode):
    rows_written = 0
    with open(source, newline="", encoding="utf-8") as input_handle, open(
        destination, "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {source}")
        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            date = int(row["date"])
            if mode == "development" and date > 20220428:
                continue
            if mode == "final" and date >= 20220429:
                for column in OUTCOME_COLUMNS:
                    if column in row:
                        row[column] = "0"
            writer.writerow(row)
            rows_written += 1
    try:
        os.chmod(destination, 0o444)
    except OSError:
        pass
    return rows_written


def prepare_data_view(source_dir, view_dir, mode):
    """Create a read-only candidate view with held-out outcomes unavailable."""
    if mode not in {"development", "final"}:
        raise ValueError(f"unknown data-view mode: {mode}")
    source_dir = Path(source_dir).resolve()
    view_dir = Path(view_dir).resolve()
    manifest_path = view_dir / "view_manifest.json"
    missing = [name for name in SOURCE_DATA_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"data view cannot be built; missing files: {missing}")
    source_hashes = hash_named_files(source_dir, SOURCE_DATA_FILES)
    if manifest_path.is_file():
        manifest = _read_manifest(manifest_path)
        if manifest.get("schema_version") != 2 or manifest.get("mode") != mode:
            raise RuntimeError(f"data-view manifest mismatch: {view_dir}")
        if "source_dir" in manifest:
            raise RuntimeError(f"data-view manifest exposes a forbidden source path: {view_dir}")
        if manifest.get("source_files_sha256") != source_hashes:
            raise RuntimeError(f"data-view source integrity mismatch: {view_dir}")
        verify_named_hashes(view_dir, manifest.get("files_sha256"), "data view")
        return view_dir

    if view_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.replace(view_dir, view_dir.with_name(view_dir.name + f".orphaned-{suffix}"))
    view_dir.mkdir(parents=True)
    _copy_read_only(
        source_dir / "log_standard_4_08_to_4_21_pure.csv",
        view_dir / "log_standard_4_08_to_4_21_pure.csv",
    )
    late_rows = _write_late_log_view(
        source_dir / "log_standard_4_22_to_5_08_pure.csv",
        view_dir / "log_standard_4_22_to_5_08_pure.csv",
        mode,
    )
    for name in STATIC_DATA_FILES:
        _copy_read_only(source_dir / name, view_dir / name)
    view_hashes = hash_named_files(view_dir, SOURCE_DATA_FILES)
    manifest = {
        "schema_version": 2,
        "mode": mode,
        "source_files_sha256": source_hashes,
        "files_sha256": view_hashes,
        "late_log_rows": late_rows,
        "test_rows_present": mode == "final",
        "test_outcomes_masked": mode == "final",
        "random_exposure_log_exposed": False,
        "created_at": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    try:
        os.chmod(manifest_path, 0o444)
    except OSError:
        pass
    return view_dir


def prepare_runtime_view(run_dir):
    """Copy only the trusted local modules candidates are allowed to import."""
    runtime_dir = Path(run_dir) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "runtime_manifest.json"
    source_hashes = hash_named_files(PROJECT_ROOT, RUNTIME_FILES)
    if manifest_path.is_file():
        manifest = _read_manifest(manifest_path)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("files_sha256") != source_hashes
        ):
            raise RuntimeError("trusted runtime source or manifest changed")
        verify_named_hashes(runtime_dir, source_hashes, "candidate runtime")
        return runtime_dir
    unexpected = [path.name for path in runtime_dir.iterdir()]
    if unexpected:
        raise RuntimeError(f"runtime exists without an integrity manifest: {unexpected}")
    for name in RUNTIME_FILES:
        _copy_read_only(PROJECT_ROOT / name, runtime_dir / name)
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "files_sha256": source_hashes,
            "created_at": utc_now(),
        },
    )
    try:
        os.chmod(manifest_path, 0o444)
    except OSError:
        pass
    return runtime_dir


def prepare_iteration_dir(run_dir, iteration):
    """Create an iteration directory without deleting artifacts from a crash."""
    iteration_dir = Path(run_dir) / "iterations" / f"iter_{iteration:04d}"
    if iteration_dir.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        orphan = iteration_dir.with_name(iteration_dir.name + f".orphaned-{suffix}")
        counter = 1
        while orphan.exists():
            orphan = iteration_dir.with_name(
                iteration_dir.name + f".orphaned-{suffix}-{counter}"
            )
            counter += 1
        os.replace(iteration_dir, orphan)
    iteration_dir.mkdir(parents=True)
    return iteration_dir


def reconcile_state_from_log(state, history):
    """Recover when a crash durably logged a node before checkpointing state."""
    if len(history) < state["next_iteration"]:
        raise RuntimeError("state is ahead of the durable run log")
    if not history:
        if state["next_iteration"] != 0:
            raise RuntimeError("nonzero state has an empty durable run log")
        return False
    checkpoint_history = history[: state["next_iteration"]]
    checkpoint_best = next(
        (entry for entry in reversed(checkpoint_history) if entry.get("is_best")),
        None,
    )
    if checkpoint_best is not None:
        pinned_hash = state.get("best_code_sha256")
        if (
            state.get("best_iteration") != checkpoint_best.get("iteration")
            or state.get("best_score")
            != checkpoint_best.get("metrics", {}).get("valid", {}).get("primary")
            or (pinned_hash is not None and pinned_hash != checkpoint_best.get("code_sha256"))
        ):
            raise RuntimeError("checkpointed best node disagrees with the durable run log")
    elif state["next_iteration"]:
        raise RuntimeError("checkpointed run-log prefix has no accepted baseline node")

    best_iteration = None
    best_score = None
    best_code_sha256 = None
    best_curve = []
    for entry in history:
        if entry.get("is_best"):
            best_iteration = entry["iteration"]
            best_score = entry["metrics"]["valid"]["primary"]
            best_code_sha256 = entry.get("code_sha256")
        if best_score is None:
            raise RuntimeError("run log has no accepted baseline node")
        if entry.get("status") == "ok" and entry.get("metrics", {}).get("valid"):
            best_curve.append(best_score)
    recovered = {
        "next_iteration": len(history),
        "best_iteration": best_iteration,
        "best_score": best_score,
        "best_code_sha256": best_code_sha256,
        "best_curve": best_curve,
        "total_tokens": {
        "input_tokens": sum(entry.get("tokens_input", 0) for entry in history),
        "output_tokens": sum(entry.get("tokens_output", 0) for entry in history),
        },
    }
    changed = any(state.get(key) != value for key, value in recovered.items())
    state.update(recovered)
    return changed


def state_checkpoint(state, state_path, base_elapsed, session_started):
    state["elapsed_seconds"] = base_elapsed + (time.time() - session_started)
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)


def validate_baseline(metrics, tolerance):
    with open(BASELINE_SCORES, encoding="utf-8") as handle:
        expected = json.load(handle)["scores"]["fm_official"]["valid"]
    deltas = {}
    failures = []
    for metric in ("GAUC", "nDCG@5", "primary"):
        if metric not in metrics:
            failures.append(f"{metric}=missing")
            continue
        delta = float(metrics[metric]) - float(expected[metric])
        deltas[metric] = delta
        if abs(delta) > tolerance:
            failures.append(
                f"{metric} actual={float(metrics[metric]):.6f} "
                f"published={float(expected[metric]):.6f} delta={delta:+.6f}"
            )
    if failures:
        raise RuntimeError(
            "baseline reproduction failed (each metric is gated at "
            f"tolerance {tolerance:.6f}): " + "; ".join(failures)
        )
    return {
        "expected": expected,
        "metric_deltas": deltas,
        "primary_delta": deltas["primary"],
        "tolerance": tolerance,
    }


def confirmation_run(
    iteration_dir,
    candidate_code_path,
    parent_code_path,
    validation_rows,
    data_dir,
    seed,
    timeout_seconds,
    memory_limit_mb,
    deadline,
    runtime_dir=None,
):
    confirmation_seed = seed + 1
    results = {}
    for label, path in (("candidate", candidate_code_path), ("parent", parent_code_path)):
        remaining = deadline - time.time()
        if remaining <= 1:
            return {
                "promotion_check_passed": False,
                "confirmed": False,
                "confirmed_semantics": "harness promotion policy, not statistical significance",
                "error": "wall-clock budget exhausted during confirmation",
            }
        result = run_candidate(
            path,
            Path(iteration_dir) / "confirmation" / label,
            data_dir,
            confirmation_seed,
            "valid",
            len(validation_rows),
            min(timeout_seconds, remaining),
            memory_limit_mb,
            runtime_dir,
        )
        if not result["ok"]:
            return {
                "promotion_check_passed": False,
                "confirmed": False,
                "confirmed_semantics": "harness promotion policy, not statistical significance",
                "error": f"{label} replicate failed: {result['error']}",
            }
        results[label] = score_validation(validation_rows, result["scores"])
    paired_gain = results["candidate"]["primary"] - results["parent"]["primary"]
    policy_passed = paired_gain > 0
    return {
        "promotion_check_passed": policy_passed,
        # Compatibility field only. This is a harness-policy boolean, not a
        # statistical-significance claim from two seeds.
        "confirmed": policy_passed,
        "confirmed_semantics": "harness promotion policy, not statistical significance",
        "seed": confirmation_seed,
        "candidate_metrics": results["candidate"],
        "parent_metrics": results["parent"],
        "paired_primary_gain": paired_gain,
    }


def write_submission(path, rows, scores):
    if len(rows) != len(scores):
        raise ValueError(f"submission length mismatch: {len(scores)} vs {len(rows)}")
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id in range(len(rows)):
            row = rows[row_id]
            writer.writerow([row_id, row[1], row[2], f"{float(scores[row_id]):.17g}"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finalize_once(
    state,
    run_dir,
    data_dir,
    timeout_seconds,
    memory_limit_mb,
    deadline,
    runtime_dir=None,
):
    started_record = Path(run_dir) / "finalization_started.json"
    finalization_record = Path(run_dir) / "finalization.json"
    if state.get("finalized") or started_record.exists() or finalization_record.exists():
        raise RuntimeError("this run was already finalized; held-out prediction is one-time")
    best_code = run_dir / "iterations" / f"iter_{state['best_iteration']:04d}" / "pipeline.py"
    read_verified_code(best_code, state.get("best_code_sha256"))
    if runtime_dir is not None:
        verify_integrity_record(
            runtime_dir,
            "runtime_manifest.json",
            state.get("runtime_integrity"),
            "candidate runtime",
        )
    remaining = deadline - time.time()
    if remaining <= 1:
        raise TimeoutError("no wall-clock budget remains for finalization")

    # O_EXCL is the concurrency boundary. The durable marker is created before
    # reading or materializing any held-out rows. A crash after this point is a
    # consumed finalization attempt and requires a new run, never an automatic
    # retry that could expose the held-out split twice.
    started_at = utc_now()
    try:
        atomic_create_json(
            started_record,
            {
                "schema_version": 1,
                "started_at": started_at,
                "best_iteration": state["best_iteration"],
                "best_code_sha256": state["best_code_sha256"],
                "test_metrics_evaluated": False,
            },
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "this run already began finalization; held-out prediction will not be retried"
        ) from exc

    verify_named_hashes(
        data_dir,
        state.get("source_data_sha256"),
        "source data before finalization",
    )
    final_data_dir = prepare_data_view(
        data_dir, Path(run_dir) / "data_views" / "final", "final"
    )
    from data import load

    # Load IDs and row order only from the masked final view, never from the
    # source log that contains held-out outcomes.
    test_rows = load(final_data_dir, split_names=("test",))["test"]
    remaining = deadline - time.time()
    if remaining <= 1:
        raise TimeoutError("wall-clock budget expired while preparing finalization")
    result = run_candidate(
        best_code,
        run_dir / "finalization",
        final_data_dir,
        state["seed"],
        "test",
        len(test_rows),
        min(timeout_seconds, remaining),
        memory_limit_mb,
        runtime_dir,
    )
    if not result["ok"]:
        raise RuntimeError(f"final held-out prediction failed: {result['error']}")
    submission_path = run_dir / "submission.csv"
    write_submission(submission_path, test_rows, result["scores"])
    state["finalized"] = True
    state["finalized_at"] = utc_now()
    state["submission_path"] = str(submission_path)
    state["submission_sha256"] = sha256_file(submission_path)
    atomic_json(
        finalization_record,
        {
            "schema_version": 1,
            "started_at": started_at,
            "finalized_at": state["finalized_at"],
            "best_iteration": state["best_iteration"],
            "best_code_sha256": state["best_code_sha256"],
            "submission_path": state["submission_path"],
            "submission_sha256": state["submission_sha256"],
            "test_metrics_evaluated": False,
        },
    )
    return submission_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "KuaiRand-Pure" / "data"))
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--run_dir", help="empty directory for a new run")
    location.add_argument("--resume", help="existing run directory to resume")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--max_iterations", type=int, default=50, help="total nodes including baseline")
    parser.add_argument("--wall_clock_hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_bpr_bootstrap", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="generate held-out submission once after search")
    parser.add_argument("--pipeline_timeout_minutes", type=float, default=20.0)
    parser.add_argument("--memory_limit_mb", type=int, default=4096)
    parser.add_argument("--api_retries", type=int, default=3)
    parser.add_argument("--max_fix_attempts", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--convergence_rounds", type=int, default=3)
    parser.add_argument("--baseline_tolerance", type=float, default=0.002)
    parser.add_argument("--manual_interventions", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max_iterations must be at least 1")
    if args.wall_clock_hours <= 0 or args.pipeline_timeout_minutes <= 0:
        parser.error("time limits must be positive")
    if args.memory_limit_mb <= 0:
        parser.error("--memory_limit_mb must be positive")
    if args.api_retries < 0 or args.max_fix_attempts < 0:
        parser.error("retry counts must be nonnegative")
    if args.epsilon < 0 or args.baseline_tolerance < 0:
        parser.error("score tolerances must be nonnegative")
    if args.convergence_rounds < 1:
        parser.error("--convergence_rounds must be at least 1")
    if args.manual_interventions < 0:
        parser.error("--manual_interventions must be nonnegative")
    return args


def _run_locked(args, data_dir, run_dir, openai_config=None):
    if args.resume:
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            raise SystemExit(f"resume state not found: {state_path}")
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("data_dir") and Path(state["data_dir"]).resolve() != data_dir:
            raise SystemExit("resume data_dir differs from the original run")
        # Legacy states stored an absolute source path. Hash pinning supersedes
        # it and avoids leaving that path in candidate-adjacent run metadata.
        state.pop("data_dir", None)
        history = load_jsonl(run_dir / "run_log.jsonl")
        try:
            reconciled = reconcile_state_from_log(state, history)
        except RuntimeError as exc:
            raise SystemExit(f"cannot resume safely: {exc}") from exc
        if reconciled:
            atomic_json(state_path, state)
        started_record = run_dir / "finalization_started.json"
        finalization_record = run_dir / "finalization.json"
        if finalization_record.is_file():
            if not started_record.is_file():
                raise SystemExit("cannot resume safely: finalization success has no start marker")
            finalized = _read_manifest(finalization_record)
            started = _read_manifest(started_record)
            submission_path = run_dir / "submission.csv"
            if (
                not submission_path.is_file()
                or sha256_file(submission_path) != finalized.get("submission_sha256")
                or finalized.get("best_iteration") != state.get("best_iteration")
                or finalized.get("best_code_sha256") != state.get("best_code_sha256")
                or started.get("best_iteration") != finalized.get("best_iteration")
                or started.get("best_code_sha256") != finalized.get("best_code_sha256")
            ):
                raise SystemExit("cannot resume safely: finalization integrity mismatch")
            state["finalized"] = True
            state["finalized_at"] = finalized["finalized_at"]
            state["submission_path"] = str(submission_path)
            state["submission_sha256"] = finalized["submission_sha256"]
            atomic_json(state_path, state)
            raise SystemExit("this run is finalized and immutable; start a new run to continue")
        elif started_record.is_file() and not finalization_record.is_file():
            raise SystemExit(
                "cannot resume: finalization previously started but did not complete; "
                "held-out prediction will not be retried"
            )
        elif state.get("finalized") and not finalization_record.is_file():
            raise SystemExit("cannot resume safely: finalized state has no durable record")
        state["manual_interventions"] += args.manual_interventions
    else:
        state_path = run_dir / "state.json"
        state = {
            "schema_version": 3,
            "run_id": run_dir.name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "seed": args.seed,
            "model": args.model,
            "provider": "builtin",
            "max_iterations": args.max_iterations,
            "wall_clock_hours": args.wall_clock_hours,
            "epsilon": args.epsilon,
            "convergence_rounds": args.convergence_rounds,
            "next_iteration": 0,
            "best_iteration": None,
            "best_score": None,
            "best_code_sha256": None,
            "best_curve": [],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
            "manual_interventions": args.manual_interventions,
            "elapsed_seconds": 0.0,
            "finalized": False,
            "bpr_bootstrap": not args.skip_bpr_bootstrap,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
        history = []
        atomic_json(state_path, state)

    base_elapsed = float(state.get("elapsed_seconds", 0.0))
    session_started = time.time()
    deadline = session_started + max(
        0.0, float(state["wall_clock_hours"]) * 3600.0 - base_elapsed
    )
    timeout_seconds = args.pipeline_timeout_minutes * 60.0
    runtime_dir = prepare_runtime_view(run_dir)
    development_data_dir = prepare_data_view(
        data_dir, run_dir / "data_views" / "development", "development"
    )
    development_manifest = _read_manifest(development_data_dir / "view_manifest.json")
    current_integrity = {
        "source_data_sha256": development_manifest.get("source_files_sha256"),
        "development_view_integrity": manifest_integrity_record(
            development_data_dir, "view_manifest.json"
        ),
        "runtime_integrity": manifest_integrity_record(
            runtime_dir, "runtime_manifest.json"
        ),
    }
    for key, actual in current_integrity.items():
        expected = state.get(key)
        if expected is None:
            if history or state.get("next_iteration", 0) != 0:
                raise SystemExit(
                    f"cannot resume safely: existing run has no pinned {key} metadata"
                )
            state[key] = actual
        elif expected != actual:
            raise SystemExit(f"cannot resume safely: {key} changed")
    state["schema_version"] = 3
    atomic_json(state_path, state)

    from data import load

    validation_rows = load(development_data_dir, split_names=("valid",))["valid"]
    known_hashes = {entry.get("code_sha256") for entry in history if entry.get("code_sha256")}

    def record(entry):
        history.append(entry)
        append_jsonl(run_dir / "run_log.jsonl", entry)
        state["next_iteration"] = entry["iteration"] + 1
        state_checkpoint(state, state_path, base_elapsed, session_started)

    if state["next_iteration"] == 0:
        code = BASELINE_PIPELINE.read_text(encoding="utf-8")
        proposal = {
            "parent_id": None,
            "operation": "draft",
            "direction": "baseline/fm",
            "hypothesis": "Reproduce the official pointwise FM before research iterations.",
            "reasoning": "A trusted reference node is required before branching.",
            "sources": ["Rendle, Factorization Machines (2010)"],
            "code": code,
        }
        iteration_dir = prepare_iteration_dir(run_dir, 0)
        result = execute_with_repairs(
            proposal,
            iteration_dir,
            None,
            validation_rows,
            development_data_dir,
            state["seed"],
            timeout_seconds,
            args.memory_limit_mb,
            0,
            known_hashes,
            deadline,
            runtime_dir,
        )
        if not result["ok"]:
            state["status"] = "failed"
            state_checkpoint(state, state_path, base_elapsed, session_started)
            raise SystemExit(f"baseline reproduction failed to run: {result['error']}")
        reproduction = validate_baseline(result["metrics"], args.baseline_tolerance)
        state["best_iteration"] = 0
        state["best_score"] = result["metrics"]["primary"]
        state["best_code_sha256"] = result["code_sha256"]
        state["best_curve"] = [state["best_score"]]
        entry = {
            "iteration": 0,
            "parent_id": None,
            "timestamp": utc_now(),
            "operation": proposal["operation"],
            "direction": proposal["direction"],
            "hypothesis": proposal["hypothesis"],
            "reasoning": proposal["reasoning"],
            "sources": proposal["sources"],
            "status": "ok",
            "metrics": {"valid": result["metrics"]},
            "diagnostics": result["diagnostics"],
            "baseline_reproduction": reproduction,
            "attempts": result["attempts"],
            "code_sha256": result["code_sha256"],
            "diff": None,
            "is_best": True,
            "selection": {"accepted": True, "reason": "baseline"},
            "tokens_input": 0,
            "tokens_output": 0,
            "error": None,
        }
        known_hashes.add(result["code_sha256"])
        record(entry)
        print(f"[iter 0] baseline valid primary {state['best_score']:.6f}")

    proposer = None
    stop_reason = None
    while state["next_iteration"] < state["max_iterations"]:
        if time.time() >= deadline:
            stop_reason = "wall_clock_limit"
            break
        # Always preserve enough of the six-hour run budget for the exact
        # validation-selected winner to produce its one-time final scores.  A
        # user may deliberately invoke --finalize in a later resume session.
        if deadline - time.time() <= timeout_seconds + 1:
            stop_reason = "wall_clock_reserve_for_finalization"
            break
        if has_converged(state["best_curve"], state["epsilon"], state["convergence_rounds"]):
            stop_reason = "converged"
            break

        iteration = state["next_iteration"]
        best_iteration = state["best_iteration"]
        parent_id = select_parent(history, best_iteration, iteration, state["epsilon"])
        nodes = history_by_iteration(history)
        parent_entry = nodes[parent_id]
        parent_path = run_dir / "iterations" / f"iter_{parent_id:04d}" / "pipeline.py"
        parent_code = read_verified_code(parent_path, parent_entry.get("code_sha256"))

        proposal_tokens = {"input_tokens": 0, "output_tokens": 0}
        api_events = []
        if iteration == 1 and state.get("bpr_bootstrap", True):
            proposal = {
                "parent_id": parent_id,
                "operation": "improve",
                "direction": "loss/bpr",
                "hypothesis": (
                    "Replace pointwise logloss with same-user BPR pairs so training "
                    "optimizes observed positive-over-negative ordering."
                ),
                "reasoning": (
                    "GAUC and nDCG@5 depend on within-user order; the textbook BPR loss, "
                    "with benchmark-adapted observed-negative sampling, directly pushes "
                    "logged long-view impressions above logged zero-label impressions."
                ),
                "sources": [
                    "Knowledge base section 1; Rendle et al., Bayesian Personalized Ranking (UAI 2009)"
                ],
                "code": BPR_PIPELINE.read_text(encoding="utf-8"),
            }
        else:
            if proposer is None:
                try:
                    proposer = OpenAIProposer(
                        args.model,
                        api_retries=args.api_retries,
                        config=openai_config,
                    )
                except RuntimeError as exc:
                    state["status"] = "needs_configuration"
                    state["stop_reason"] = "llm_configuration_missing"
                    state_checkpoint(state, state_path, base_elapsed, session_started)
                    raise SystemExit(str(exc))
                state["provider"] = "builtin+openai"
                state["model"] = args.model
            recommendation = recommend_direction(history)
            prompt = build_user_prompt(
                parent_entry,
                parent_code,
                history,
                nodes[best_iteration],
                recommendation,
            )
            try:
                raw, proposal_tokens, api_events = proposer.propose(prompt, deadline)
                proposal = normalize_proposal(raw, parent_id)
            except Exception as exc:
                entry = {
                    "iteration": iteration,
                    "parent_id": parent_id,
                    "timestamp": utc_now(),
                    "operation": "draft",
                    "direction": recommendation,
                    "hypothesis": "LLM proposal request",
                    "reasoning": "No executable proposal was returned.",
                    "sources": [],
                    "status": "failed",
                    "metrics": None,
                    "diagnostics": {},
                    "attempts": [],
                    "api_events": api_events,
                    "code_sha256": None,
                    "diff": None,
                    "is_best": False,
                    "selection": {"accepted": False, "reason": "proposal_error"},
                    "tokens_input": proposal_tokens["input_tokens"],
                    "tokens_output": proposal_tokens["output_tokens"],
                    "error": repr(exc),
                }
                record(entry)
                print(f"[iter {iteration}] proposal failed: {exc}", file=sys.stderr)
                continue

        iteration_dir = prepare_iteration_dir(run_dir, iteration)
        try:
            result = execute_with_repairs(
                proposal,
                iteration_dir,
                proposer
                if iteration != 1 or not state.get("bpr_bootstrap", True)
                else None,
                validation_rows,
                development_data_dir,
                state["seed"],
                timeout_seconds,
                args.memory_limit_mb,
                args.max_fix_attempts,
                known_hashes,
                deadline,
                runtime_dir,
            )
        except Exception:
            result = {
                "ok": False,
                "code": proposal["code"],
                "code_sha256": sha256_text(proposal["code"]),
                "metrics": None,
                "diagnostics": {},
                "attempts": [],
                "tokens": {"input_tokens": 0, "output_tokens": 0},
                "api_events": [],
                "error": traceback.format_exc(),
            }

        for key in state["total_tokens"]:
            state["total_tokens"][key] += proposal_tokens[key] + result["tokens"][key]
        raw_gain = None
        is_best = False
        selection = {"accepted": False, "reason": "failed"}
        if result["ok"]:
            raw_gain = result["metrics"]["primary"] - state["best_score"]
            if raw_gain >= state["epsilon"]:
                is_best = True
                selection = {"accepted": True, "reason": "gain_exceeds_noise_threshold", "raw_gain": raw_gain}
            elif raw_gain > 0:
                current_best_entry = history_by_iteration(history)[state["best_iteration"]]
                current_best_path = (
                    run_dir
                    / "iterations"
                    / f"iter_{state['best_iteration']:04d}"
                    / "pipeline.py"
                )
                read_verified_code(current_best_path, current_best_entry.get("code_sha256"))
                confirmation = confirmation_run(
                    iteration_dir,
                    iteration_dir / "pipeline.py",
                    current_best_path,
                    validation_rows,
                    development_data_dir,
                    state["seed"],
                    timeout_seconds,
                    args.memory_limit_mb,
                    deadline,
                    runtime_dir,
                )
                replicate_gain = confirmation.get("paired_primary_gain")
                mean_paired_gain = (
                    (raw_gain + replicate_gain) / 2.0
                    if replicate_gain is not None
                    else None
                )
                confirmation["seed_0_paired_primary_gain"] = raw_gain
                confirmation["two_seed_mean_paired_gain"] = mean_paired_gain
                is_best = bool(confirmation.get("promotion_check_passed")) and (
                    mean_paired_gain is not None
                    and mean_paired_gain >= state["epsilon"] / 2.0
                )
                confirmation["promotion_check_passed"] = is_best
                confirmation["confirmed"] = is_best
                selection = {
                    "accepted": is_best,
                    "reason": "small_gain_seed_confirmation",
                    "raw_gain": raw_gain,
                    "confirmation": confirmation,
                }
            else:
                selection = {"accepted": False, "reason": "no_validation_gain", "raw_gain": raw_gain}

        if is_best:
            state["best_iteration"] = iteration
            state["best_score"] = result["metrics"]["primary"]
            state["best_code_sha256"] = result["code_sha256"]
        if result["ok"]:
            state["best_curve"].append(state["best_score"])
        diff = code_diff(parent_code, result["code"], parent_id, iteration)
        entry = {
            "iteration": iteration,
            "parent_id": parent_id,
            "requested_parent_id": proposal.get("requested_parent_id", proposal.get("parent_id")),
            "timestamp": utc_now(),
            "operation": proposal["operation"],
            "direction": proposal["direction"],
            "hypothesis": proposal["hypothesis"],
            "reasoning": proposal["reasoning"],
            "sources": proposal["sources"],
            "status": "ok" if result["ok"] else "failed",
            "metrics": {"valid": result["metrics"]} if result["ok"] else None,
            "diagnostics": result["diagnostics"],
            "attempts": result["attempts"],
            "api_events": api_events + result["api_events"],
            "code_sha256": result["code_sha256"],
            "diff": diff,
            "is_best": is_best,
            "selection": selection,
            "tokens_input": proposal_tokens["input_tokens"] + result["tokens"]["input_tokens"],
            "tokens_output": proposal_tokens["output_tokens"] + result["tokens"]["output_tokens"],
            "error": result["error"],
        }
        known_hashes.add(result["code_sha256"])
        record(entry)
        if result["ok"]:
            print(
                f"[iter {iteration}] {proposal['direction']} valid primary "
                f"{result['metrics']['primary']:.6f}{'  <-- NEW BEST' if is_best else ''}"
            )
        else:
            print(f"[iter {iteration}] FAILED: {result['error']}", file=sys.stderr)

    if stop_reason is None:
        stop_reason = "iteration_cap" if state["next_iteration"] >= state["max_iterations"] else "stopped"

    best_source = run_dir / "iterations" / f"iter_{state['best_iteration']:04d}" / "pipeline.py"
    read_verified_code(best_source, state.get("best_code_sha256"))
    shutil.copy2(best_source, run_dir / "best_pipeline.py")
    if args.finalize:
        finalize_once(
            state,
            run_dir,
            data_dir,
            timeout_seconds,
            args.memory_limit_mb,
            deadline,
            runtime_dir,
        )

    state["status"] = "complete"
    state["stop_reason"] = stop_reason
    state_checkpoint(state, state_path, base_elapsed, session_started)
    failures = sum(entry.get("status") != "ok" for entry in history)
    recoveries = sum(
        any(attempt.get("attempt", 0) > 0 and attempt.get("ok") for attempt in entry.get("attempts", []))
        for entry in history
    )
    summary = {
        "run_id": state["run_id"],
        "status": state["status"],
        "stop_reason": stop_reason,
        "best_iteration": state["best_iteration"],
        "best_valid_metrics": history_by_iteration(history)[state["best_iteration"]]["metrics"]["valid"],
        "official_baseline_valid_primary": 0.6016,
        "valid_primary_delta": state["best_score"] - 0.6016,
        "iterations_evaluated": len(history),
        "failed_iterations": failures,
        "recovery_events": recoveries,
        "manual_interventions": state["manual_interventions"],
        "wall_clock_seconds": state["elapsed_seconds"],
        "total_tokens": state["total_tokens"],
        "gpu_hours": 0.0,
        "provider": state["provider"],
        "model": state["model"],
        "finalized": state["finalized"],
        "submission_path": state.get("submission_path"),
        "test_metrics_evaluated": False,
        "best_pipeline_path": str(run_dir / "best_pipeline.py"),
        "run_log_path": str(run_dir / "run_log.jsonl"),
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "search_tree.json", {"nodes": history})
    print("\n=== RUN COMPLETE ===")
    print(json.dumps(summary, indent=2))


def main(argv=None):
    args = parse_args(argv)
    # Resolve the trusted parent-process configuration once. Only the model ID
    # is retained on ``args`` or written to run metadata; the API key remains in
    # the redacted config object and is never forwarded to candidate processes.
    openai_config = load_openai_config(model=args.model)
    args.model = openai_config.model
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data directory does not exist: {data_dir}")
    run_dir = (
        Path(args.resume).resolve()
        if args.resume
        else create_run_dir(args.run_dir)
    )
    try:
        with RunLock(run_dir):
            return _run_locked(args, data_dir, run_dir, openai_config=openai_config)
    except RuntimeError as exc:
        if "run is already active or cannot be locked" in str(exc):
            raise SystemExit(str(exc)) from exc
        raise


if __name__ == "__main__":
    main()
