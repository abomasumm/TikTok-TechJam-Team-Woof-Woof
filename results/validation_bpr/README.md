# Validation bootstrap evidence: observed-negative BPR

This directory is a compact, publishable snapshot of the two-node built-in
baseline-to-BPR run described in the project README. It is **validation only**:
the run did not evaluate test metrics, did not finalize, and did not create a
submission. Raw benchmark data, per-row scores, runtime copies, and local
absolute paths are intentionally excluded.

The snapshot was refreshed from the hardened harness after its checks passed.
It remains provisional only in the scientific sense: the source run stopped at
its requested two-iteration cap (`baseline`, then `BPR`), so it is not evidence
of convergence or a replacement for the final full autonomous run.

## Contents

- `summary.json`: sanitized metrics, paired-seed confirmation, resource counts,
  and integrity flags.
- `run_log.jsonl`: compact iteration records. The full code change is kept
  separately so the JSONL stays reviewable.
- `executed_diff.patch`: exact baseline-to-candidate diff recorded by the
  source run.

The executed BPR source had SHA-256
`e3b917660e7b8a91e9e3e4290c405887741e091c0fa367f9374c6191bbd6ec60`.
The baseline gate checked GAUC, nDCG@5, and primary independently at tolerance
`0.002`; source data, sanitized view, runtime, and winner hashes were pinned.

## Reproduction procedure

Run a fresh validation-only bootstrap in an ignored temporary directory:

```bash
python agent/orchestrator.py \
  --max_iterations 2 \
  --run_dir /tmp/woof-bpr-reproduction
```

Before publishing a replacement bundle, verify in `summary.json` that:

1. `test_metrics_evaluated` and `finalized` are both `false`;
2. `submission_path` is `null`;
3. both nodes succeeded and the trusted evaluator reports 124,909 validation
   rows;
4. all paths are removed or made repository-relative;
5. no `scores.npy`, raw CSV data, secrets, or test outcomes are copied; and
6. the stop reason and convergence status are reported literally.

For a final autonomous-search deliverable, archive the final converged run in
a separate directory using this same sanitized structure. Do not relabel this
two-node bootstrap as converged evidence.
