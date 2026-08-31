# Validation-only improvement campaign

This directory summarizes the feasible-first development campaign that replaced
the original `0.603396` BPR bootstrap with the promoted `0.605490` ensemble.
Every score is from the fixed public validation split. No hidden/test metric was
computed, and this evidence does not claim hidden-test improvement.

## Final reproduced result

| Pipeline | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Reproduced pointwise FM | 0.6671326322 | 0.5358048805 | 0.6014687564 |
| Original BPR, seed 0 | 0.6697110261 | 0.5370812958 | 0.6033961610 |
| Promoted single model, seed 0 | 0.6713011777 | 0.5382499207 | 0.6047755492 |
| Promoted three-seed ensemble | **0.6723332004** | **0.5386476783** | **0.6054904394** |

The final gain is `+0.0040216830` over the reproduced FM and `+0.0020942784`
over the old BPR. Relative to the published validation primary `0.6016`, the
gain is `+0.0038904394`.

The promoted single-model primary was `0.6047755492`, `0.6049002208`, and
`0.6052727326` for seeds 0, 1, and 2. The final prediction is the mean of the
three component score vectors after standardizing scores within each user.

## Promoted components

1. Observed-negative same-user BPR.
2. Learning rate `0.0005`.
3. Thirty duration quantiles fitted on training only.
4. A six-value `hour_4h` categorical field fitted on training only.
5. Smoothed training-only user-by-tab affinity, fixed weight `0.1`.
6. Three deterministic seeds with label-free within-user score averaging.

## Rejected directions

After accepting learning rate `0.0005`, the campaign also evaluated and did not
promote the remaining learning-rate/batch/L2 variants, extra uniform negatives,
pointwise warm-up, pointwise-FM blending, equal-pair user weighting, causal
positive-video history, click multitask, watch-ratio multitask, and a small
DeepFM branch. Hard-negative BPR and full listwise softmax had already failed
in the earlier autonomous run and were not repeated.
The compact decisions and scores are in [decision_log.jsonl](decision_log.jsonl);
the required hypotheses, exact diffs, metrics, attempts, and errors are in the
[full run log](run_log.jsonl). Durable context for future agent runs is in
[agent/experiment_memory.md](../../agent/experiment_memory.md).

The final autonomous breadth run then evaluated one successful candidate from
every remaining priority family. Its validation-only results were:

| Direction | Primary | Decision |
|---|---:|---|
| Causal signed-recency history | 0.6049889157 | Rejected |
| Staged train-only click pretraining | 0.6050792595 | Rejected |
| Staged censored watch-time pretraining | 0.6046068034 | Rejected |
| Train-only recency-weighted BPR | 0.6043204549 | Rejected |
| Field-pair-gated FM | 0.6052951724 | Rejected |

The run evaluated seven nodes, stopped as `campaign_converged`, used zero
manual interventions and zero candidate-code repair events, consumed 51,282
input and 31,804 output API tokens, and took 1,588.1 seconds wall-clock. Six
transient provider timeouts were retried automatically. It retained the
iteration-1 ensemble. The official convergence condition first became true at
iteration 4; campaign mode continued through iteration 6 to complete priority
coverage. The exact selected source is archived as
[best_pipeline.py](best_pipeline.py).

## Final output

The selected source was finalized once after model selection. It produced
170,588 aligned test scores, and the independent submission schema/alignment
check passed. No hidden/test metric was computed. Submission SHA-256:

```text
03b8bc99cbc6f391ad3fd8a80bc2da2ce3cab8d4bf011fcc8c844a39b3c35de2
```

Final-output generation took 92.9 additional seconds, bringing total agent
wall-clock through finalization to 1,681.3 seconds. The CSV remains a local,
Git-ignored upload artifact; sanitized metadata is in
[finalization.json](finalization.json).

Rejected implementations under [agent/experiments](../../agent/experiments/)
are isolated evidence and
are never imported by the promoted pipeline. Keeping them does not change model
selection; it makes the negative results reviewable and prevents false claims
that a method was never attempted.

## Reproduce

From the repository root with the dataset installed:

```bash
python -m unittest discover -s tests -v
python agent/preflight.py
python agent/orchestrator.py \
  --max_iterations 2 \
  --run_dir /tmp/woof-ensemble-reproduction
python -m json.tool /tmp/woof-ensemble-reproduction/summary.json
```

Expected validation metrics are approximately `0.672333 / 0.538648 /
0.605490`. The two-node run uses only built-in code and therefore makes no API
call. It stops because of the requested iteration cap; it is not an official
converged full search.
