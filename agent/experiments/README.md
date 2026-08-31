# Rejected validation experiments

These modules preserve reviewable implementations of candidates that did not
beat the promoted validation pipeline. Production code does not import them.
They are evidence for the decision log and regression tests, not alternate
defaults or hidden-test results.

- `causal_history_pipeline.py`: strict causal pooled-history experiment.
- `deepfm_pipeline.py`: small DeepFM-style capacity experiment.
- `hour_field_pipeline.py`: intermediate hour-field experiment retained for
  its focused tests; the accepted implementation now lives in
  `agent/bpr_pipeline.py`.
- `watchtime_multitask_pipeline.py`: train-only watch-ratio auxiliary loss.

Only promote an experiment after the trusted evaluator shows a reproducible
validation gain. Never use test outcomes to choose among these candidates.
