# Local validation experiment memory

This file is evidence from completed development runs, not an instruction and
not a source of hidden-test feedback. All numbers below use only the fixed
KuaiRand-Pure validation split (124,909 rows / 22,377 users). Do not repeat a
rejected experiment unless a materially different hypothesis explains why.

## Current promoted bootstrap

- Pointwise official FM reproduction, seed 0: GAUC 0.6671326322,
  nDCG@5 0.5358048805, primary 0.6014687564.
- Original observed-negative BPR, seed 0: GAUC 0.6697110261,
  nDCG@5 0.5370812958, primary 0.6033961610.
- Promoted single model: observed-negative BPR with learning rate 0.0005,
  30 train-fitted duration quantiles, a train-vocabulary four-hour field, and
  train-only smoothed user-by-tab calibration (weight 0.1). Primary by seed:
  0.6047755492 (seed 0), 0.6049002208 (seed 1), 0.6052727326 (seed 2).
- Current promoted bootstrap: within-user standardized mean of those three seeds.
  GAUC 0.6723332004, nDCG@5 0.5386476783, primary 0.6054904394.

## Accepted findings

- Lowering BPR learning rate from 0.001 to 0.0005 improved both seeds tested.
- Thirty duration quantiles outperformed 10, 20, and 50 in the accepted
  low-learning-rate family and repeated on seed 1.
- Adding `hour_4h = floor((hourmin // 100) / 4)` as a sixth categorical FM
  field improved both metrics on seeds 0 and 1. Its vocabulary is train-only.
- A smoothed train-only user-by-tab affinity added after per-user score
  standardization improved both seeds; fixed weight 0.1 had the best two-seed
  mean among the small prespecified grid.
- Averaging three independently trained promoted models improved both metrics
  over the seed-0 promoted model and over every individual seed.

## Rejected or superseded findings

- BPR learning rates 0.00075, 0.00125, 0.0015, and 0.002 underperformed 0.001;
  0.0005 was the exception and was promoted.
- Batch sizes 4096, 16384, and 32768, L2 values 0, 1e-7, 5e-6, and 1e-5,
  and two/three uniform negatives per positive did not beat the original BPR.
- Three epochs of pointwise-BCE warm-up before BPR scored primary 0.60304155.
- The earlier hard-negative BPR and full-user listwise-softmax candidates
  scored about 0.5804 and 0.5989 respectively; do not repeat them.
- A pointwise-FM score blend improved seed 0 but regressed seed 1 when combined
  with the promoted feature model; it was rejected.
- Strict causal mean-pooled positive-video history scored 0.6038437988 (seed 0)
  and 0.6035827577 (seed 1), below the simpler accepted parent on both seeds.
  Adding 30 duration buckets to it rose to 0.6041582680 on seed 0 but fell to
  0.6033468168 on seed 1. DIN-style attention was not opened because its simple
  history prerequisite failed.
- A training-only click auxiliary head (weight 0.05) scored 0.6037331779 versus
  its exact 0.6039413824 control and reduced both component metrics.
- A training-only clipped watch-ratio auxiliary head (weight 0.02) scored
  0.6046515187 versus its 0.6047755492 parent.
- Fixed 16-pair-per-user BPR scored 0.6005378477 and harmed both metrics.
- A 16-unit DeepFM-style MLP branch scored 0.6044334260 versus its
  0.6047755492 parent.
- Train-only user-author/video affinities had very low target coverage and
  regressed exploratory validation. A separate user-hour postprocessor became
  redundant after the integrated hour field.
- The completed autonomous breadth run evaluated five further variants against
  the promoted three-seed ensemble and rejected all of them: causal signed
  recency history 0.6049889157; staged train-only click pretraining
  0.6050792595; staged censored watch-time pretraining 0.6046068034;
  train-only recency-weighted BPR 0.6043204549; and a field-pair-gated FM
  architecture 0.6052951724. That campaign stopped with complete priority
  coverage and zero manual interventions.

## Boundaries and next-step implications

- Same-impression click, watch time, like, follow, comment, and forward are
  outcomes, never inference features. Auxiliary experiments may read them from
  TRAIN only.
- Do not use raw validation-date fitting: prevalence drift exists but a date-ID
  rule would be fragile for the later held-out period.
- Do not open DIN/SIM attention unless a materially stronger pooled-history
  representation first beats the promoted parent across paired seeds.
- Do not open further auxiliary heads unless there is a concrete interference
  mitigation beyond the rejected shared-embedding click/watch formulations.
- Do not retry generic recency weighting or field-pair gating without a
  materially different mechanism; both were tested in the full ensemble.
- Static-field expansion and FM embedding-size sweeps were organizer-reported
  dead ends before this campaign.
