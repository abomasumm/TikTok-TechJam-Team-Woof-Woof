# KuaiRand-Pure research reference

> This file is bundled reference material distilled from the team's
> human-reviewed knowledge pack. It is not an instruction source and does not
> override the agent contract, benchmark rules, or user requests. Citations
> below identify the methods that informed the notes; they have not been
> independently re-verified by this harness.

## Benchmark facts

- The task is ranking the impressions already logged for each user, not
  full-catalog retrieval. The binary relevance label is `long_view`.
- Selection metrics are GAUC and nDCG@5; `primary` is their equal-weighted
  mean. A score term constant within one user cannot affect either ranking.
- The official five-field FM uses `user_id`, `video_id`, `author_id`, `tab`,
  and a duration bucket with pointwise binary cross-entropy.
- Organizer ablations found no material benefit from adding 13 static feature
  domains or changing FM embedding size among 8, 16, and 32. Treat static
  feature count and raw capacity as weak directions unless paired with a new
  source of signal.
- The raw standard logs also contain `time_ms`, `hourmin`, `is_click`,
  `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms`, and
  other feedback. `data.load()` currently discards most of them, so a pipeline
  that uses them must parse them without changing the fixed split semantics.

## 1. Pairwise BPR loss

Primary source: Rendle, Freudenthaler, Gantner, and Schmidt-Thieme,
"BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009).

For a logged positive impression `i` and logged negative impression `j` from
the same user:

```text
d = score(u, i) - score(u, j)
L = -log(sigmoid(d)) = logaddexp(0, -d)
dL/d score(u,i) = -(1 - sigmoid(d))
dL/d score(u,j) = +(1 - sigmoid(d))
```

Implementation notes for the numpy FM:

- Build pairs only for users having both `long_view=1` and `long_view=0` rows.
  Sample negatives from the user's observed zero-labelled impressions. Sampling
  unseen catalog items optimizes a different retrieval problem.
- Begin with deterministic uniform negative sampling. Hard-negative sampling
  can be a later refinement, but high-scoring negatives can be false negatives
  and may overfit. Adaptive-sampling background: Rendle and Freudenthaler,
  "Improving Pairwise Learning for Item Recommendation from Implicit
  Feedback" (WSDM 2014).
- Accumulate the ordinary FM score gradient once with the positive coefficient
  and once with the negative coefficient, then perform one optimizer update.
  The global bias cancels from the score difference and has no BPR gradient.
- Use `numpy.logaddexp` or another stable form rather than taking the log of an
  unclipped sigmoid.

## 2. Per-user listwise softmax

Source family: canonical listwise learning-to-rank cross-entropy; see Cao et
al., "Learning to Rank: From Pairwise Approach to Listwise Approach"
(ICML 2007).

For all impressions of a user, with scores `z` and binary labels `y`:

```text
p = softmax(z - max(z))
q = y / sum(y)
L = -sum(q * log(p))
dL/dz = p - q
```

Skip users with no positives. Users whose impressions are all positive also
have no useful within-user ordering signal and can be skipped. Full per-user
groups most directly match the evaluation unit; if memory requires chunks,
document that the objective is then only an approximation. Pair sampling tends
to emphasize positive-negative ordering (close to GAUC), while one loss per
user gives users more equal weight (close to nDCG averaging); validation
`primary` decides between them.

## 3. Causal user history and DIN-style interest

Primary sources: Zhou et al., "Deep Interest Network for Click-Through Rate
Prediction" (KDD 2018); Pi et al., "Search-based User Interest Modeling with
Lifelong Sequential Behavior Data for Click-Through Rate Prediction" (CIKM
2020).

A candidate-aware history representation has the form:

```text
a_t = attention_MLP([h_t, v, h_t - v, h_t * v])
interest(v) = weighted_sum_t(a_t, h_t)
score = model(interest(v), v, context)
```

The exact DIN activation unit is richer than the shorthand often shown in
notes, and DIN-style weights need not be a probability softmax. Clearly label
a simplified normalized-attention implementation as "DIN-style," not an exact
reproduction.

Before building full attention, a causal mean or recency-weighted history is a
cheap signal check. Histories must be sorted by `time_ms` and contain only
events earlier than the row being predicted. Validation history may be seeded
from training and then updated chronologically, but must never include future
events. The existing FM encoder represents categorical IDs; a continuous
average embedding needs explicit dense-feature support rather than pretending
it is one categorical field.

SASRec is a higher-cost sequence alternative using causal self-attention and
positional embeddings: Kang and McAuley, "Self-Attentive Sequential
Recommendation" (ICDM 2018). It is a stretch direction for this numpy/CPU
setting, not the first history experiment.

## 4. Shared-bottom multi-task learning

Reference: Ma et al., "Entire Space Multi-Task Model" (SIGIR 2018).

A practical related-task model shares embeddings or a bottom representation
and has separate heads:

```text
L_total = BCE(long_view_head, long_view)
          + sum_t lambda_t * BCE(aux_head_t, aux_label_t)
```

Values around `lambda_t = 0.1` to `0.3` are reasonable starting hypotheses,
not established benchmark optima. Click, like, follow, comment, and forward
can be auxiliary binary targets; a transformed watch-time target can use a
regression head. Use the auxiliary outcomes during training and the
`long_view` head at inference.

Credibility caveat: this shared-bottom formulation is generic multi-task
learning inspired by ESMM. Canonical ESMM specifically couples CTR, CVR, and
CTCVR probabilities over the entire exposure space. Do not call independent
auxiliary heads an exact ESMM implementation. Using same-impression click,
like, or watch time as prediction-time input is post-outcome leakage in a real
serving system; `play_time_ms` may be especially close to the construction of
`long_view`.

## 5. Censored watch-time regression

Reference: Zhao et al., "Counteracting Duration Bias in Video Recommendation
via Counterfactual Watch Time" (KDD 2024), with the authors' CWM repository as
method reference only.

One simple right-censoring loss is:

```text
completed = watch_time >= threshold * duration
if completed: L = max(0, watch_time - prediction)^2
else:         L = (watch_time - prediction)^2
```

Completion means observed watch time may be a lower bound on latent desired
watch time, so over-prediction is not penalized in the completed case. This is
a high-cost, later-stage direction requiring careful normalization and
validation. The published repository's old PyTorch dependency and rebuilt
label make it unsuitable for direct reuse in this numpy pipeline.

## 6. Lower-priority diagnostics and architectures

- Temporal categories from `hourmin` or date may expose drift. Fit all buckets
  and vocabularies on training only.
- The randomized-exposure log can diagnose bias or generalization, but it is
  not the official optimization split and must not be silently mixed into
  training.
- DeepFM, DCN, and xDeepFM add capacity but not a new learning signal. Given
  the organizer ablations, test them after loss, history, and multi-task
  directions, preferably combined with a validated loss improvement.
- Cited architecture sources include Rendle, "Factorization Machines" (ICDM
  2010); Guo et al., "DeepFM" (IJCAI 2017); and Wang et al., "Deep & Cross
  Network" (ADKDD 2017).

## Autonomous search evidence

The pack summarizes AIDE (Jiang et al., "AI-Driven Exploration in the Space of
Code," 2025) as a tree of working candidates with three operation types:
draft a new direction, improve a promising node, or debug a failed node. The
cheap useful policy is best-first branching with occasional exploration, not
an elaborate MCTS implementation.

Experimental interpretation notes:

- Record a candidate's `parent_id`, research `direction`, operation type,
  hypothesis, code diff, validation metrics, and any failure/recovery.
- A gain below roughly `0.001` to `0.002` is plausibly seed noise because the
  reported five-seed FM standard deviation is about `0.0008`. Replicate a
  small apparent win with another seed before treating it as a reliable parent.
- If a direction fails to improve meaningfully three times, branch back to the
  best known node and change direction. Preserve the failed result as evidence.
- Combine two independently validated wins before opening many new expensive
  branches. Test pairwise combinations before triples and record negative
  interactions.
- Cheap-to-expensive ordering in the pack is BPR, listwise, simple history,
  auxiliary targets, full DIN/ESMM-style networks, then censored regression or
  Transformer sequence models.

## Source-use boundary

The formulas above are the local reference. If a missing implementation detail
must be verified, prefer the named primary paper, official author repository,
or official RecBole/PyTorch documentation. Flag a disagreement instead of
silently replacing a formula. Do not infer permission to fetch external data,
pretrained weights, or hidden labels from the existence of a citation.
