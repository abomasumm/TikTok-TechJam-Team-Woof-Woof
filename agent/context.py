"""Static context handed to the LLM every iteration: task definition, constraints,
dead ends already tried, and the ranked list of unexplored directions.
Kept separate from orchestrator.py so it's easy to review/edit on its own.
"""

TASK_DESCRIPTION = """
You are improving a recommender-system ranking pipeline on the KuaiRand-Pure dataset.

TASK
- Within-user ranking: for each user, rank the videos they were actually shown (their
  logged impressions) -- this is NOT full-catalog retrieval.
- Relevance label: `long_view` (binary, 0/1), the native column already in the data.
- Metrics: GAUC and nDCG@5. primary = mean(GAUC, nDCG@5). Higher is better.
- Fixed date-based splits: train 20220408-20220421, valid 20220422-20220428,
  test 20220429-20220508 (test here is the LOCAL held-out split, not the
  competition's truly hidden test set -- you only ever see valid/test scores
  from this local evaluate.py, which mirrors the official scoring exactly).

OFFICIAL BASELINE TO BEAT (Factorization Machine, k=16, lr=0.001, 5 fields)
  valid: GAUC 0.6674 | nDCG@5 0.5357 | primary 0.6016
  test:  GAUC 0.6610 | nDCG@5 0.5282 | primary 0.5946
Random floor: primary ~0.475. Popularity baseline: primary ~0.572.
Oracle ceiling (perfect ranking): primary ~0.8645 -- so judge headroom against
0.8645, not 1.0. The baseline already captures ~31% of the attainable range.

CODE CONTRACT (do not violate -- the harness parses your file mechanically)
Your file `pipeline.py` MUST:
1. Only import from: numpy, Python standard library, and the local modules
   `data` and `evaluate` (both importable as top-level modules, e.g. `from data
   import load` / `from evaluate import evaluate` -- they live in the project
   root, which is the working directory the script is executed from).
   No other third-party packages are installed in this environment (no torch,
   no pandas, no sklearn, no pip installs available) -- numpy-only, like the
   original baseline.
2. Never modify the scoring logic. `evaluate.py`'s `evaluate(user_ids, labels,
   scores)` function is the pinned, official metric implementation -- call it,
   don't reimplement or alter it.
3. Expose a CLI: `python pipeline.py --data_dir DIR --out_dir DIR --seed N`.
   On completion it must write, inside --out_dir:
     - metrics.json : {"valid": {...evaluate() dict...}, "test": {...evaluate() dict...}}
       IMPORTANT: evaluate()'s dict values are numpy scalars (float32/float64),
       which json.dump() cannot serialize directly -- cast every value to a
       native Python float (e.g. `{k: float(v) for k, v in d.items()}`) before
       dumping, or json.dump will raise TypeError.
     - valid_scores.npy : float array, one score per row of data.load(data_dir)['valid'],
                          in the exact same row order (row i's score is for splits['valid'][i]).
     - test_scores.npy  : same, for splits['test'].
   Scores are arbitrary reals (only relative order within a user matters); NaN/Inf
   are invalid and will fail downstream validation.
4. Must run to completion within a few minutes on a single CPU core (no GPU
   available). The reference FM baseline takes ~40s.
5. Must be deterministic given --seed (seed all RNGs you use).

WHAT COUNTS AS AN IMPROVEMENT
Anything in the pipeline is fair game: features, model architecture, loss
function, training procedure/optimizer, early-stopping/regularization, how you
use the other feedback signals in the logs, etc. You are not limited to
tweaking the FM's hyperparameters.
""".strip()

DEAD_ENDS = """
ALREADY TESTED BY THE ORGANIZERS -- DO NOT RE-TRY THESE, THEY DID NOT HELP:
1. Adding more static features (bringing in all 13 CWM feature domains: music_id,
   video_type, upload_type, + 6 coarse user-side buckets like follow_user_num_range)
   on top of the original 5 fields -- primary 0.5940 vs 0.5950 for the 5-field
   version. No real difference, if anything slightly worse. Reason: user_id x
   video_id crosses already capture most of the learnable signal; coarse user-side
   buckets are redundant given user_id itself.
2. Increasing FM embedding dimension k in {8, 16, 32} -- 0.5895 / 0.5902 / 0.5887,
   essentially flat. 1.14M training rows can't support much more capacity, and
   capacity was never the bottleneck.
3. IMPORTANT STRUCTURAL FACT: purely user-side features (features that are
   CONSTANT within a user) contribute exactly zero to ranking quality on this
   task, because ranking happens within each user's impression set -- a term
   that's constant across a user's rows can't change their relative order.
   User-side information only helps through CROSS terms with item-side features
   (e.g. user_x * item_y interactions), never as a standalone additive term.
   Don't waste an iteration on pure user-side first-order features expecting
   them to move the score.
""".strip()

RANKED_IDEAS = """
UNEXPLORED DIRECTIONS -- ranked by the organizers' best guess at potential impact
(they have NOT verified these; this is where headroom is believed to be):
1. [Highest expected value] Change the loss function to match the ranking
   metrics. Training currently uses pointwise logloss (independent per-row
   binary cross-entropy), but GAUC/nDCG are ranking metrics computed WITHIN
   each user's impression set. A pairwise loss (e.g. BPR: for each user, sample
   a (positive, negative) impression pair and push score(pos) > score(neg)) or
   a listwise loss (softmax cross-entropy over each user's own impressions)
   directly optimizes what's being measured, instead of a proxy.
2. User history / behavioral sequence features. Every user has hundreds to
   thousands of interactions in train, and NONE of that sequence is currently
   used as a feature (only static IDs). Modeling recent watch history (e.g. a
   simple attention/pooling over a user's last-K interacted videos, DIN/SIM-
   style interest modeling) is completely unexplored.
3. Multi-task learning using the other logged feedback signals as auxiliary
   tasks: is_click, is_like, is_follow, is_comment, is_forward, play_time_ms
   are all present in the raw logs alongside long_view. Jointly predicting
   these (shared representation, separate output heads) may regularize/enrich
   the representation used for the long_view head.
4. [Research-depth, harder] Watch-time modeling via censored regression --
   when a video is watched all the way through, the true "would-be" watch time
   is right-censored by the video's duration, so a one-sided/censored loss
   (rather than squared error) is more correct than treating play_time_ms as a
   normal regression target. See CWM (KDD 2024, github.com/hyz20/CWM) for the
   idea -- but note CWM's actual code is numpy-incompatible (torch==1.6.0) and
   evaluates on its own rebuilt label, so treat it as a paper reference only,
   not code to import.
5. Different model architectures: DeepFM / DCN / xDeepFM. Given capacity was
   NOT the bottleneck (see dead end #2), this is lower priority than #1-4 --
   a fancier architecture is unlikely to help if the loss function and features
   are still misaligned with the task.
6. Time-based features and train/test distribution drift: hour-of-day/minute
   (`hourmin`), `date`, and whether performance degrades over the 10-day test
   window relative to the start of test.
7. [Advanced] Unbiased/counterfactual validation using the randomized-exposure
   log (log_random_4_22_to_5_08_pure.csv, 1.18M rows, NOT part of the official
   train/valid/test splits) as an extra diagnostic to check whether a model
   is overfitting to position/popularity bias in the normal logged (biased)
   traffic, rather than learning genuine preference signal.
""".strip()

def build_system_prompt():
    return (
        "You are an autonomous ML research agent improving a recommender-system "
        "ranking pipeline, working one iteration at a time inside a fixed harness.\n\n"
        + TASK_DESCRIPTION + "\n\n" + DEAD_ENDS + "\n\n" + RANKED_IDEAS + "\n\n"
        "Each turn, you will be shown the current best-performing pipeline.py code "
        "and a short history of what's been tried so far (hypothesis -> result). "
        "Propose ONE focused change per iteration -- don't bundle five ideas into "
        "one file, since if it fails or underperforms you won't know which part "
        "was responsible. Prefer building on the CURRENT BEST code shown to you "
        "rather than rewriting from scratch, unless you have a specific reason to "
        "start over. State your hypothesis briefly and concretely: what you're "
        "changing and why you expect it to help, referencing the task's own "
        "metric/loss mismatch or missing signal, not generic ML platitudes."
    )
