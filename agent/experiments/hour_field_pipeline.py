"""Isolated BPR experiment with a four-hour context field.

This candidate keeps the observed-negative BPR objective and adds one
impression-time feature: ``hour_4h = floor(hour / 4)``.  Hour values and the
30-bin duration quantiles are fitted from TRAIN only.  Validation outcomes are
used solely for early stopping and never enter the feature matrix.

The experiment intentionally supports validation only.  It cannot be used to
score or inspect the held-out split without first being reviewed and promoted
into the final pipeline.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# A direct ``python agent/experiments/...`` invocation places only this nested
# directory on ``sys.path``. Add the repository root so the experiment has the
# same import behavior as an orchestrator-launched candidate.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.bpr_pipeline import BPRFM, build_pair_groups, sample_pairs
from data import FIELDS, encode, iter_interactions, load
from evaluate import evaluate


EXPERIMENT_FIELDS = [*FIELDS, "hour_4h"]


def aligned_hour_buckets(data_dir, split_name, expected_users, expected_labels):
    """Read a known-at-impression context and verify row/label alignment."""
    buckets = []
    streamed_users = []
    streamed_labels = []
    for observed_split, values in iter_interactions(
        data_dir,
        split_names=(split_name,),
        columns=("user_id", "hourmin", "long_view"),
    ):
        if observed_split != split_name:
            raise RuntimeError("an unexpected split crossed the experiment boundary")
        user_id, hourmin, long_view = values
        hour = int(hourmin) // 100
        if not 0 <= hour <= 23:
            raise ValueError(f"hourmin produced an invalid hour: {hourmin!r}")
        buckets.append(hour // 4)
        streamed_users.append(user_id)
        streamed_labels.append(1 if long_view != "0" else 0)

    expected_labels = np.asarray(expected_labels, dtype=np.int8)
    streamed_labels = np.asarray(streamed_labels, dtype=np.int8)
    if streamed_users != list(expected_users):
        raise RuntimeError("raw hour rows do not align with encoded users")
    if not np.array_equal(streamed_labels, expected_labels):
        raise RuntimeError("raw hour rows do not align with encoded labels")
    return np.asarray(buckets, dtype=np.int8)


def append_train_vocabulary_field(
    encoded,
    base_dimension,
    train_values,
    valid_values,
):
    """Append one categorical field using only the training vocabulary."""
    train_values = np.asarray(train_values)
    valid_values = np.asarray(valid_values)
    if train_values.shape != (len(encoded["train"][1]),):
        raise ValueError("training context must align with training labels")
    if valid_values.shape != (len(encoded["valid"][1]),):
        raise ValueError("validation context must align with validation labels")

    vocabulary = {
        value: index for index, value in enumerate(np.unique(train_values).tolist())
    }
    unknown = len(vocabulary)

    augmented = {}
    for split_name, values in (("train", train_values), ("valid", valid_values)):
        X, labels, users = encoded[split_name]
        context = np.fromiter(
            (vocabulary.get(value, unknown) for value in values.tolist()),
            dtype=np.int32,
            count=len(values),
        )
        context += int(base_dimension)
        augmented[split_name] = (
            np.column_stack((X, context)).astype(np.int32, copy=False),
            labels,
            users,
        )

    return augmented, int(base_dimension + len(vocabulary) + 1), vocabulary


def train_and_predict(
    data_dir,
    target_split="valid",
    seed=0,
    k=16,
    lr=0.0005,
    epochs=40,
    batch_size=8192,
    patience=4,
    duration_buckets=30,
):
    if target_split != "valid":
        raise ValueError("this isolated experiment is validation-only")

    splits = load(data_dir, split_names=("train", "valid"))
    encoded, base_dimension = encode(splits, duration_buckets=duration_buckets)
    train_hours = aligned_hour_buckets(
        data_dir,
        "train",
        encoded["train"][2],
        encoded["train"][1],
    )
    valid_hours = aligned_hour_buckets(
        data_dir,
        "valid",
        encoded["valid"][2],
        encoded["valid"][1],
    )
    encoded, dimension, hour_vocabulary = append_train_vocabulary_field(
        encoded,
        base_dimension,
        train_hours,
        valid_hours,
    )

    X_train, y_train, train_users = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]
    groups = build_pair_groups(train_users, y_train)
    model = BPRFM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    best_state = None
    bad_epochs = 0
    loss_history = []
    validation_history = []
    pair_count = 0

    for _epoch in range(1, epochs + 1):
        positive_indices, negative_indices = sample_pairs(groups, rng)
        pair_count = len(positive_indices)
        losses = []
        for start in range(0, pair_count, batch_size):
            positive = positive_indices[start : start + batch_size]
            negative = negative_indices[start : start + batch_size]
            losses.append(model.pair_step(X_train[positive], X_train[negative]))

        metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
        loss_history.append(float(np.mean(losses)))
        validation_history.append(float(metrics["primary"]))
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = float(metrics["primary"])
            best_state = (model.V.copy(), model.W.copy())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.V, model.W = best_state
    scores = model.predict(X_valid).astype(np.float64)
    diagnostics = {
        "method": "observed_negative_bpr_fm_with_four_hour_field",
        "fields": EXPERIMENT_FIELDS,
        "hour_definition": "floor((hourmin // 100) / 4)",
        "hour_vocabulary": sorted(int(value) for value in hour_vocabulary),
        "hour_vocabulary_source": "train_only",
        "duration_buckets": int(duration_buckets),
        "duration_edges_source": "train_only",
        "learning_rate": float(lr),
        "seed": int(seed),
        "target_outcomes_used_as_features": False,
        "eligible_user_groups": len(groups),
        "pairs_per_epoch": pair_count,
        "epochs_completed": len(loss_history),
        "train_loss": loss_history,
        "validation_primary_during_training": validation_history,
        "best_validation_primary_during_training": best_primary,
    }
    return scores, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_split", choices=("valid",), default="valid")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--duration_buckets", type=int, default=30)
    args = parser.parse_args()
    if args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.duration_buckets < 2:
        raise ValueError("duration_buckets must be at least 2")

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    scores, diagnostics = train_and_predict(
        args.data_dir,
        target_split=args.target_split,
        seed=args.seed,
        lr=args.lr,
        duration_buckets=args.duration_buckets,
    )
    diagnostics["elapsed_seconds"] = time.time() - started
    diagnostics["target_split"] = args.target_split
    np.save(os.path.join(args.out_dir, "scores.npy"), scores)
    with open(
        os.path.join(args.out_dir, "diagnostics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps({"target_split": args.target_split, "rows": len(scores)}))


if __name__ == "__main__":
    main()
