"""Validation-only watch-time multitask experiment on the promoted BPR FM.

The primary objective remains same-user observed-negative ``long_view`` BPR.
An auxiliary head shares the FM embeddings but has separate linear weights and
predicts a censored completion fraction from TRAIN rows only:

``watch_target = clip(play_time_ms / max(duration_ms, 1), 0, 1)``.

Clipping treats completion as the ceiling and prevents replay/outlier watch
times from dominating the auxiliary objective.  Soft-label binary cross
entropy trains the bounded completion target with a conservative fixed weight
of 0.02.  Validation watch time is never loaded.  Checkpoint selection and
inference use only the long-view head and validation ``long_view`` ranking
metrics.  This isolated experiment intentionally refuses held-out scoring.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.bpr_pipeline import (  # noqa: E402
    MODEL_FIELDS,
    _within_user_zscore,
    aligned_hour_buckets,
    append_train_vocabulary_field,
    build_pair_groups,
    sample_pairs,
    sigmoid,
    user_tab_affinity,
)
from data import encode, iter_interactions, load  # noqa: E402
from evaluate import evaluate  # noqa: E402


WATCH_WEIGHT = 0.02


def normalized_watch_target(play_time_ms, duration_ms):
    """Return completion fractions clipped to [0, 1]."""
    play_time_ms = np.asarray(play_time_ms, dtype=np.float64)
    duration_ms = np.asarray(duration_ms, dtype=np.float64)
    if play_time_ms.shape != duration_ms.shape:
        raise ValueError("play time and duration must have identical shapes")
    if not np.all(np.isfinite(play_time_ms)) or not np.all(np.isfinite(duration_ms)):
        raise ValueError("watch-time inputs must be finite")
    safe_duration = np.maximum(duration_ms, 1.0)
    return np.clip(play_time_ms / safe_duration, 0.0, 1.0).astype(np.float32)


def load_train_watch_targets(data_dir, expected_users, expected_labels):
    """Load auxiliary outcomes from TRAIN only and verify row alignment."""
    users = []
    labels = []
    play_times = []
    durations = []
    for split_name, values in iter_interactions(
        data_dir,
        split_names=("train",),
        columns=("user_id", "long_view", "play_time_ms", "duration_ms"),
    ):
        if split_name != "train":
            raise RuntimeError("a non-training outcome crossed the auxiliary boundary")
        user_id, long_view, play_time_ms, duration_ms = values
        users.append(user_id)
        labels.append(1 if long_view != "0" else 0)
        play_times.append(float(play_time_ms))
        durations.append(float(duration_ms))

    labels = np.asarray(labels, dtype=np.int8)
    if users != list(expected_users):
        raise RuntimeError("watch-time rows do not align with encoded training users")
    if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int8)):
        raise RuntimeError("watch-time rows do not align with training labels")
    return normalized_watch_target(play_times, durations)


class WatchtimeMultitaskBPRFM:
    """Shared FM embeddings with distinct long-view and watch-time heads."""

    def __init__(self, dimension, k=16, lr=0.0005, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, (dimension, k)).astype(np.float32)
        self.W_long = np.zeros(dimension, dtype=np.float32)
        self.W_watch = np.zeros(dimension, dtype=np.float32)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW_long = np.zeros_like(self.W_long)
        self.vW_long = np.zeros_like(self.W_long)
        self.mW_watch = np.zeros_like(self.W_watch)
        self.vW_watch = np.zeros_like(self.W_watch)
        self.t = 0

    def _shared_values(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * (
            (summed * summed).sum(axis=1)
            - (embeddings * embeddings).sum(axis=(1, 2))
        )
        return embeddings, summed, interaction

    def long_logits(self, X):
        embeddings, summed, interaction = self._shared_values(X)
        return self.W_long[X].sum(axis=1) + interaction, embeddings, summed

    def watch_logits(self, X):
        embeddings, summed, interaction = self._shared_values(X)
        return self.W_watch[X].sum(axis=1) + interaction, embeddings, summed

    def joint_step(
        self,
        X_positive,
        X_negative,
        X_auxiliary,
        watch_targets,
        watch_weight=WATCH_WEIGHT,
    ):
        if len(X_positive) == 0 or len(X_positive) != len(X_negative):
            raise ValueError("positive and negative BPR batches must align and be nonempty")
        if len(X_auxiliary) == 0 or len(X_auxiliary) != len(watch_targets):
            raise ValueError("auxiliary rows and watch targets must align and be nonempty")
        if watch_weight < 0.0:
            raise ValueError("watch_weight must be nonnegative")

        positive_scores, positive_embeddings, positive_summed = self.long_logits(
            X_positive
        )
        negative_scores, negative_embeddings, negative_summed = self.long_logits(
            X_negative
        )
        difference = positive_scores - negative_scores
        probability = sigmoid(difference)
        positive_coefficient = (
            (probability - 1.0) / len(X_positive)
        ).astype(np.float32)
        negative_coefficient = -positive_coefficient

        grad_v = np.zeros_like(self.V)
        grad_long = np.zeros_like(self.W_long)
        grad_watch = np.zeros_like(self.W_watch)
        np.add.at(grad_long, X_positive, positive_coefficient[:, None])
        np.add.at(grad_long, X_negative, negative_coefficient[:, None])
        np.add.at(
            grad_v,
            X_positive,
            positive_coefficient[:, None, None]
            * (positive_summed[:, None, :] - positive_embeddings),
        )
        np.add.at(
            grad_v,
            X_negative,
            negative_coefficient[:, None, None]
            * (negative_summed[:, None, :] - negative_embeddings),
        )

        watch_targets = np.asarray(watch_targets, dtype=np.float32)
        if np.any((watch_targets < 0.0) | (watch_targets > 1.0)):
            raise ValueError("normalized watch targets must lie in [0, 1]")
        watch_scores, watch_embeddings, watch_summed = self.watch_logits(X_auxiliary)
        watch_probability = sigmoid(watch_scores)
        watch_coefficient = (
            float(watch_weight)
            * (watch_probability - watch_targets)
            / len(X_auxiliary)
        ).astype(np.float32)
        np.add.at(grad_watch, X_auxiliary, watch_coefficient[:, None])
        np.add.at(
            grad_v,
            X_auxiliary,
            watch_coefficient[:, None, None]
            * (watch_summed[:, None, :] - watch_embeddings),
        )

        grad_v += self.l2 * self.V
        grad_long += self.l2 * self.W_long
        grad_watch += self.l2 * self.W_watch
        self._adam_update(grad_v, grad_long, grad_watch)
        bpr_loss = float(np.mean(np.logaddexp(0.0, -difference)))
        watch_loss = float(
            np.mean(np.logaddexp(0.0, watch_scores) - watch_targets * watch_scores)
        )
        return bpr_loss, watch_loss

    def _adam_update(self, grad_v, grad_long, grad_watch):
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, mean, variance in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W_long, grad_long, self.mW_long, self.vW_long),
            (self.W_watch, grad_watch, self.mW_watch, self.vW_watch),
        ):
            mean *= beta1
            mean += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * gradient * gradient
            mean_hat = mean / (1.0 - beta1**self.t)
            variance_hat = variance / (1.0 - beta2**self.t)
            parameter -= self.lr * mean_hat / (np.sqrt(variance_hat) + epsilon)

    def predict_long(self, X, batch_size=200_000):
        chunks = [
            self.long_logits(X[start : start + batch_size])[0]
            for start in range(0, len(X), batch_size)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


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
    user_tab_weight=0.1,
    watch_weight=WATCH_WEIGHT,
):
    if target_split != "valid":
        raise ValueError("this isolated experiment is validation-only")
    if watch_weight < 0.0:
        raise ValueError("watch_weight must be nonnegative")

    splits = load(data_dir, split_names=("train", "valid"))
    encoded, dimension = encode(splits, duration_buckets=duration_buckets)
    hour_values = {
        split_name: aligned_hour_buckets(data_dir, split_name, values[2])
        for split_name, values in encoded.items()
    }
    encoded, dimension, hour_vocabulary = append_train_vocabulary_field(
        encoded, dimension, hour_values
    )
    X_train, y_train, train_users = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]
    watch_targets = load_train_watch_targets(data_dir, train_users, y_train)
    groups = build_pair_groups(train_users, y_train)

    model = WatchtimeMultitaskBPRFM(dimension, k=k, lr=lr, seed=seed)
    pair_rng = np.random.default_rng(seed)
    auxiliary_rng = np.random.default_rng(seed + 104729)
    best_primary = -1.0
    best_state = None
    bad_epochs = 0
    bpr_history = []
    watch_history = []
    validation_history = []
    pair_count = 0

    for _epoch in range(1, epochs + 1):
        positive_indices, negative_indices = sample_pairs(groups, pair_rng)
        pair_count = len(positive_indices)
        auxiliary_indices = auxiliary_rng.choice(
            len(X_train), size=pair_count, replace=False
        )
        bpr_losses = []
        watch_losses = []
        for start in range(0, pair_count, batch_size):
            stop = start + batch_size
            positive = positive_indices[start:stop]
            negative = negative_indices[start:stop]
            auxiliary = auxiliary_indices[start:stop]
            bpr_loss, watch_loss = model.joint_step(
                X_train[positive],
                X_train[negative],
                X_train[auxiliary],
                watch_targets[auxiliary],
                watch_weight=watch_weight,
            )
            bpr_losses.append(bpr_loss)
            watch_losses.append(watch_loss)

        validation_metrics = evaluate(
            valid_users, y_valid, model.predict_long(X_valid)
        )
        bpr_history.append(float(np.mean(bpr_losses)))
        watch_history.append(float(np.mean(watch_losses)))
        validation_history.append(float(validation_metrics["primary"]))
        if validation_metrics["primary"] > best_primary + 1e-5:
            best_primary = float(validation_metrics["primary"])
            best_state = (
                model.V.copy(),
                model.W_long.copy(),
                model.W_watch.copy(),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.V, model.W_long, model.W_watch = best_state
    scores = model.predict_long(X_valid).astype(np.float64)
    if user_tab_weight:
        affinity = user_tab_affinity(data_dir, "valid", valid_users)
        scores = _within_user_zscore(scores, valid_users) + user_tab_weight * _within_user_zscore(
            affinity, valid_users
        )

    diagnostics = {
        "method": "shared_embedding_long_view_bpr_with_training_only_watch_fraction",
        "primary_objective": "same-user observed-negative long_view BPR",
        "auxiliary_objective": "TRAIN-only soft-label BCE on clipped watch fraction",
        "watch_target": "clip(play_time_ms / max(duration_ms, 1), 0, 1)",
        "watch_weight": float(watch_weight),
        "watch_target_mean": float(np.mean(watch_targets)),
        "watch_target_completion_rate": float(np.mean(watch_targets >= 1.0)),
        "inference_head": "long_view_only",
        "target_split_auxiliary_outcomes_loaded": False,
        "checkpoint_metric": "validation long_view primary only",
        "fields": list(MODEL_FIELDS),
        "duration_buckets": int(duration_buckets),
        "hour_vocabulary": sorted(int(value) for value in hour_vocabulary),
        "learning_rate": float(lr),
        "user_tab_weight": float(user_tab_weight),
        "seed": int(seed),
        "eligible_user_groups": len(groups),
        "pairs_per_epoch": pair_count,
        "auxiliary_examples_per_epoch": pair_count,
        "epochs_completed": len(bpr_history),
        "train_bpr_loss": bpr_history,
        "train_watch_bce": watch_history,
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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    scores, diagnostics = train_and_predict(
        args.data_dir,
        target_split=args.target_split,
        seed=args.seed,
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
