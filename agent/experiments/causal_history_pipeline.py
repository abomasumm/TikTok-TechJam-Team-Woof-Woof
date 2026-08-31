"""Observed-negative BPR FM with causal mean-pooled positive-video history.

This is an isolated validation experiment, not the canonical pipeline.  The
history signal is a deliberately inexpensive precursor to DIN: the score adds
the dot product between the current video's shared FM embedding and the mean
embedding of up to ``max_history`` earlier positive videos for that user.

Leakage boundary:

* A training row sees only positive TRAIN events with a strictly smaller
  ``time_ms``. Events tied at the same timestamp are added only after every row
  in that timestamp group has received its history.
* Validation (and the contract-compatible test target) receives a frozen
  history made only from TRAIN positives. Target rows and target outcomes never
  update that history.
* Validation labels are used only for early stopping and are never model
  inputs. The evaluator is called only for validation.

The mean is an explicit dense representation in the scoring function; it is
not encoded as a fake categorical field. Gradients flow through both the
current-video occurrence and every video occurrence in the pooled history.
This is a simple mean-pooling experiment inspired by DIN/SIM, not an exact
implementation of either architecture.
"""

import argparse
import collections
import json
import os
import time

import numpy as np

from data import FIELDS, encode, iter_interactions, load
from evaluate import evaluate


VIDEO_FIELD_INDEX = FIELDS.index("video_id")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def load_aligned_train_timestamps(data_dir, expected_users, expected_labels):
    """Load only TRAIN chronology and verify alignment with ``data.load``."""
    users = []
    timestamps = []
    labels = []
    for split_name, values in iter_interactions(
        data_dir,
        split_names=("train",),
        columns=("user_id", "time_ms", "long_view"),
    ):
        if split_name != "train":
            raise RuntimeError("non-training row crossed the history boundary")
        user_id, time_ms, long_view = values
        users.append(user_id)
        timestamps.append(int(time_ms))
        labels.append(1 if long_view != "0" else 0)

    labels_array = np.asarray(labels, dtype=np.int8)
    if users != list(expected_users):
        raise RuntimeError("streaming TRAIN users do not align with encoded rows")
    if not np.array_equal(labels_array, np.asarray(expected_labels, dtype=np.int8)):
        raise RuntimeError("streaming TRAIN labels do not align with encoded rows")
    return np.asarray(timestamps, dtype=np.int64)


def build_causal_train_histories(
    users,
    video_ids,
    timestamps,
    labels,
    max_history=20,
):
    """Return strict-prior TRAIN histories and each user's final TRAIN history.

    Rows sharing the same user and timestamp receive the identical snapshot,
    taken before any outcome at that timestamp is incorporated. This makes
    ``time_ms < current_time_ms`` an explicit invariant rather than relying on
    source-file row order to break ties.
    """
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    users = list(users)
    video_ids = np.asarray(video_ids, dtype=np.int32)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    row_count = len(users)
    if not (
        video_ids.shape == (row_count,)
        and timestamps.shape == (row_count,)
        and labels.shape == (row_count,)
    ):
        raise ValueError("history inputs must be aligned one-dimensional arrays")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("history labels must be binary")

    histories = np.full((row_count, max_history), -1, dtype=np.int32)
    counts = np.zeros(row_count, dtype=np.int16)
    rows_by_user = collections.defaultdict(list)
    for row_index, user_id in enumerate(users):
        rows_by_user[user_id].append(row_index)

    final_histories = {}
    for user_id, row_indices in rows_by_user.items():
        row_indices.sort(key=lambda index: (int(timestamps[index]), index))
        positive_history = collections.deque(maxlen=max_history)
        cursor = 0
        while cursor < len(row_indices):
            timestamp = timestamps[row_indices[cursor]]
            end = cursor + 1
            while (
                end < len(row_indices)
                and timestamps[row_indices[end]] == timestamp
            ):
                end += 1

            snapshot = tuple(positive_history)
            snapshot_count = len(snapshot)
            if snapshot_count:
                for row_index in row_indices[cursor:end]:
                    histories[row_index, :snapshot_count] = snapshot
                    counts[row_index] = snapshot_count

            # Update only after every same-timestamp row has received the
            # pre-event snapshot, so tied events cannot leak into one another.
            for row_index in row_indices[cursor:end]:
                if labels[row_index] == 1:
                    positive_history.append(int(video_ids[row_index]))
            cursor = end
        final_histories[user_id] = tuple(positive_history)
    return histories, counts, final_histories


def build_frozen_target_histories(target_users, final_train_histories, max_history=20):
    """Seed every target row from TRAIN only; never update from target rows."""
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    target_users = list(target_users)
    histories = np.full((len(target_users), max_history), -1, dtype=np.int32)
    counts = np.zeros(len(target_users), dtype=np.int16)
    for row_index, user_id in enumerate(target_users):
        values = final_train_histories.get(user_id, ())[-max_history:]
        count = len(values)
        if count:
            histories[row_index, :count] = values
            counts[row_index] = count
    return histories, counts


class CausalHistoryBPRFM:
    def __init__(
        self,
        dimension,
        k=16,
        lr=0.0005,
        l2=1e-6,
        history_scale=1.0,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, (dimension, k)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.lr = lr
        self.l2 = l2
        self.history_scale = np.float32(history_scale)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def _history_mean(self, history_ids, history_counts):
        mask = history_ids >= 0
        safe_ids = np.where(mask, history_ids, 0)
        pooled = (self.V[safe_ids] * mask[:, :, None]).sum(axis=1)
        denominator = np.maximum(history_counts, 1).astype(np.float32)[:, None]
        return pooled / denominator

    def logits(self, X, history_ids, history_counts):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * (
            (summed * summed).sum(axis=1)
            - (embeddings * embeddings).sum(axis=(1, 2))
        )
        history_mean = self._history_mean(history_ids, history_counts)
        video_embeddings = embeddings[:, VIDEO_FIELD_INDEX, :]
        history_interaction = self.history_scale * np.einsum(
            "ij,ij->i", video_embeddings, history_mean
        )
        scores = self.W[X].sum(axis=1) + interaction + history_interaction
        return scores, embeddings, summed, history_mean, video_embeddings

    def _accumulate_gradient(
        self,
        grad_v,
        grad_w,
        X,
        history_ids,
        history_counts,
        coefficient,
        embeddings,
        summed,
        history_mean,
        video_embeddings,
    ):
        np.add.at(grad_w, X, coefficient[:, None])
        np.add.at(
            grad_v,
            X,
            coefficient[:, None, None] * (summed[:, None, :] - embeddings),
        )

        # Current-video derivative of <v_current, mean(history)>.
        np.add.at(
            grad_v,
            X[:, VIDEO_FIELD_INDEX],
            coefficient[:, None] * self.history_scale * history_mean,
        )

        # Historical-video derivatives. Padding positions are excluded and the
        # mean normalization is per row, including when histories are short.
        mask = history_ids >= 0
        if np.any(mask):
            denominator = np.maximum(history_counts, 1).astype(np.float32)
            per_history = (
                coefficient[:, None]
                * self.history_scale
                * video_embeddings
                / denominator[:, None]
            )
            contributions = np.broadcast_to(
                per_history[:, None, :],
                history_ids.shape + (self.V.shape[1],),
            )
            np.add.at(grad_v, history_ids[mask], contributions[mask])

    def pair_step(
        self,
        X_positive,
        positive_history_ids,
        positive_history_counts,
        X_negative,
        negative_history_ids,
        negative_history_counts,
    ):
        batch_size = len(X_positive)
        positive_values = self.logits(
            X_positive, positive_history_ids, positive_history_counts
        )
        negative_values = self.logits(
            X_negative, negative_history_ids, negative_history_counts
        )
        difference = positive_values[0] - negative_values[0]
        probability = sigmoid(difference)
        positive_gradient = ((probability - 1.0) / batch_size).astype(np.float32)
        negative_gradient = -positive_gradient

        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        self._accumulate_gradient(
            grad_v,
            grad_w,
            X_positive,
            positive_history_ids,
            positive_history_counts,
            positive_gradient,
            *positive_values[1:],
        )
        self._accumulate_gradient(
            grad_v,
            grad_w,
            X_negative,
            negative_history_ids,
            negative_history_counts,
            negative_gradient,
            *negative_values[1:],
        )
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W
        self._adam_update(grad_v, grad_w)
        return float(np.mean(np.logaddexp(0.0, -difference)))

    def _adam_update(self, grad_v, grad_w):
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, mean, variance in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            mean *= beta1
            mean += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * (gradient * gradient)
            mean_hat = mean / (1.0 - beta1**self.t)
            variance_hat = variance / (1.0 - beta2**self.t)
            parameter -= self.lr * mean_hat / (np.sqrt(variance_hat) + epsilon)

    def predict(self, X, history_ids, history_counts, batch_size=50_000):
        chunks = [
            self.logits(
                X[start : start + batch_size],
                history_ids[start : start + batch_size],
                history_counts[start : start + batch_size],
            )[0]
            for start in range(0, len(X), batch_size)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def build_pair_groups(users, labels):
    grouped = collections.defaultdict(lambda: [[], []])
    for index, (user_id, label) in enumerate(zip(users, labels)):
        grouped[user_id][int(label)].append(index)
    groups = []
    for negative, positive in grouped.values():
        if positive and negative:
            groups.append(
                (
                    np.asarray(positive, dtype=np.int64),
                    np.asarray(negative, dtype=np.int64),
                )
            )
    if not groups:
        raise ValueError("BPR requires a user with both positive and negative labels")
    return groups


def sample_pairs(groups, rng):
    positive_parts = []
    negative_parts = []
    for positives, negatives in groups:
        positive_parts.append(positives)
        sampled = rng.integers(0, len(negatives), size=len(positives))
        negative_parts.append(negatives[sampled])
    positive_indices = np.concatenate(positive_parts)
    negative_indices = np.concatenate(negative_parts)
    order = rng.permutation(len(positive_indices))
    return positive_indices[order], negative_indices[order]


def train_and_predict(
    data_dir,
    target_split="valid",
    seed=0,
    k=16,
    lr=0.0005,
    epochs=40,
    batch_size=8192,
    patience=4,
    max_history=20,
    history_scale=1.0,
    duration_buckets=10,
):
    requested = (
        ("train", "valid")
        if target_split == "valid"
        else ("train", "valid", "test")
    )
    splits = load(data_dir, split_names=requested)
    encoded, dimension = encode(splits, duration_buckets=duration_buckets)
    X_train, y_train, train_users = encoded["train"]
    X_valid, y_valid, valid_users = encoded["valid"]

    timestamps = load_aligned_train_timestamps(data_dir, train_users, y_train)
    train_history_ids, train_history_counts, final_train_histories = (
        build_causal_train_histories(
            train_users,
            X_train[:, VIDEO_FIELD_INDEX],
            timestamps,
            y_train,
            max_history=max_history,
        )
    )
    valid_history_ids, valid_history_counts = build_frozen_target_histories(
        valid_users,
        final_train_histories,
        max_history=max_history,
    )
    target_users = encoded[target_split][2]
    if target_split == "valid":
        target_history_ids = valid_history_ids
        target_history_counts = valid_history_counts
    else:
        target_history_ids, target_history_counts = build_frozen_target_histories(
            target_users,
            final_train_histories,
            max_history=max_history,
        )

    groups = build_pair_groups(train_users, y_train)
    model = CausalHistoryBPRFM(
        dimension,
        k=k,
        lr=lr,
        history_scale=history_scale,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    best_state = None
    bad_epochs = 0
    loss_history = []
    valid_history = []
    pair_count = 0

    for _epoch in range(1, epochs + 1):
        positive_indices, negative_indices = sample_pairs(groups, rng)
        pair_count = len(positive_indices)
        losses = []
        for start in range(0, pair_count, batch_size):
            positive = positive_indices[start : start + batch_size]
            negative = negative_indices[start : start + batch_size]
            losses.append(
                model.pair_step(
                    X_train[positive],
                    train_history_ids[positive],
                    train_history_counts[positive],
                    X_train[negative],
                    train_history_ids[negative],
                    train_history_counts[negative],
                )
            )
        valid_scores = model.predict(
            X_valid, valid_history_ids, valid_history_counts
        )
        valid_metrics = evaluate(valid_users, y_valid, valid_scores)
        loss_history.append(float(np.mean(losses)))
        valid_history.append(float(valid_metrics["primary"]))
        if valid_metrics["primary"] > best_primary + 1e-5:
            best_primary = float(valid_metrics["primary"])
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.V, model.W = best_state
    scores = model.predict(
        encoded[target_split][0], target_history_ids, target_history_counts
    ).astype(np.float64)
    diagnostics = {
        "method": "causal_mean_pooled_positive_video_history_bpr_fm",
        "source": (
            "Knowledge pack section 3; mean-pooling precursor inspired by "
            "Zhou et al. (DIN, KDD 2018) and Pi et al. (SIM, CIKM 2020)"
        ),
        "implementation_scope": "simple mean pooling, not exact DIN or SIM",
        "negative_sampling": "uniform logged same-user long_view=0 impressions",
        "history_source": "strictly earlier TRAIN positives only",
        "same_timestamp_policy": "excluded until all tied rows are featurized",
        "target_history_policy": "frozen from TRAIN; target outcomes never incorporated",
        "history_gradient": "current and pooled historical video embeddings",
        "max_history": int(max_history),
        "history_scale": float(history_scale),
        "duration_buckets": int(duration_buckets),
        "train_rows_with_history": int(np.count_nonzero(train_history_counts)),
        "valid_rows_with_history": int(np.count_nonzero(valid_history_counts)),
        "mean_train_history_length": float(np.mean(train_history_counts)),
        "mean_valid_history_length": float(np.mean(valid_history_counts)),
        "eligible_user_groups": len(groups),
        "pairs_per_epoch": pair_count,
        "epochs_completed": len(loss_history),
        "learning_rate": float(lr),
        "train_loss": loss_history,
        "validation_primary_during_training": valid_history,
        "best_validation_primary_during_training": best_primary,
    }
    return scores, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_split", choices=("valid", "test"), default="valid")
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--history_scale", type=float, default=1.0)
    parser.add_argument("--duration_buckets", type=int, default=10)
    args = parser.parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.max_history <= 0:
        raise ValueError("max_history must be positive")
    if args.duration_buckets < 2:
        raise ValueError("duration_buckets must be at least 2")

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    scores, diagnostics = train_and_predict(
        args.data_dir,
        target_split=args.target_split,
        seed=args.seed,
        lr=args.learning_rate,
        max_history=args.max_history,
        history_scale=args.history_scale,
        duration_buckets=args.duration_buckets,
    )
    diagnostics["elapsed_seconds"] = float(time.time() - started)
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
