"""Promoted bootstrap: hour-aware, tab-calibrated BPR FM ensemble.

The loss is the textbook BPR pairwise objective, adapted to this impression-
ranking benchmark by using observed outcomes: each eligible `long_view=1`
training impression is paired with a uniformly sampled logged `long_view=0`
impression from the same user.  Positives from all-positive users cannot form a
pair and are skipped; unexposed catalogue items are never treated as negatives.
Each component also uses 30 training-fitted duration buckets, a four-hour
categorical field whose vocabulary is fitted on training only, and a smoothed
training-only user-by-tab affinity. Three deterministic components are averaged
after within-user score standardization. No target outcome is used by the
context features or ensemble aggregation.

Because one pair is sampled per eligible positive, an epoch has fewer optimizer
steps than a full pointwise epoch. Comparisons with the pointwise baseline are
pipeline comparisons, not clean loss-only ablations; the update count and
effective dense-regularization schedule differ too.
"""

import argparse
import collections
import json
import os
import time

import numpy as np

from data import FIELDS, encode, iter_interactions, load
from evaluate import evaluate


MODEL_FIELDS = [*FIELDS, "hour_4h"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class BPRFM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(1)
        interaction = 0.5 * ((summed**2).sum(1) - (embeddings**2).sum((1, 2)))
        return self.W[X].sum(1) + interaction, embeddings, summed

    def pair_step(self, X_positive, X_negative):
        batch_size = len(X_positive)
        z_positive, e_positive, s_positive = self.logits(X_positive)
        z_negative, e_negative, s_negative = self.logits(X_negative)
        difference = z_positive - z_negative
        probability = sigmoid(difference)
        positive_gradient = ((probability - 1.0) / batch_size).astype(np.float32)
        negative_gradient = -positive_gradient

        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, X_positive, positive_gradient[:, None])
        np.add.at(grad_w, X_negative, negative_gradient[:, None])
        np.add.at(
            grad_v,
            X_positive,
            positive_gradient[:, None, None] * (s_positive[:, None, :] - e_positive),
        )
        np.add.at(
            grad_v,
            X_negative,
            negative_gradient[:, None, None] * (s_negative[:, None, :] - e_negative),
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
            mean += (1 - beta1) * gradient
            variance *= beta2
            variance += (1 - beta2) * (gradient * gradient)
            mean_hat = mean / (1 - beta1**self.t)
            variance_hat = variance / (1 - beta2**self.t)
            parameter -= self.lr * mean_hat / (np.sqrt(variance_hat) + epsilon)

    def predict(self, X, batch_size=200_000):
        chunks = [
            self.logits(X[start : start + batch_size])[0]
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


def _within_user_zscore(values, users):
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(values)
    grouped = collections.defaultdict(list)
    for index, user_id in enumerate(users):
        grouped[user_id].append(index)
    for indices in grouped.values():
        indices = np.asarray(indices, dtype=np.int64)
        group_values = values[indices]
        scale = float(np.std(group_values))
        if scale > 1e-12:
            output[indices] = (group_values - float(np.mean(group_values))) / scale
    return output


def user_hour_affinity(data_dir, target_split, target_users):
    """Training-only smoothed user-by-four-hour long-view affinity."""
    user_count = collections.defaultdict(int)
    user_positive = collections.defaultdict(float)
    context_count = collections.defaultdict(int)
    context_positive = collections.defaultdict(float)
    total = 0
    positives = 0.0
    for _split, (user_id, hourmin, label) in iter_interactions(
        data_dir,
        split_names=("train",),
        columns=("user_id", "hourmin", "long_view"),
    ):
        hour_bucket = (int(hourmin) // 100) // 4
        outcome = 1.0 if label != "0" else 0.0
        user_count[user_id] += 1
        user_positive[user_id] += outcome
        context_count[(user_id, hour_bucket)] += 1
        context_positive[(user_id, hour_bucket)] += outcome
        total += 1
        positives += outcome
    global_rate = positives / total

    target_context = list(
        iter_interactions(
            data_dir,
            split_names=(target_split,),
            columns=("user_id", "hourmin"),
        )
    )
    if len(target_context) != len(target_users):
        raise ValueError("raw target rows do not align with encoded target rows")
    affinity = np.empty(len(target_context), dtype=np.float64)
    for index, (_split, (user_id, hourmin)) in enumerate(target_context):
        if user_id != target_users[index]:
            raise ValueError("raw target user order does not align with encoded rows")
        user_rate = (user_positive[user_id] + 50.0 * global_rate) / (
            user_count[user_id] + 50.0
        )
        key = (user_id, (int(hourmin) // 100) // 4)
        affinity[index] = (context_positive[key] + 10.0 * user_rate) / (
            context_count[key] + 10.0
        )
    return affinity


def user_tab_affinity(data_dir, target_split, target_users):
    """Training-only smoothed user-by-feed-tab long-view affinity."""
    user_count = collections.defaultdict(int)
    user_positive = collections.defaultdict(float)
    context_count = collections.defaultdict(int)
    context_positive = collections.defaultdict(float)
    total = 0
    positives = 0.0
    for _split, (user_id, tab, label) in iter_interactions(
        data_dir,
        split_names=("train",),
        columns=("user_id", "tab", "long_view"),
    ):
        outcome = 1.0 if label != "0" else 0.0
        user_count[user_id] += 1
        user_positive[user_id] += outcome
        context_count[(user_id, tab)] += 1
        context_positive[(user_id, tab)] += outcome
        total += 1
        positives += outcome
    global_rate = positives / total

    target_context = list(
        iter_interactions(
            data_dir,
            split_names=(target_split,),
            columns=("user_id", "tab"),
        )
    )
    if len(target_context) != len(target_users):
        raise ValueError("raw target rows do not align with encoded target rows")
    affinity = np.empty(len(target_context), dtype=np.float64)
    for index, (_split, (user_id, tab)) in enumerate(target_context):
        if user_id != target_users[index]:
            raise ValueError("raw target user order does not align with encoded rows")
        user_rate = (user_positive[user_id] + 50.0 * global_rate) / (
            user_count[user_id] + 50.0
        )
        key = (user_id, tab)
        affinity[index] = (context_positive[key] + 10.0 * user_rate) / (
            context_count[key] + 10.0
        )
    return affinity


def aligned_hour_buckets(data_dir, split_name, expected_users):
    """Load a known-at-impression hour field without reading target outcomes."""
    buckets = []
    streamed_users = []
    for observed_split, (user_id, hourmin) in iter_interactions(
        data_dir,
        split_names=(split_name,),
        columns=("user_id", "hourmin"),
    ):
        if observed_split != split_name:
            raise RuntimeError("an unexpected split crossed the hour-field boundary")
        hour = int(hourmin) // 100
        if not 0 <= hour <= 23:
            raise ValueError(f"hourmin produced an invalid hour: {hourmin!r}")
        streamed_users.append(user_id)
        buckets.append(hour // 4)
    if streamed_users != list(expected_users):
        raise RuntimeError("raw hour rows do not align with encoded users")
    return np.asarray(buckets, dtype=np.int8)


def append_train_vocabulary_field(encoded, base_dimension, values_by_split):
    """Append a categorical field whose vocabulary is fitted on TRAIN only."""
    train_values = np.asarray(values_by_split["train"])
    if train_values.shape != (len(encoded["train"][1]),):
        raise ValueError("training context must align with training labels")
    vocabulary = {
        value: index for index, value in enumerate(np.unique(train_values).tolist())
    }
    unknown = len(vocabulary)
    augmented = {}
    for split_name, (X, labels, users) in encoded.items():
        values = np.asarray(values_by_split[split_name])
        if values.shape != (len(labels),):
            raise ValueError(f"{split_name} context must align with labels")
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


def _train_single(
    data_dir,
    target_split="valid",
    seed=0,
    k=16,
    lr=0.0005,
    epochs=40,
    batch_size=8192,
    patience=4,
    user_hour_weight=0.0,
    user_tab_weight=0.1,
    duration_buckets=30,
):
    requested = ("train", "valid") if target_split == "valid" else ("train", "valid", "test")
    splits = load(data_dir, split_names=requested)
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
    groups = build_pair_groups(train_users, y_train)

    model = BPRFM(dimension, k=k, lr=lr, seed=seed)
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
            losses.append(model.pair_step(X_train[positive], X_train[negative]))
        valid_metrics = evaluate(valid_users, y_valid, model.predict(X_valid))
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
    scores = model.predict(encoded[target_split][0]).astype(np.float64)
    if user_hour_weight:
        target_users = encoded[target_split][2]
        affinity = user_hour_affinity(data_dir, target_split, target_users)
        scores = _within_user_zscore(scores, target_users) + user_hour_weight * _within_user_zscore(
            affinity, target_users
        )
    if user_tab_weight:
        target_users = encoded[target_split][2]
        affinity = user_tab_affinity(data_dir, target_split, target_users)
        scores = _within_user_zscore(scores, target_users) + user_tab_weight * _within_user_zscore(
            affinity, target_users
        )
    diagnostics = {
        "method": "observed_negative_bpr_style_factorization_machine",
        "source": "Rendle et al., Bayesian Personalized Ranking (UAI 2009)",
        "negative_sampling": "uniform logged same-user long_view=0 impressions",
        "fields": list(MODEL_FIELDS),
        "hour_definition": "floor((hourmin // 100) / 4)",
        "hour_vocabulary": sorted(int(value) for value in hour_vocabulary),
        "hour_vocabulary_source": "train_only",
        "eligible_user_groups": len(groups),
        "pairs_per_epoch": pair_count,
        "epochs_completed": len(loss_history),
        "train_loss": loss_history,
        "validation_primary_during_training": valid_history,
        "best_validation_primary_during_training": best_primary,
        "user_hour_weight": user_hour_weight,
        "user_tab_weight": user_tab_weight,
        "duration_buckets": duration_buckets,
    }
    return scores, diagnostics


def train_and_predict(
    data_dir,
    target_split="valid",
    seed=0,
    k=16,
    lr=0.0005,
    epochs=40,
    batch_size=8192,
    patience=4,
    user_hour_weight=0.0,
    user_tab_weight=0.1,
    duration_buckets=30,
    ensemble_size=3,
):
    """Train independent promoted models and average within-user z-scores."""
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be at least 1")
    target_users = [
        values[0]
        for _split, values in iter_interactions(
            data_dir,
            split_names=(target_split,),
            columns=("user_id",),
        )
    ]
    component_scores = []
    component_diagnostics = []
    for component_seed in range(seed, seed + ensemble_size):
        scores, diagnostics = _train_single(
            data_dir,
            target_split=target_split,
            seed=component_seed,
            k=k,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            user_hour_weight=user_hour_weight,
            user_tab_weight=user_tab_weight,
            duration_buckets=duration_buckets,
        )
        if len(scores) != len(target_users):
            raise RuntimeError("ensemble component rows do not align with target users")
        component_scores.append(_within_user_zscore(scores, target_users))
        component_diagnostics.append(diagnostics)
    scores = np.mean(np.stack(component_scores, axis=0), axis=0)
    diagnostics = {
        "method": "seed_ensemble_of_promoted_bpr_fm",
        "ensemble_size": ensemble_size,
        "component_seeds": list(range(seed, seed + ensemble_size)),
        "aggregation": "mean of per-user standardized component scores",
        "target_outcomes_used_for_aggregation": False,
        "components": component_diagnostics,
    }
    return scores, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_split", choices=("valid", "test"), default="valid")
    parser.add_argument("--ensemble_size", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    scores, diagnostics = train_and_predict(
        args.data_dir,
        target_split=args.target_split,
        seed=args.seed,
        ensemble_size=args.ensemble_size,
    )
    diagnostics["elapsed_seconds"] = time.time() - started
    diagnostics["target_split"] = args.target_split
    np.save(os.path.join(args.out_dir, "scores.npy"), scores)
    with open(os.path.join(args.out_dir, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps({"target_split": args.target_split, "rows": len(scores)}))


if __name__ == "__main__":
    main()
