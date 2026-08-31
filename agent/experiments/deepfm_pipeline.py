"""Validation-only DeepFM-style extension of the promoted BPR pipeline.

This isolated architecture experiment preserves the promoted input and
training configuration: the five base categorical fields, a TRAIN-vocabulary
four-hour bucket, 30 TRAIN-fitted duration buckets, uniform observed-negative
same-user BPR, and the promoted training-only user/tab score blend.  Its one
focused change is a small one-hidden-layer MLP over the concatenated field
embeddings, added to the linear and second-order FM branches.

The experiment accepts only ``--target_split valid``. It neither loads nor
scores the held-out split, and it does not modify the canonical pipeline.
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

from agent.bpr_pipeline import (
    MODEL_FIELDS,
    _within_user_zscore,
    aligned_hour_buckets,
    append_train_vocabulary_field,
    build_pair_groups,
    sample_pairs,
    user_tab_affinity,
)
from data import encode, load
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class DeepFMBPR:
    """FM plus a small shared-embedding MLP, optimized with BPR."""

    def __init__(
        self,
        dimension,
        field_count,
        k=16,
        hidden_dim=16,
        lr=0.0005,
        l2=1e-6,
        mlp_l2=1e-6,
        seed=0,
    ):
        if dimension <= 0 or field_count <= 0 or k <= 0 or hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, (dimension, k)).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        input_dim = field_count * k
        self.hidden_weight = rng.normal(
            0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)
        ).astype(np.float32)
        self.hidden_bias = np.zeros(hidden_dim, dtype=np.float32)
        self.output_weight = rng.normal(
            0.0, np.sqrt(2.0 / hidden_dim), hidden_dim
        ).astype(np.float32)
        self.output_bias = np.zeros(1, dtype=np.float32)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.mlp_l2 = float(mlp_l2)
        self.first_moments = [
            np.zeros_like(parameter) for parameter in self.parameters()
        ]
        self.second_moments = [
            np.zeros_like(parameter) for parameter in self.parameters()
        ]
        self.t = 0

    def parameters(self):
        return (
            self.V,
            self.W,
            self.hidden_weight,
            self.hidden_bias,
            self.output_weight,
            self.output_bias,
        )

    def state(self):
        return tuple(parameter.copy() for parameter in self.parameters())

    def load_state(self, state):
        if len(state) != len(self.parameters()):
            raise ValueError("invalid DeepFM state")
        for parameter, saved in zip(self.parameters(), state):
            if parameter.shape != saved.shape:
                raise ValueError("DeepFM state shape mismatch")
            parameter[...] = saved

    def _forward(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        fm_interaction = 0.5 * (
            (summed * summed).sum(axis=1)
            - (embeddings * embeddings).sum(axis=(1, 2))
        )
        flat = embeddings.reshape(len(X), -1)
        hidden_pre = flat @ self.hidden_weight + self.hidden_bias
        hidden = np.maximum(hidden_pre, 0.0)
        deep_score = hidden @ self.output_weight + self.output_bias[0]
        score = self.W[X].sum(axis=1) + fm_interaction + deep_score
        cache = (embeddings, summed, flat, hidden_pre, hidden)
        return score, cache

    def logits(self, X):
        return self._forward(X)[0]

    def _accumulate_branch_gradients(self, gradients, X, coefficient, cache):
        grad_v, grad_w, grad_hidden_weight, grad_hidden_bias, grad_output_weight, grad_output_bias = gradients
        embeddings, summed, flat, hidden_pre, hidden = cache

        np.add.at(grad_w, X, coefficient[:, None])
        np.add.at(
            grad_v,
            X,
            coefficient[:, None, None] * (summed[:, None, :] - embeddings),
        )

        grad_output_weight += hidden.T @ coefficient
        grad_output_bias[0] += coefficient.sum()
        hidden_gradient = coefficient[:, None] * self.output_weight[None, :]
        hidden_gradient *= hidden_pre > 0.0
        grad_hidden_weight += flat.T @ hidden_gradient
        grad_hidden_bias += hidden_gradient.sum(axis=0)
        embedding_gradient = (
            hidden_gradient @ self.hidden_weight.T
        ).reshape(embeddings.shape)
        np.add.at(grad_v, X, embedding_gradient)

    def pair_step(self, X_positive, X_negative):
        if len(X_positive) == 0 or len(X_positive) != len(X_negative):
            raise ValueError("BPR batches must be non-empty aligned pairs")
        positive_score, positive_cache = self._forward(X_positive)
        negative_score, negative_cache = self._forward(X_negative)
        difference = positive_score - negative_score
        probability = sigmoid(difference)
        positive_gradient = ((probability - 1.0) / len(X_positive)).astype(np.float32)
        negative_gradient = -positive_gradient

        gradients = [np.zeros_like(parameter) for parameter in self.parameters()]
        self._accumulate_branch_gradients(
            gradients, X_positive, positive_gradient, positive_cache
        )
        self._accumulate_branch_gradients(
            gradients, X_negative, negative_gradient, negative_cache
        )
        gradients[0] += self.l2 * self.V
        gradients[1] += self.l2 * self.W
        gradients[2] += self.mlp_l2 * self.hidden_weight
        gradients[4] += self.mlp_l2 * self.output_weight
        self._adam_update(gradients)
        return float(np.mean(np.logaddexp(0.0, -difference)))

    def _adam_update(self, gradients):
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, mean, variance in zip(
            self.parameters(),
            gradients,
            self.first_moments,
            self.second_moments,
        ):
            mean *= beta1
            mean += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * (gradient * gradient)
            mean_hat = mean / (1.0 - beta1**self.t)
            variance_hat = variance / (1.0 - beta2**self.t)
            parameter -= self.lr * mean_hat / (np.sqrt(variance_hat) + epsilon)

    def predict(self, X, batch_size=100_000):
        chunks = [
            self.logits(X[start : start + batch_size])
            for start in range(0, len(X), batch_size)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def train_and_predict(
    data_dir,
    target_split="valid",
    seed=0,
    k=16,
    hidden_dim=16,
    lr=0.0005,
    epochs=40,
    batch_size=8192,
    patience=4,
    duration_buckets=30,
    user_tab_weight=0.1,
):
    if target_split != "valid":
        raise ValueError("this isolated experiment is validation-only")
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
    groups = build_pair_groups(train_users, y_train)
    model = DeepFMBPR(
        dimension,
        field_count=X_train.shape[1],
        k=k,
        hidden_dim=hidden_dim,
        lr=lr,
        seed=seed,
    )
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
            best_state = model.state()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state(best_state)
    scores = model.predict(X_valid).astype(np.float64)
    if user_tab_weight:
        affinity = user_tab_affinity(data_dir, "valid", valid_users)
        scores = _within_user_zscore(scores, valid_users) + user_tab_weight * _within_user_zscore(
            affinity, valid_users
        )
    diagnostics = {
        "method": "deepfm_style_one_hidden_layer_observed_negative_bpr",
        "architecture_change": "one ReLU MLP branch over concatenated shared embeddings",
        "fields": list(MODEL_FIELDS),
        "embedding_dimension": int(k),
        "mlp_hidden_dimension": int(hidden_dim),
        "hour_definition": "floor((hourmin // 100) / 4)",
        "hour_vocabulary": sorted(int(value) for value in hour_vocabulary),
        "hour_vocabulary_source": "train_only",
        "duration_buckets": int(duration_buckets),
        "duration_edges_source": "train_only",
        "negative_sampling": "uniform logged same-user long_view=0 impressions",
        "learning_rate": float(lr),
        "user_tab_weight": float(user_tab_weight),
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
    parser.add_argument("--hidden_dim", type=int, default=16)
    args = parser.parse_args()
    if args.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    scores, diagnostics = train_and_predict(
        args.data_dir,
        target_split=args.target_split,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
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
