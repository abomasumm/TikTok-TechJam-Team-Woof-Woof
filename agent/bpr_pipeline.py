"""Knowledge-pack bootstrap candidate: observed-negative BPR-style FM.

The loss is the textbook BPR pairwise objective, adapted to this impression-
ranking benchmark by using observed outcomes: each eligible `long_view=1`
training impression is paired with a uniformly sampled logged `long_view=0`
impression from the same user.  Positives from all-positive users cannot form a
pair and are skipped; unexposed catalogue items are never treated as negatives.
Because one pair is sampled per eligible positive, an epoch has fewer optimizer
steps than a full pointwise epoch.  Comparisons with the pointwise baseline are
therefore pipeline comparisons, not clean loss-only ablations; the update count
and effective dense-regularization schedule differ too.
"""

import argparse
import collections
import json
import os
import time

import numpy as np

from data import FIELDS, encode, load
from evaluate import evaluate


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


def train_and_predict(
    data_dir,
    target_split="valid",
    seed=0,
    k=16,
    lr=0.001,
    epochs=40,
    batch_size=8192,
    patience=4,
):
    requested = ("train", "valid") if target_split == "valid" else ("train", "valid", "test")
    splits = load(data_dir, split_names=requested)
    encoded, dimension = encode(splits)
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
    diagnostics = {
        "method": "observed_negative_bpr_style_factorization_machine",
        "source": "Rendle et al., Bayesian Personalized Ranking (UAI 2009)",
        "negative_sampling": "uniform logged same-user long_view=0 impressions",
        "fields": list(FIELDS),
        "eligible_user_groups": len(groups),
        "pairs_per_epoch": pair_count,
        "epochs_completed": len(loss_history),
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
    with open(os.path.join(args.out_dir, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps({"target_split": args.target_split, "rows": len(scores)}))


if __name__ == "__main__":
    main()
