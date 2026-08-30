"""Iteration 0: the official FM baseline, reshaped into the agent harness's
I/O contract (writes metrics.json + valid_scores.npy + test_scores.npy to
--out_dir). This is NOT authored by the LLM -- it's the fixed starting point
every run branches from, so iteration 0's numbers double as a reproduction
check against baseline_scores.json.
"""
import argparse, json, os, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def train_and_score(data_dir, seed=0, k=16, lr=0.001, epochs=40, bs=8192, patience=4):
    splits = load(data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state

    valid_scores = m.predict(Xva)
    test_scores = m.predict(Xte)
    valid_metrics = evaluate(uva, yva, valid_scores)
    test_metrics = evaluate(ute, yte, test_scores)
    return valid_metrics, test_metrics, valid_scores, test_scores


def _native(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    valid_metrics, test_metrics, valid_scores, test_scores = train_and_score(a.data_dir, seed=a.seed)
    elapsed = time.time() - t0

    np.save(os.path.join(a.out_dir, 'valid_scores.npy'), valid_scores.astype(np.float64))
    np.save(os.path.join(a.out_dir, 'test_scores.npy'), test_scores.astype(np.float64))
    valid_metrics, test_metrics = _native(valid_metrics), _native(test_metrics)
    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'valid': valid_metrics, 'test': test_metrics, 'elapsed_sec': elapsed}, fh, indent=2)

    print(f"AGENT_RESULT_JSON: {json.dumps({'valid': valid_metrics, 'test': test_metrics})}")


if __name__ == '__main__':
    main()
