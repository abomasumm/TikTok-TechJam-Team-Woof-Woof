"""Focused leakage and gradient tests for the causal-history experiment."""

import unittest

import numpy as np

from agent.experiments.causal_history_pipeline import (
    CausalHistoryBPRFM,
    build_causal_train_histories,
    build_frozen_target_histories,
)


class CausalHistoryTests(unittest.TestCase):
    def test_train_history_is_strictly_prior_and_excludes_tied_events(self):
        users = ["u", "u", "u", "u", "v"]
        videos = np.asarray([10, 11, 12, 13, 20], dtype=np.int32)
        timestamps = np.asarray([100, 100, 101, 102, 50], dtype=np.int64)
        labels = np.asarray([1, 1, 0, 1, 1], dtype=np.int8)

        histories, counts, final = build_causal_train_histories(
            users, videos, timestamps, labels, max_history=3
        )

        self.assertEqual(counts.tolist(), [0, 0, 2, 2, 0])
        self.assertEqual(histories[0].tolist(), [-1, -1, -1])
        self.assertEqual(histories[1].tolist(), [-1, -1, -1])
        self.assertEqual(histories[2].tolist(), [10, 11, -1])
        self.assertEqual(histories[3].tolist(), [10, 11, -1])
        self.assertEqual(final["u"], (10, 11, 13))

    def test_target_history_is_frozen_and_identical_for_same_user(self):
        frozen, counts = build_frozen_target_histories(
            ["u", "u", "new"],
            {"u": (1, 2, 3)},
            max_history=2,
        )

        self.assertEqual(frozen.tolist(), [[2, 3], [2, 3], [-1, -1]])
        self.assertEqual(counts.tolist(), [2, 2, 0])

    def test_pair_step_with_history_increases_positive_margin(self):
        model = CausalHistoryBPRFM(
            dimension=12, k=4, lr=0.01, l2=0.0, history_scale=1.0, seed=3
        )
        positive = np.asarray([[0, 4, 6, 8, 10]], dtype=np.int32)
        negative = np.asarray([[0, 5, 7, 8, 10]], dtype=np.int32)
        positive_history = np.asarray([[2, 3]], dtype=np.int32)
        negative_history = np.asarray([[2, 3]], dtype=np.int32)
        counts = np.asarray([2], dtype=np.int16)
        before = float(
            model.logits(positive, positive_history, counts)[0][0]
            - model.logits(negative, negative_history, counts)[0][0]
        )

        model.pair_step(
            positive,
            positive_history,
            counts,
            negative,
            negative_history,
            counts,
        )

        after = float(
            model.logits(positive, positive_history, counts)[0][0]
            - model.logits(negative, negative_history, counts)[0][0]
        )
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
