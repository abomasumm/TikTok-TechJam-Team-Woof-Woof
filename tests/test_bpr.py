import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from agent.bpr_pipeline import (
    BPRFM,
    _require_masked_test_view,
    _within_user_zscore,
    append_train_vocabulary_field,
    build_pair_groups,
    sample_pairs,
    user_tab_affinity,
)


class BPRTests(unittest.TestCase):
    def test_test_prediction_requires_masked_final_view_manifest(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with self.assertRaisesRegex(RuntimeError, "masked final data view"):
                _require_masked_test_view(data_dir)

            manifest_path = os.path.join(data_dir, "view_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 2,
                        "mode": "development",
                        "test_rows_present": False,
                        "test_outcomes_masked": False,
                    },
                    handle,
                )
            with self.assertRaisesRegex(RuntimeError, "masked outcomes"):
                _require_masked_test_view(data_dir)

            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 2,
                        "mode": "final",
                        "test_rows_present": True,
                        "test_outcomes_masked": True,
                    },
                    handle,
                )
            _require_masked_test_view(data_dir)

    def test_groups_skip_users_without_both_labels(self):
        users = ["mixed", "mixed", "positive", "negative"]
        labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
        groups = build_pair_groups(users, labels)
        self.assertEqual(len(groups), 1)
        positives, negatives = groups[0]
        self.assertEqual(positives.tolist(), [0])
        self.assertEqual(negatives.tolist(), [1])

    def test_sampled_pairs_are_logged_same_user_pos_and_neg(self):
        users = ["a", "a", "a", "b", "b"]
        labels = np.asarray([1, 0, 0, 1, 0], dtype=np.float32)
        groups = build_pair_groups(users, labels)
        positives, negatives = sample_pairs(groups, np.random.default_rng(7))
        for positive, negative in zip(positives, negatives):
            self.assertEqual(users[positive], users[negative])
            self.assertEqual(labels[positive], 1)
            self.assertEqual(labels[negative], 0)

    def test_one_gradient_step_increases_pair_margin(self):
        model = BPRFM(dim=5, k=3, lr=0.05, l2=0.0, seed=3)
        positive = np.asarray([[0, 1]], dtype=np.int32)
        negative = np.asarray([[0, 2]], dtype=np.int32)
        before = float(model.logits(positive)[0][0] - model.logits(negative)[0][0])
        loss = model.pair_step(positive, negative)
        after = float(model.logits(positive)[0][0] - model.logits(negative)[0][0])
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(after, before)

    def test_hour_field_vocabulary_is_fit_on_train_only(self):
        encoded = {
            "train": (
                np.asarray([[0], [1]], dtype=np.int32),
                np.asarray([1, 0], dtype=np.float32),
                ["u1", "u2"],
            ),
            "valid": (
                np.asarray([[0], [1]], dtype=np.int32),
                np.asarray([0, 1], dtype=np.float32),
                ["u1", "u2"],
            ),
        }
        augmented, dimension, vocabulary = append_train_vocabulary_field(
            encoded,
            2,
            {
                "train": np.asarray([0, 2], dtype=np.int8),
                "valid": np.asarray([2, 5], dtype=np.int8),
            },
        )
        self.assertEqual(vocabulary, {0: 0, 2: 1})
        self.assertEqual(dimension, 5)
        self.assertEqual(augmented["valid"][0][:, -1].tolist(), [3, 4])

    def test_within_user_zscore_does_not_mix_users(self):
        transformed = _within_user_zscore(
            np.asarray([1.0, 3.0, 10.0]), ["a", "a", "b"]
        )
        np.testing.assert_allclose(transformed, [-1.0, 1.0, 0.0])

    def test_tab_affinity_never_requests_target_outcomes(self):
        calls = []

        def fake_interactions(_data_dir, split_names, columns):
            calls.append((split_names, columns))
            if split_names == ("train",):
                yield "train", ("u", "1", "1")
            else:
                yield "valid", ("u", "1")

        with mock.patch("agent.bpr_pipeline.iter_interactions", fake_interactions):
            values = user_tab_affinity("unused", "valid", ["u"])
        self.assertEqual(values.shape, (1,))
        self.assertEqual(calls[-1][1], ("user_id", "tab"))


if __name__ == "__main__":
    unittest.main()
