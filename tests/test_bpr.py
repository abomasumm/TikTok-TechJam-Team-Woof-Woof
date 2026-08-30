import unittest

import numpy as np

from agent.bpr_pipeline import BPRFM, build_pair_groups, sample_pairs


class BPRTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
