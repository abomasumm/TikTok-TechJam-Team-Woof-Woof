import unittest
from unittest import mock

import numpy as np

from agent.experiments import watchtime_multitask_pipeline as watchtime


class WatchtimeMultitaskTests(unittest.TestCase):
    def test_normalized_watch_target_clips_completion_and_bad_duration(self):
        targets = watchtime.normalized_watch_target(
            [0.0, 50.0, 200.0, 10.0],
            [100.0, 100.0, 100.0, 0.0],
        )

        np.testing.assert_allclose(targets, [0.0, 0.5, 1.0, 1.0])

    @mock.patch.object(watchtime, "iter_interactions")
    def test_loader_requests_training_only_and_checks_alignment(self, stream):
        stream.return_value = iter(
            [
                ("train", ("u1", "1", "50", "100")),
                ("train", ("u2", "0", "0", "100")),
            ]
        )

        targets = watchtime.load_train_watch_targets(
            "unused",
            ["u1", "u2"],
            np.asarray([1, 0], dtype=np.float32),
        )

        np.testing.assert_allclose(targets, [0.5, 0.0])
        stream.assert_called_once_with(
            "unused",
            split_names=("train",),
            columns=("user_id", "long_view", "play_time_ms", "duration_ms"),
        )

    def test_watch_head_does_not_directly_change_long_view_inference(self):
        model = watchtime.WatchtimeMultitaskBPRFM(
            dimension=5, k=3, lr=0.01, l2=0.0, seed=4
        )
        X = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        before = model.predict_long(X)

        model.W_watch += 100.0
        after = model.predict_long(X)

        np.testing.assert_array_equal(after, before)

    def test_joint_step_updates_watch_head_and_increases_bpr_margin(self):
        model = watchtime.WatchtimeMultitaskBPRFM(
            dimension=6, k=3, lr=0.05, l2=0.0, seed=7
        )
        positive = np.asarray([[0, 1]], dtype=np.int32)
        negative = np.asarray([[0, 2]], dtype=np.int32)
        auxiliary = np.asarray([[3, 4]], dtype=np.int32)
        before_margin = float(
            model.long_logits(positive)[0][0] - model.long_logits(negative)[0][0]
        )
        before_watch = model.W_watch.copy()

        losses = model.joint_step(
            positive,
            negative,
            auxiliary,
            np.asarray([1.0], dtype=np.float32),
            watch_weight=0.02,
        )
        after_margin = float(
            model.long_logits(positive)[0][0] - model.long_logits(negative)[0][0]
        )

        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertGreater(after_margin, before_margin)
        self.assertFalse(np.array_equal(model.W_watch, before_watch))


if __name__ == "__main__":
    unittest.main()
