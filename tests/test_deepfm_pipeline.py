"""Focused shape and gradient tests for the isolated DeepFM experiment."""

import unittest

import numpy as np

from agent.experiments.deepfm_pipeline import DeepFMBPR


class DeepFMPipelineTests(unittest.TestCase):
    def test_logits_have_one_finite_value_per_row(self):
        model = DeepFMBPR(
            dimension=30,
            field_count=6,
            k=4,
            hidden_dim=5,
            seed=2,
        )
        X = np.asarray(
            [[0, 5, 10, 15, 20, 25], [1, 6, 11, 16, 21, 26]],
            dtype=np.int32,
        )

        values = model.logits(X)

        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_pair_step_increases_margin_and_updates_mlp(self):
        model = DeepFMBPR(
            dimension=30,
            field_count=6,
            k=4,
            hidden_dim=5,
            lr=0.01,
            l2=0.0,
            mlp_l2=0.0,
            seed=4,
        )
        positive = np.asarray([[0, 5, 10, 15, 20, 25]], dtype=np.int32)
        negative = np.asarray([[0, 6, 11, 15, 21, 25]], dtype=np.int32)
        before_margin = float(model.logits(positive)[0] - model.logits(negative)[0])
        before_hidden = model.hidden_weight.copy()
        before_output = model.output_weight.copy()

        model.pair_step(positive, negative)

        after_margin = float(model.logits(positive)[0] - model.logits(negative)[0])
        self.assertGreater(after_margin, before_margin)
        self.assertFalse(np.array_equal(model.hidden_weight, before_hidden))
        self.assertFalse(np.array_equal(model.output_weight, before_output))

    def test_state_round_trip_restores_all_predictions(self):
        model = DeepFMBPR(
            dimension=30,
            field_count=6,
            k=3,
            hidden_dim=4,
            seed=7,
        )
        X = np.asarray([[0, 5, 10, 15, 20, 25]], dtype=np.int32)
        state = model.state()
        expected = model.logits(X).copy()
        for parameter in model.parameters():
            parameter += 1.0

        model.load_state(state)

        np.testing.assert_allclose(model.logits(X), expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
