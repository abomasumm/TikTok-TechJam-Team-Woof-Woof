import unittest

import numpy as np

from agent.experiments.hour_field_pipeline import append_train_vocabulary_field


class HourFieldPipelineTests(unittest.TestCase):
    def test_context_field_uses_training_vocabulary_and_unknown_slot(self):
        encoded = {
            "train": (
                np.asarray([[0, 2], [1, 3]], dtype=np.int32),
                np.asarray([1, 0], dtype=np.float32),
                ["u1", "u2"],
            ),
            "valid": (
                np.asarray([[0, 4], [1, 5]], dtype=np.int32),
                np.asarray([0, 1], dtype=np.float32),
                ["u1", "u2"],
            ),
        }

        augmented, dimension, vocabulary = append_train_vocabulary_field(
            encoded,
            base_dimension=6,
            train_values=np.asarray([0, 2], dtype=np.int8),
            valid_values=np.asarray([2, 5], dtype=np.int8),
        )

        self.assertEqual(vocabulary, {0: 0, 2: 1})
        self.assertEqual(dimension, 9)
        self.assertEqual(augmented["train"][0][:, -1].tolist(), [6, 7])
        self.assertEqual(augmented["valid"][0][:, -1].tolist(), [7, 8])
        self.assertEqual(augmented["valid"][0].shape, (2, 3))

    def test_context_alignment_is_checked(self):
        encoded = {
            "train": (
                np.asarray([[0]], dtype=np.int32),
                np.asarray([1], dtype=np.float32),
                ["u1"],
            ),
            "valid": (
                np.asarray([[1]], dtype=np.int32),
                np.asarray([0], dtype=np.float32),
                ["u1"],
            ),
        }

        with self.assertRaisesRegex(ValueError, "training context"):
            append_train_vocabulary_field(
                encoded,
                base_dimension=2,
                train_values=np.asarray([], dtype=np.int8),
                valid_values=np.asarray([0], dtype=np.int8),
            )


if __name__ == "__main__":
    unittest.main()
