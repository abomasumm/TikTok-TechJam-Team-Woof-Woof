import unittest

from evaluate import auc, evaluate, ndcg_at_k


class EvaluateTests(unittest.TestCase):
    def test_auc_ties_receive_half_credit(self):
        self.assertAlmostEqual(auc([0, 1], [0.25, 0.25]), 0.5)

    def test_ndcg_zero_positive_user_is_zero(self):
        self.assertEqual(ndcg_at_k([0, 0, 0], 5), 0.0)

    def test_perfect_within_user_ranking(self):
        result = evaluate(
            ["u1", "u1", "u2", "u2"],
            [1, 0, 0, 1],
            [2.0, 1.0, 1.0, 2.0],
        )
        self.assertAlmostEqual(result["GAUC"], 1.0)
        self.assertAlmostEqual(result["nDCG@5"], 1.0)
        self.assertAlmostEqual(result["primary"], 1.0)

    def test_all_positive_user_is_excluded_from_gauc_but_in_ndcg(self):
        result = evaluate(["u", "u"], [1, 1], [0.1, 0.2])
        self.assertEqual(result["GAUC"], 0.5)
        self.assertEqual(result["nDCG@5"], 1.0)
        self.assertEqual(result["primary"], 0.75)


if __name__ == "__main__":
    unittest.main()
