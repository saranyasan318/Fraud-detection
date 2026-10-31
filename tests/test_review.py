import itertools
import random
import unittest

import numpy as np
import pandas as pd

from review import select_for_review


class SelectForReviewTests(unittest.TestCase):
    # ---- budget edge cases -------------------------------------------------------
    def test_budget_zero_returns_empty(self):
        self.assertEqual(select_for_review([0.9, 0.1], ["A", "B"], 0), [])
        self.assertEqual(select_for_review([], [], 0), [])

    def test_budget_larger_than_batch_returns_all_unique_ids_ranked(self):
        out = select_for_review([0.2, 0.9, 0.5], ["A", "B", "C"], 100)
        self.assertEqual(out, ["B", "C", "A"])

    def test_budget_equal_to_batch_size(self):
        self.assertEqual(select_for_review([0.2, 0.9], ["A", "B"], 2), ["B", "A"])

    def test_empty_batch_with_positive_budget(self):
        self.assertEqual(select_for_review([], [], 5), [])

    def test_negative_or_non_integer_budget_rejected(self):
        with self.assertRaises(ValueError):
            select_for_review([0.5], ["A"], -1)
        for bad in (1.5, "3", None, True):
            with self.assertRaises(TypeError, msg=repr(bad)):
                select_for_review([0.5], ["A"], bad)

    def test_numpy_integer_budget_accepted(self):
        self.assertEqual(select_for_review([0.5, 0.6], ["A", "B"], np.int64(1)), ["B"])

    # ---- ordering and ties -------------------------------------------------------
    def test_descending_score_order(self):
        self.assertEqual(select_for_review([0.1, 0.8, 0.5, 0.95], list("ABCD"), 3), ["D", "B", "C"])

    def test_ties_broken_by_ascending_id(self):
        out = select_for_review([0.9, 0.9, 0.9, 0.1], ["T3", "T1", "T2", "T4"], 2)
        self.assertEqual(out, ["T1", "T2"])

    def test_result_independent_of_input_order(self):
        rows = [("T3", 0.9), ("T1", 0.9), ("T2", 0.9), ("T4", 0.5), ("T0", 0.5)]
        expected = ["T1", "T2", "T3", "T0"]
        for perm in itertools.permutations(rows):
            ids, scores = zip(*perm)
            self.assertEqual(select_for_review(scores, ids, 4), expected)

    def test_tie_at_the_budget_boundary_is_resolved_by_id(self):
        # three-way tie, only one slot left after the clear winner
        out = select_for_review([0.99, 0.5, 0.5, 0.5], ["Z", "C", "A", "B"], 2)
        self.assertEqual(out, ["Z", "A"])

    def test_numeric_ids_use_natural_order_not_text_order(self):
        self.assertEqual(select_for_review([0.5, 0.5, 0.5], [10, 9, 2], 3), [2, 9, 10])

    def test_mixed_id_types_are_still_deterministic(self):
        a = select_for_review([0.5, 0.5, 0.5], [10, "x", 2], 3)
        b = select_for_review([0.5, 0.5, 0.5], ["x", 2, 10], 3)
        self.assertEqual(a, b)

    def test_matches_reference_sort_on_random_batch(self):
        rnd = random.Random(0)
        ids = [f"T{i:05d}" for i in range(5000)]
        scores = [round(rnd.random(), 2) for _ in ids]          # rounding forces many ties
        expected = [i for i, _ in sorted(zip(ids, scores), key=lambda p: (-p[1], p[0]))][:250]
        self.assertEqual(select_for_review(scores, ids, 250), expected)

    # ---- uniqueness --------------------------------------------------------------
    def test_duplicate_ids_collapse_to_highest_score(self):
        out = select_for_review([0.2, 0.5, 0.9], ["A", "B", "A"], 2)
        self.assertEqual(out, ["A", "B"])
        self.assertEqual(select_for_review([0.2, 0.5, 0.9], ["A", "B", "A"], 1), ["A"])

    def test_output_is_unique_and_never_exceeds_budget(self):
        ids = ["A", "A", "A", "B", "B"]
        out = select_for_review([0.1, 0.2, 0.3, 0.4, 0.5], ids, 4)
        self.assertEqual(out, ["B", "A"])
        self.assertEqual(len(out), len(set(out)))
        self.assertLessEqual(len(out), 4)

    # ---- validation --------------------------------------------------------------
    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            select_for_review([0.1, 0.2], ["A"], 1)                    # length mismatch
        with self.assertRaises(ValueError):
            select_for_review([0.1, float("nan")], ["A", "B"], 1)      # NaN score
        with self.assertRaises(ValueError):
            select_for_review([0.1, float("inf")], ["A", "B"], 1)      # inf score
        with self.assertRaises(ValueError):
            select_for_review([0.1, 0.2], ["A", None], 1)              # missing id
        with self.assertRaises(ValueError):
            select_for_review([[0.1], [0.2]], ["A", "B"], 1)           # not 1-D
        with self.assertRaises(ValueError):
            select_for_review(["high", "low"], ["A", "B"], 1)          # non-numeric

    def test_nan_rejected_even_when_budget_is_zero(self):
        with self.assertRaises(ValueError):
            select_for_review([float("nan")], ["A"], 0)

    # ---- input handling ----------------------------------------------------------
    def test_inputs_not_mutated(self):
        scores, ids = [0.3, 0.9, 0.3], ["C", "B", "A"]
        select_for_review(scores, ids, 2)
        self.assertEqual((scores, ids), ([0.3, 0.9, 0.3], ["C", "B", "A"]))

    def test_accepts_numpy_and_pandas_inputs(self):
        s = pd.Series([0.3, 0.9, 0.3], index=[7, 8, 9])
        ids = pd.Series(["C", "B", "A"], index=[7, 8, 9])
        self.assertEqual(select_for_review(s, ids, 3), ["B", "A", "C"])
        self.assertEqual(select_for_review(s.to_numpy(), ids.to_numpy(), 1), ["B"])


if __name__ == "__main__":
    unittest.main()
