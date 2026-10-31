import math
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from metrics import all_legitimate_baseline, classification_metrics, tn_from_totals


class HeldOutBatchTests(unittest.TestCase):
    def test_tn_and_metrics_for_the_assignment_batch(self):
        tn = tn_from_totals(10_000, tp=80, fn=20, fp=320)
        self.assertEqual(tn, 9580)
        m = classification_metrics(tp=80, fp=320, fn=20, tn=tn)
        self.assertAlmostEqual(m["precision"], 0.20)
        self.assertAlmostEqual(m["recall"], 0.80)
        self.assertAlmostEqual(m["f1"], 0.32)
        self.assertAlmostEqual(m["accuracy"], 0.966)
        self.assertEqual(m["total"], 10_000)

    def test_agrees_with_scikit_learn(self):
        y_true = np.array([1] * 80 + [1] * 20 + [0] * 320 + [0] * 9580)
        y_pred = np.array([1] * 80 + [0] * 20 + [1] * 320 + [0] * 9580)
        m = classification_metrics(80, 320, 20, 9580)
        self.assertAlmostEqual(m["precision"], precision_score(y_true, y_pred))
        self.assertAlmostEqual(m["recall"], recall_score(y_true, y_pred))
        self.assertAlmostEqual(m["f1"], f1_score(y_true, y_pred))
        self.assertAlmostEqual(m["accuracy"], accuracy_score(y_true, y_pred))

    def test_all_legitimate_baseline(self):
        b = all_legitimate_baseline(10_000, positives=100)
        self.assertEqual((b["tp"], b["fp"], b["fn"], b["tn"]), (0, 0, 100, 9900))
        self.assertAlmostEqual(b["accuracy"], 0.99)
        self.assertEqual(b["recall"], 0.0)
        self.assertEqual(b["f1"], 0.0)
        self.assertTrue(math.isnan(b["precision"]))      # no alerts -> precision undefined

    def test_baseline_has_higher_accuracy_but_catches_no_fraud(self):
        model = classification_metrics(80, 320, 20, 9580)
        base = all_legitimate_baseline(10_000, 100)
        self.assertGreater(base["accuracy"], model["accuracy"])
        self.assertGreater(model["recall"], base["recall"])
        self.assertGreater(model["f1"], base["f1"])


class GuardTests(unittest.TestCase):
    def test_impossible_totals_rejected(self):
        with self.assertRaises(ValueError):
            tn_from_totals(100, tp=80, fn=20, fp=320)

    def test_negative_non_integer_and_bool_counts_rejected(self):
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                classification_metrics(bad, 0, 0, 10)

    def test_undefined_ratios_are_nan_not_zero(self):
        m = classification_metrics(0, 0, 0, 10)          # no fraud at all, no alerts
        self.assertTrue(math.isnan(m["precision"]) and math.isnan(m["recall"]) and math.isnan(m["f1"]))
        self.assertEqual(m["accuracy"], 1.0)

    def test_empty_matrix_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics(0, 0, 0, 0)

    def test_baseline_positives_cannot_exceed_total(self):
        with self.assertRaises(ValueError):
            all_legitimate_baseline(10, 11)


if __name__ == "__main__":
    unittest.main()
