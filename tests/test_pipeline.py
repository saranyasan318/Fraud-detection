import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import config
import fraud_pipeline as fp

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "data" / "smoke_fixture.csv"


def raw_frame(**overrides):
    """The fixture as raw text, optionally with columns replaced."""
    df = fp.load_csv(FIXTURE)
    for col, values in overrides.items():
        df[col] = values
    return df


def write_csv(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(text)
    f.close()
    return f.name


class SchemaTests(unittest.TestCase):
    def test_fixture_loads_with_expected_shape(self):
        df = fp.validate_schema(fp.load_csv(FIXTURE))
        self.assertEqual(len(df), 12)
        self.assertEqual(int(df[config.LABEL_COLUMN].sum()), 5)
        self.assertEqual(df[config.ID_COLUMN].nunique(), 12)

    def test_duplicate_ids_rejected(self):
        df = raw_frame()
        df.loc[1, "id"] = "T1"
        with self.assertRaisesRegex(fp.SchemaError, "duplicate ids"):
            fp.validate_schema(df)

    def test_missing_id_rejected(self):
        df = raw_frame()
        df.loc[0, "id"] = None
        with self.assertRaisesRegex(fp.SchemaError, "missing id"):
            fp.validate_schema(df)

    def test_non_binary_or_missing_labels_rejected(self):
        for bad in ("2", "-1", "yes", "0.5", None):
            df = raw_frame()
            df.loc[0, "is_fraud"] = bad
            with self.assertRaisesRegex(fp.SchemaError, "label", msg=repr(bad)):
                fp.validate_schema(df)

    def test_all_problems_reported_together(self):
        df = raw_frame()
        df.loc[1, "id"] = "T1"
        df.loc[2, "is_fraud"] = "9"
        with self.assertRaises(fp.SchemaError) as ctx:
            fp.validate_schema(df)
        self.assertIn("duplicate ids", str(ctx.exception))
        self.assertIn("label", str(ctx.exception))

    def test_missing_column_empty_file_and_no_rows(self):
        with self.assertRaisesRegex(fp.SchemaError, "missing required columns"):
            fp.validate_schema(raw_frame().drop(columns=["amount"]))
        with self.assertRaisesRegex(fp.SchemaError, "empty"):
            fp.load_csv(write_csv(""))
        with self.assertRaisesRegex(fp.SchemaError, "no data rows"):
            fp.validate_schema(fp.load_csv(write_csv("id,amount,account_age_days,failed_logins,is_fraud\n")))

    def test_extra_columns_are_ignored(self):
        df = fp.validate_schema(raw_frame(chargeback_flag="1"))
        self.assertEqual(list(df.columns), config.REQUIRED_COLUMNS)

    def test_whitespace_and_header_case_tolerated(self):
        path = write_csv(" ID , Amount ,account_age_days,failed_logins,is_fraud\n T1 , 10 ,1,0,0\n T2 ,20,2,1,1\n")
        df = fp.validate_schema(fp.load_csv(path))
        self.assertEqual(df["id"].tolist(), ["T1", "T2"])


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.raw = fp.validate_schema(raw_frame())

    def test_invalid_values_become_nan_and_no_rows_are_dropped(self):
        d = self.raw.copy()
        d.loc[0, "amount"] = "-5"                 # negative
        d.loc[1, "amount"] = "abc"                # non-numeric
        d.loc[2, "amount"] = "inf"                # infinite
        d.loc[3, "failed_logins"] = "1.5"         # must be a whole count
        d.loc[4, "account_age_days"] = "99999"    # beyond plausible maximum
        d.loc[5, "account_age_days"] = None       # genuinely missing
        X, rep = fp.clean_features(d)
        self.assertEqual(len(X), len(d))
        self.assertEqual(X["amount"].isna().sum(), 3)
        self.assertTrue(np.isnan(X.loc[3, "failed_logins"]))
        self.assertTrue(np.isnan(X.loc[4, "account_age_days"]))
        self.assertEqual(rep["amount"]["non_numeric"], 1)
        self.assertEqual(rep["amount"]["infinite"], 1)
        self.assertEqual(rep["amount"]["out_of_range"], 1)
        self.assertEqual(rep["failed_logins"]["non_integer"], 1)
        self.assertEqual(rep["account_age_days"]["originally_missing"], 1)
        self.assertEqual(rep["account_age_days"]["missing_after_cleaning"], 2)

    def test_clean_values_pass_through_unchanged(self):
        X, rep = fp.clean_features(self.raw)
        self.assertEqual(X["amount"].tolist(), [9000, 120, 7800, 90, 650, 5200, 50, 150, 6200, 2500, 330, 4000])
        self.assertTrue(all(v["missing_after_cleaning"] == 0 for v in rep.values()))

    def test_cleaning_cannot_depend_on_labels(self):
        shuffled = self.raw.copy()
        shuffled["is_fraud"] = shuffled["is_fraud"].sample(frac=1, random_state=1).to_numpy()
        pd.testing.assert_frame_equal(fp.clean_features(self.raw)[0], fp.clean_features(shuffled)[0])
        self.assertNotIn(config.LABEL_COLUMN, fp.clean_features(self.raw)[0].columns)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids, cls.X, cls.y, _ = fp.prepare(FIXTURE)

    def test_scores_are_probabilities(self):
        s = fp.score(fp.fit_pipeline(self.X, self.y), self.X)
        self.assertEqual(s.shape, (12,))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_reproducible_across_fits_and_across_row_order(self):
        a = fp.score(fp.fit_pipeline(self.X, self.y), self.X)
        b = fp.score(fp.fit_pipeline(self.X, self.y), self.X)
        np.testing.assert_array_equal(a, b)
        perm = np.random.default_rng(0).permutation(12)
        c = fp.score(fp.fit_pipeline(self.X.iloc[perm], self.y.iloc[perm]), self.X)
        np.testing.assert_allclose(a, c, atol=1e-6)

    def test_imputer_and_scaler_learn_from_training_rows_only(self):
        train, test = self.X.iloc[:8].copy(), self.X.iloc[8:].copy()
        train.loc[train.index[0], "amount"] = np.nan             # something to impute
        pipe = fp.fit_pipeline(train, self.y.iloc[:8])
        skewed = pipe.named_steps["pre"].named_transformers_["skewed"]
        expected_median = np.nanmedian(np.log1p(train["amount"]))
        self.assertAlmostEqual(skewed.named_steps["impute"].statistics_[0], expected_median)

        # Changing held-out rows (even absurdly) must not change anything learned at fit time.
        test["amount"] = 1e9
        again = fp.fit_pipeline(train, self.y.iloc[:8])
        np.testing.assert_array_equal(
            again.named_steps["pre"].named_transformers_["skewed"].named_steps["scale"].mean_,
            skewed.named_steps["scale"].mean_)

    def test_missing_values_at_prediction_time_are_imputed(self):
        pipe = fp.fit_pipeline(self.X, self.y)
        X_new = self.X.copy()
        X_new.iloc[0] = np.nan                                    # an entirely unknown transaction
        self.assertTrue(np.isfinite(fp.score(pipe, X_new)).all())

    def test_negative_amount_cannot_crash_the_log_transform(self):
        pipe = fp.fit_pipeline(self.X, self.y)
        X_new = self.X.head(2).copy()
        X_new.loc[X_new.index[0], "amount"] = -50
        self.assertTrue(np.isfinite(fp.score(pipe, X_new)).all())

    def test_training_requires_both_classes(self):
        with self.assertRaises(ValueError):
            fp.fit_pipeline(self.X, np.zeros(12, dtype=int))

    def test_smoke_test_reports_determinism_and_counts(self):
        r = fp.smoke_test(FIXTURE)
        self.assertTrue(r["deterministic"])
        self.assertEqual((r["rows"], r["frauds"], r["legitimate"]), (12, 5, 7))


if __name__ == "__main__":
    unittest.main()
