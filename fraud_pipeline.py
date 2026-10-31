"""Load, schema-validate, clean and model transaction data (Q1 b).

Design rules that prevent label leakage
---------------------------------------
* clean_features() receives feature columns only and never sees the label, so it cannot
  use it. It only blanks impossible values (set to NaN); it fills nothing.
* Every statistic that is *learned* (log-feature handling is stateless, but the median
  imputer and the scaler are not) lives inside the sklearn Pipeline, so it is fitted on
  whatever training rows are passed to .fit() and nothing else.
* Rows are never dropped because of their feature values or their label outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import config


class SchemaError(ValueError):
    """Raised with every schema problem found, not only the first."""


def load_csv(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError("file is empty") from exc
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        df[col] = df[col].str.strip().mask(lambda s: s == "")   # blank cell -> missing
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    if df.columns.duplicated().any():
        raise SchemaError(f"duplicate column names: {df.columns[df.columns.duplicated()].tolist()}")
    if df.empty:
        raise SchemaError("no data rows")

    errors: list[str] = []
    ids = df[config.ID_COLUMN]
    blank = ids.isna()
    if blank.any():
        errors.append(f"{int(blank.sum())} row(s) have a missing id")
    dup = ids[ids.notna() & ids.duplicated(keep=False)]
    if not dup.empty:
        errors.append(f"duplicate ids: {sorted(dup.unique())[:5]}"
                      f"{' ...' if dup.nunique() > 5 else ''}")

    label = pd.to_numeric(df[config.LABEL_COLUMN], errors="coerce")
    bad = label.isna() | ~label.isin([0, 1])
    if bad.any():
        examples = df.loc[bad, config.LABEL_COLUMN].fillna("<missing>").unique().tolist()[:5]
        errors.append(f"{int(bad.sum())} row(s) have a label outside {{0, 1}} (e.g. {examples})")

    if errors:
        raise SchemaError("; ".join(errors))

    out = df[config.REQUIRED_COLUMNS].copy()
    out[config.ID_COLUMN] = out[config.ID_COLUMN].astype(str)
    out[config.LABEL_COLUMN] = label.astype(int)
    return out.reset_index(drop=True)


def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    
    X = pd.DataFrame(index=df.index)
    report: dict[str, dict[str, int]] = {}
    for col in config.FEATURES:
        rule = config.FEATURE_RULES[col]
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce").astype(float)

        originally_missing = raw.isna()
        non_numeric = num.isna() & raw.notna()
        non_finite = np.isinf(num)
        num = num.mask(non_finite)

        out_of_range = pd.Series(False, index=num.index)
        if rule["min"] is not None:
            out_of_range |= num < rule["min"]
        if rule["max"] is not None:
            out_of_range |= num > rule["max"]
        non_integer = (num % 1 != 0) & num.notna() if rule["integer"] else pd.Series(False, index=num.index)

        X[col] = num.mask(out_of_range | non_integer)
        report[col] = {
            "originally_missing": int(originally_missing.sum()),
            "non_numeric": int(non_numeric.sum()),
            "infinite": int(non_finite.sum()),
            "out_of_range": int(out_of_range.sum()),
            "non_integer": int(non_integer.sum()),
            "missing_after_cleaning": int(X[col].isna().sum()),
        }
    return X, report


def prepare(path):
   
    df = validate_schema(load_csv(path))
    X, report = clean_features(df)
    return df[config.ID_COLUMN], X, df[config.LABEL_COLUMN], report



def _log1p_nonneg(a):
    """log1p with negatives clipped to 0 (NaN passes through)."""
    return np.log1p(np.clip(a, 0, None))


def build_pipeline(seed: int = config.SEED) -> Pipeline:
    """Unfitted preprocessing + classifier. Fitting it learns medians/scales from train rows only."""
    def numeric(steps):
        return Pipeline([*steps,
                         ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                  keep_empty_features=True)),
                         ("scale", StandardScaler())])

    pre = ColumnTransformer(
        [("skewed", numeric([("log", FunctionTransformer(_log1p_nonneg))]), config.LOG_FEATURES),
         ("linear", numeric([]), config.LINEAR_FEATURES)],
        remainder="drop",
    )
    clf = LogisticRegression(C=config.LOGREG_C, max_iter=config.LOGREG_MAX_ITER,
                             class_weight="balanced", solver="lbfgs", random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_pipeline(X: pd.DataFrame, y, seed: int = config.SEED) -> Pipeline:
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("training data must contain both classes (0 and 1)")
    return build_pipeline(seed).fit(X[config.FEATURES], y)


def score(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Fraud score in [0, 1]. Feed it cleaned features (output of clean_features)."""
    return pipe.predict_proba(X[config.FEATURES])[:, 1]


def smoke_test(path) -> dict:
    """Prove the pipeline runs end to end and is reproducible. NOT a performance estimate."""
    ids, X, y, report = prepare(path)
    first, second = score(fit_pipeline(X, y), X), score(fit_pipeline(X, y), X)
    assert first.shape == (len(X),) and np.isfinite(first).all()
    assert ((first >= 0) & (first <= 1)).all()
    return {
        "rows": len(X), "frauds": int(y.sum()), "legitimate": int((y == 0).sum()),
        "cleaning_report": report,
        "deterministic": bool(np.array_equal(first, second)),
        "scores": pd.Series(first, index=ids, name="score"),
    }
