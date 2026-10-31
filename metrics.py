
from __future__ import annotations

import math


def _count(name: str, value) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def tn_from_totals(n_total: int, tp: int, fn: int, fp: int) -> int:
    """TN = N - TP - FN - FP, rejecting impossible inputs."""
    n_total, tp, fn, fp = (_count(k, v) for k, v in
                           (("n_total", n_total), ("tp", tp), ("fn", fn), ("fp", fp)))
    tn = n_total - tp - fn - fp
    if tn < 0:
        raise ValueError(f"TP+FN+FP = {tp + fn + fp} exceeds the batch size {n_total}")
    return tn


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan      # NaN = undefined, never silently 0


def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
  
    tp, fp, fn, tn = (_count(k, v) for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)))
    total = tp + fp + fn + tn
    if total == 0:
        raise ValueError("empty confusion matrix")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": (tp + tn) / total,
        "alerts": tp + fp,
        "alert_rate": (tp + fp) / total,
        "prevalence": (tp + fn) / total,
    }


def all_legitimate_baseline(n_total: int, positives: int) -> dict:
    """A 'model' that flags nothing: every fraud is missed, every legitimate is correct."""
    n_total, positives = _count("n_total", n_total), _count("positives", positives)
    if positives > n_total:
        raise ValueError("positives cannot exceed n_total")
    return classification_metrics(tp=0, fp=0, fn=positives, tn=n_total - positives)


def fmt(x: float, digits: int = 4) -> str:
    return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:.{digits}f}"
