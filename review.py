
from __future__ import annotations

import math
import operator
from typing import Hashable, Iterable, Sequence

import numpy as np


def select_for_review(scores: Sequence[float], ids: Iterable[Hashable], budget: int) -> list:

    if isinstance(budget, bool):
        raise TypeError("budget must be an integer, not bool")
    try:
        budget = operator.index(budget)
    except TypeError:
        raise TypeError(f"budget must be an integer, got {type(budget).__name__}") from None
    if budget < 0:
        raise ValueError("budget must be >= 0")

    try:
        score_arr = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("scores must be numeric") from exc
    id_list = list(ids)
    if score_arr.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if len(score_arr) != len(id_list):
        raise ValueError(f"scores ({len(score_arr)}) and ids ({len(id_list)}) differ in length")
    if not np.isfinite(score_arr).all():
        raise ValueError("scores must be finite (found NaN or inf)")
    if any(i is None or (isinstance(i, float) and math.isnan(i)) for i in id_list):
        raise ValueError("ids must not be missing")

    if budget == 0:
        return []

    best: dict = {}                              # id -> highest score seen
    for tid, s in zip(id_list, score_arr.tolist()):
        if tid not in best or s > best[tid]:
            best[tid] = s

    try:
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    except TypeError:                            # mixed, non-comparable ID types
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], type(kv[0]).__name__, str(kv[0])))
    return [tid for tid, _ in ranked[:budget]]
