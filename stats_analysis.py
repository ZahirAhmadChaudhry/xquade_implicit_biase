"""Lightweight statistical helpers for paired comparisons between raw and tagged prompts."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


def _clean_pairs(pairs: Iterable[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    cleaned: List[Tuple[float, float]] = []
    for raw_value, tagged_value in pairs:
        raw_float = float(raw_value)
        tagged_float = float(tagged_value)
        if math.isnan(raw_float) or math.isnan(tagged_float):
            continue
        cleaned.append((raw_float, tagged_float))
    if not cleaned:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    raw_array = np.array([item[0] for item in cleaned], dtype=np.float64)
    tagged_array = np.array([item[1] for item in cleaned], dtype=np.float64)
    return raw_array, tagged_array


def paired_summary(pairs: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    raw_array, tagged_array = _clean_pairs(pairs)
    if raw_array.size == 0:
        return {
            "count": 0,
            "mean_raw": float("nan"),
            "mean_tagged": float("nan"),
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "cohens_d": float("nan"),
            "ci_low_95": float("nan"),
            "ci_high_95": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_value": float("nan"),
        }

    diff = tagged_array - raw_array
    count = diff.size
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if count > 1 else float("nan")

    try:
        t_statistic, p_value = stats.ttest_rel(tagged_array, raw_array)
    except Exception:
        t_statistic, p_value = float("nan"), float("nan")

    try:
        std_diff_for_d = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff_for_d if std_diff_for_d > 0 else float("nan")
    except Exception:
        cohens_d = float("nan")

    try:
        ci_low, ci_high = stats.t.interval(0.95, count - 1, loc=np.mean(diff), scale=stats.sem(diff)) if count > 1 else (float("nan"), float("nan"))
    except Exception:
        ci_low, ci_high = float("nan"), float("nan")

    try:
        if count >= 5 and not np.allclose(diff, 0.0):
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff)
        else:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    except Exception:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    return {
        "count": count,
        "mean_raw": float(np.mean(raw_array)),
        "mean_tagged": float(np.mean(tagged_array)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d) if not math.isnan(cohens_d) else float("nan"),
        "ci_low_95": float(ci_low),
        "ci_high_95": float(ci_high),
        "wilcoxon_statistic": float(wilcoxon_stat),
        "wilcoxon_p_value": float(wilcoxon_p),
    }
