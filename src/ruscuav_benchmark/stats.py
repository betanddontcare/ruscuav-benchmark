"""The statistics of the paper's tables.

- :func:`training_seed_ci`: mean, standard deviation and Student-t 95 % interval over
  the training seeds of a learned method, the primary uncertainty of learned-policy
  results (the notebook's ``training_seed_ci``).
- :func:`mcnemar_exact_p` and :func:`holm_adjust`: the exact paired (McNemar) test of
  a binary outcome of two methods flown on the same missions, and Holm's step-down
  correction within a declared family of comparisons (``paired_exact_p`` and
  ``holm_adjust``).
- :func:`episode_bootstrap_ci`: a percentile bootstrap over the episodes of one run,
  the secondary (fixed-policy) uncertainty (``ms_bootstrap_ci``).

They use NumPy and SciPy as the notebook did. With the archive's versions (NumPy
2.0.2, SciPy 1.16.3) the training-seed intervals and the p-values of the paper's
tables are the archived ones bit for bit, and the episode bootstrap gives most of
the archived intervals bit for bit and the others to within a few 1e-16, relatively;
other versions may change the last digits (docs/REPRODUCIBILITY.md).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import numpy as np

__all__ = ["training_seed_ci", "mcnemar_exact_p", "holm_adjust", "episode_bootstrap_ci"]


def _numbers(values: Iterable[Any]) -> np.ndarray:
    """``pd.to_numeric(pd.Series(values), errors="coerce").dropna()`` as float64."""
    out = []
    for value in values:
        if value is None or isinstance(value, str) and not value.strip():
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            out.append(number)
    return np.asarray(out, dtype=np.float64)


def training_seed_ci(values: Iterable[Any]) -> Dict[str, float]:
    """n, mean, standard deviation (ddof 1) and the two-sided 95 % Student-t interval
    of the mean of per-seed values (missing values are dropped)."""
    from scipy.stats import t as student_t

    vals = _numbers(values)
    n = int(len(vals))
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mean": nan, "std": nan, "ci95_low": nan, "ci95_high": nan}
    mean = float(vals.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": nan, "ci95_low": nan, "ci95_high": nan}
    std = float(vals.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half}


def mcnemar_exact_p(n_ab: int, n_ba: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs: ``n_ab`` missions on
    which only method A had the outcome, ``n_ba`` on which only B had it (a binomial
    test of min(n_ab, n_ba) in n_ab + n_ba trials with p = 1/2; 1 without discordant
    pairs). Computed with SciPy's ``binomtest``, as in the archive."""
    from scipy.stats import binomtest

    n_ab, n_ba = int(n_ab), int(n_ba)
    if n_ab < 0 or n_ba < 0:
        raise ValueError("counts must not be negative")
    discordant = n_ab + n_ba
    if discordant == 0:
        return 1.0
    return float(binomtest(k=min(n_ab, n_ba), n=discordant, p=0.5, alternative="two-sided").pvalue)


def holm_adjust(raw_p: Iterable[float]) -> np.ndarray:
    """Holm's step-down adjusted p-values of one family, in the input order:
    the k-th smallest p-value times (m - k + 1), made non-decreasing, at most 1."""
    p = np.asarray(list(raw_p), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def episode_bootstrap_ci(values: Sequence[Any], stat: Callable[[np.ndarray], float] = np.mean,
                         n_boot: int = 2000, ci: float = 0.95, seed: int = 12345) -> Tuple[float, float, float]:
    """(estimate, low, high): ``stat`` of the values and the percentile interval of
    ``stat`` over ``n_boot`` resamples of the episodes with replacement (NumPy's
    ``default_rng(seed)``). Booleans count as 0 and 1; missing values are dropped."""
    arr = np.asarray(values)
    if arr.dtype.kind == "b":
        arr = arr.astype(float)
    else:
        arr = _numbers(arr.tolist())
    if len(arr) == 0:
        nan = float("nan")
        return nan, nan, nan
    rng = np.random.default_rng(int(seed))
    n = len(arr)
    boots = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(stat(arr[idx]))
    alpha = 1.0 - float(ci)
    return float(stat(arr)), float(np.quantile(boots, alpha / 2.0)), float(np.quantile(boots, 1.0 - alpha / 2.0))
