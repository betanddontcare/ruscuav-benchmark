"""Episode tables: CSV output and summary statistics with the semantics of pandas.

The archived evaluations wrote their per-episode rows with
``pandas.DataFrame(rows).to_csv(index=False)`` and computed their summaries with
pandas (with bottleneck installed). These helpers give the same files and the same
numbers without pandas, so the package can reproduce the archived outputs exactly.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np

Row = Dict[str, Any]

__all__ = [
    "Row", "write_rows_csv", "jsonable", "column", "is_missing", "is_true", "pandas_mean", "pandas_std", "pandas_median",
    "pandas_quantile", "pandas_max", "mean_of", "std_of", "median_of", "termination_counts",
]


def is_missing(value: Any) -> bool:
    """None or NaN (an empty cell in pandas)."""
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def is_true(value: Any) -> bool:
    """``value == True`` as pandas compares a cell (``df[df[col] == True]``): True for
    True, ``numpy.bool_(True)`` and 1, False for strings, None and NaN."""
    if value is None or isinstance(value, str):
        return False
    try:
        return bool(value == True)  # noqa: E712 - the comparison is the point
    except (TypeError, ValueError):
        return False


def _column_kind(values: List[Any]) -> str:
    """The dtype pandas would give a column of these values: bool, int, float or object."""
    present = [v for v in values if not is_missing(v)]
    if present and all(isinstance(v, (bool, np.bool_)) for v in present):
        return "bool" if len(present) == len(values) else "object"
    numbers = [v for v in present if isinstance(v, (int, float, np.integer, np.floating))
               and not isinstance(v, (bool, np.bool_))]
    if len(numbers) == len(present):
        if len(present) == len(values) and all(isinstance(v, (int, np.integer)) for v in present):
            return "int"
        return "float"
    return "object"


def _csv_value(value: Any, kind: str) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if kind == "float" or isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def write_rows_csv(path: Union[str, Path], rows: List[Row]) -> Path:
    """Write episode rows as ``pandas.DataFrame(rows).to_csv(index=False)`` would:
    columns in order of first appearance, empty cells for missing values and NaN,
    integer columns with gaps written as floats."""
    columns: List[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    kinds = {key: _column_kind([row.get(key) for row in rows]) for key in columns}
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_value(row.get(key), kinds[key]) for key in columns])
    return path


def jsonable(obj: Any) -> Any:
    """Plain JSON types; NaN and infinities become None."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    return obj


def column(rows: Iterable[Row], key: str) -> np.ndarray:
    """Values of one key as float64 (missing values and None are NaN; True is 1)."""
    values = []
    for row in rows:
        value = row.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def pandas_mean(values: np.ndarray) -> float:
    """``pandas.Series.mean``: NaN set to 0 for the sum, divided by the count of values."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.isnan(values)
    count = int((~mask).sum())
    if count == 0:
        return float("nan")
    return float(np.where(mask, 0.0, values).sum() / count)


def pandas_std(values: np.ndarray) -> float:
    """``pandas.Series.std(ddof=0)`` as computed with bottleneck (the archived
    summaries): a sequential two-pass sum over the values that are not NaN."""
    present = [v for v in np.asarray(values, dtype=np.float64).tolist() if not math.isnan(v)]
    if not present:
        return float("nan")
    total = 0.0
    for v in present:
        total += v
    mean = total / len(present)
    squares = 0.0
    for v in present:
        squares += (v - mean) * (v - mean)
    return math.sqrt(squares / len(present))


def _present(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[~np.isnan(values)]


def pandas_median(values: np.ndarray) -> float:
    present = _present(values)
    return float(np.median(present)) if present.size else float("nan")


def pandas_quantile(values: np.ndarray, q: float) -> float:
    """``pandas.Series.quantile(q)``: linear interpolation over the values that are not NaN."""
    present = _present(values)
    return float(np.percentile(present, 100.0 * q)) if present.size else float("nan")


def pandas_max(values: np.ndarray) -> float:
    present = _present(values)
    return float(present.max()) if present.size else float("nan")


def mean_of(rows: List[Row], key: str) -> float:
    return pandas_mean(column(rows, key))


def std_of(rows: List[Row], key: str) -> float:
    return pandas_std(column(rows, key))


def median_of(rows: List[Row], key: str) -> float:
    return pandas_median(column(rows, key))


def termination_counts(rows: List[Row]) -> Dict[str, int]:
    """``rows["termination"].fillna("None").value_counts()`` as a dict."""
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get("termination")
        key = "None" if is_missing(value) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))
