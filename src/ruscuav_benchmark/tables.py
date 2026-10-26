"""Episode tables: CSV input and output and summary statistics with the semantics of pandas.

The archived evaluations wrote their per-episode rows with
``pandas.DataFrame(rows).to_csv(index=False)``, computed their summaries with
pandas (with bottleneck installed) and built the paper's tables from the CSVs read
back with ``pandas.read_csv``. These helpers give the same files and the same
numbers without pandas, so the package can reproduce the archived outputs exactly.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

Row = Dict[str, Any]

__all__ = [
    "Row", "write_rows_csv", "read_rows_csv", "parse_float", "NA_STRINGS", "jsonable", "column", "is_missing",
    "is_true", "pandas_mean", "pandas_std", "pandas_median", "pandas_quantile", "pandas_max", "mean_of", "std_of",
    "median_of", "termination_counts",
]

#: Cells ``pandas.read_csv`` reads as missing by default (its ``na_values``).
NA_STRINGS = frozenset([
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
    "NULL", "NaN", "None", "n/a", "nan", "null",
])
_POWERS_OF_TEN = tuple(float(f"1e{k}") for k in range(309))
_SPACES = " \t\n\v\f\r"
_FLOAT_CELL = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?\Z")
#: The part of a cell the round-trip parser converts: sign, digits, point, digits, exponent.
_NUMBER_PREFIX = re.compile(r"[+-]?[0-9]*(\.[0-9]*)?([eE][+-]?[0-9]*)?")
#: A line the C parser skips as blank: spaces and tabs only.
_BLANK_LINE = re.compile(r"[ \t]+\Z")
_INF_CELLS = {"inf": math.inf, "+inf": math.inf, "infinity": math.inf, "+infinity": math.inf,
              "-inf": -math.inf, "-infinity": -math.inf}
_INT64 = (-(2 ** 63), 2 ** 63 - 1)


def parse_float(text: str, exact: bool = False) -> Optional[float]:
    """The number ``pandas.read_csv`` reads from a cell, or None if it reads none.

    By default this is the C parser's ``precise_xstrtod`` (pandas 2.2), which sums at
    most 17 significant digits in floating point and then scales by a power of ten:
    for about one cell in five of the archived CSVs it differs from ``float(text)``
    in the last bit. With ``exact`` the result is the correctly rounded ``float(text)``,
    as with ``read_csv(float_precision="round_trip")``. Infinities are read as pandas
    reads them.
    """
    special = _INF_CELLS.get(text.lower())
    if special is not None:
        return special
    if exact:
        return _round_trip(text)
    s, n, i = text, len(text), 0
    while i < n and s[i] in _SPACES:
        i += 1
    negative = False
    if i < n and s[i] in "+-":
        negative = s[i] == "-"
        i += 1
    number, exponent, digits, decimals = 0.0, 0, 0, 0
    while i < n and "0" <= s[i] <= "9":
        if digits < 17:
            number = number * 10.0 + (ord(s[i]) - 48)
            digits += 1
        else:
            exponent += 1
        i += 1
    if i < n and s[i] == ".":
        i += 1
        while digits < 17 and i < n and "0" <= s[i] <= "9":
            number = number * 10.0 + (ord(s[i]) - 48)
            i += 1
            digits += 1
            decimals += 1
        if digits >= 17:
            while i < n and "0" <= s[i] <= "9":
                i += 1
        exponent -= decimals
    if digits == 0:
        return None
    if negative:
        number = -number
    if i < n and s[i] in "eE":
        j = i + 1
        exp_negative = False
        if j < n and s[j] in "+-":
            exp_negative = s[j] == "-"
            j += 1
        exp_digits, e = 0, 0
        while exp_digits < 17 and j < n and "0" <= s[j] <= "9":
            e = e * 10 + (ord(s[j]) - 48)
            exp_digits += 1
            j += 1
        if exp_digits:
            exponent = exponent - e if exp_negative else exponent + e
            i = j
    if exponent > 308:
        return None
    if exponent > 0:
        number *= _POWERS_OF_TEN[exponent]
    elif exponent < -308:
        number = 0.0 if exponent < -616 else number / _POWERS_OF_TEN[-308 - exponent] / _POWERS_OF_TEN[308]
    else:
        number /= _POWERS_OF_TEN[-exponent]
    if math.isinf(number):
        return None
    while i < n and s[i] in _SPACES:
        i += 1
    return number if i == n else None


def _round_trip(text: str) -> Optional[float]:
    """The C parser's ``round_trip``: after leading white space, the longest prefix of
    sign, digits, point, digits and exponent must be a number (correctly rounded by
    Python); white space may follow. An overflow to +infinity is an error, one to
    -infinity is not (``r == Py_HUGE_VAL`` is the only check)."""
    n, i = len(text), 0
    while i < n and text[i] in _SPACES:
        i += 1
    prefix = _NUMBER_PREFIX.match(text, i)
    j = prefix.end()
    if not _FLOAT_CELL.match(text[i:j]):
        return None
    value = float(text[i:j])
    if value == math.inf:
        return None
    while j < n and text[j] in _SPACES:
        j += 1
    return value if j == n else None


def _integer_prefix(cell: str):
    """The C parser's integer scan of a cell: (negative, value of the leading digits,
    nothing else follows) or None without digits."""
    n, i = len(cell), 0
    while i < n and cell[i] in _SPACES:
        i += 1
    negative = False
    if i < n and cell[i] in "+-":
        negative = cell[i] == "-"
        i += 1
    j = i
    while j < n and "0" <= cell[j] <= "9":
        j += 1
    if j == i:
        return None
    k = j
    while k < n and cell[k] in _SPACES:
        k += 1
    return negative, int(cell[i:j]), k == n


def _int64_status(cells: List[str], missing: List[bool]) -> str:
    """``_try_int64``: "ok", or how its scan in row order ends: "overflow" (the digits
    of a cell exceed int64) or "invalid"."""
    for cell, gap in zip(cells, missing):
        if gap:
            continue
        scan = _integer_prefix(cell)
        if scan is None:
            return "invalid"
        negative, digits, whole = scan
        value = -digits if negative else digits
        if not _INT64[0] <= value <= _INT64[1]:
            return "overflow"
        if not whole:
            return "invalid"
    return "ok"


def _uint64_status(cells: List[str], missing: List[bool]) -> str:
    """``_try_uint64`` after an int64 overflow: "ok", "invalid" (go on to floats),
    "object" (an overflow or a negative value: pandas keeps the strings) or "raw"
    (values above int64 together with negative or missing ones: pandas keeps the
    strings and does not read missing cells as missing)."""
    seen_signed = seen_unsigned = seen_null = False
    for cell, gap in zip(cells, missing):
        if gap:
            seen_null = True
            continue
        stripped = cell.lstrip(_SPACES)
        if stripped.startswith("-"):
            seen_signed = True
            continue
        scan = _integer_prefix(stripped)
        if scan is None:
            return "invalid"
        _, digits, whole = scan
        if digits > 2 ** 64 - 1:
            return "object"
        if not whole:
            return "invalid"
        seen_unsigned = seen_unsigned or digits > _INT64[1]
    if seen_unsigned and (seen_signed or seen_null):
        return "raw"
    return "object" if seen_signed else "ok"


def _read_column(cells: List[str], exact: bool) -> List[Any]:
    """One column as ``pandas.read_csv`` (C engine) infers it, trying int64, float64,
    bool and object in turn."""
    missing = [cell in NA_STRINGS for cell in cells]
    strings = [None if gap else cell for cell, gap in zip(cells, missing)]
    status = _int64_status(cells, missing)
    if status == "ok":
        ints = [None if gap else int(cell) for cell, gap in zip(cells, missing)]
        if not any(missing):
            return ints
        # with gaps the column is float; pandas marks gaps with the smallest int64, so that value is a gap too
        return [None if v is None or v == _INT64[0] else float(v) for v in ints]
    if status == "overflow":
        unsigned = _uint64_status(cells, missing)
        if unsigned == "object":
            return strings
        if unsigned == "raw":
            return list(cells)
        if unsigned == "ok":
            return [int(cell) for cell in cells]
    present = [cell for cell, gap in zip(cells, missing) if not gap]
    numbers = [parse_float(cell, exact) for cell in present]
    if all(v is not None for v in numbers):
        values = iter(numbers)
        return [None if gap else next(values) for gap in missing]
    # booleans: "true" and "false" in any case (the C parser's strcasecmp)
    if all(cell.lower() in ("true", "false") for cell in present):
        return [None if gap else cell.lower() == "true" for cell, gap in zip(cells, missing)]
    return strings


class _Lines:
    """The lines of a file, keeping those ``csv.reader`` consumed since the last record."""

    def __init__(self, f: Iterable[str]) -> None:
        self._lines = iter(f)
        self.consumed: List[str] = []

    def __iter__(self) -> "_Lines":
        return self

    def __next__(self) -> str:
        line = next(self._lines)
        self.consumed.append(line)
        return line


def _records(f: Iterable[str]) -> List[List[str]]:
    """The records of a CSV file without the lines ``pandas.read_csv`` skips as blank:
    empty lines and lines of spaces and tabs only (not quoted)."""
    lines = _Lines(f)
    out = []
    for record in csv.reader(lines):
        raw = "".join(lines.consumed).rstrip("\r\n")
        lines.consumed.clear()
        if not record or (len(record) == 1 and record[0] == raw and _BLANK_LINE.match(raw)):
            continue
        out.append(record)
    return out


def read_rows_csv(path: Union[str, Path], exact_floats: bool = False) -> List[Row]:
    """Rows of a CSV as ``pandas.read_csv(path).to_dict("records")`` gives them, with
    Python values: int, float, bool or str, and None for missing cells (NaN in
    pandas). Columns are typed as pandas types them: integers, floats (an integer
    column with gaps is a float column), booleans, otherwise strings. Floats are read
    as pandas reads them by default (:func:`parse_float`); ``exact_floats`` reads them
    correctly rounded, as ``float_precision="round_trip"`` does. A UTF-8 byte-order
    mark is skipped, as are blank lines."""
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        lines = _records(f)
    if not lines:
        return []
    header, body = lines[0], lines[1:]
    if len(set(header)) < len(header):
        raise ValueError(f"{path}: repeated column names")
    for k, line in enumerate(body, start=2):
        if len(line) > len(header):
            raise ValueError(f"{path}, line {k}: {len(line)} fields, the header has {len(header)}")
    columns = [_read_column([line[j] if j < len(line) else "" for line in body], exact_floats)
               for j in range(len(header))]
    return [{name: columns[j][k] for j, name in enumerate(header)} for k in range(len(body))]


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
