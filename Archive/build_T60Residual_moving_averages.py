#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: build_T60Residual_moving_averages.py
=============================================================================

INPUT FILES:
- T60Residual.xlsx (same folder as this script by default)
  - Sheet: Sheet1
  - Column "Date": monthly timestamps (first column)
  - Remaining columns: numeric factor residuals (wide panel)

OUTPUT FILES (same folder by default; one workbook per window):
- T60Residual_MA3.xlsx
- T60Residual_MA6.xlsx
- T60Residual_MA12.xlsx
- T60Residual_MA24.xlsx
- T60Residual_MA60.xlsx

Each output is the same shape as the input: Date plus all factor columns, with
each numeric column replaced by its trailing moving average of length W (mean).
For months 1 .. W-1 from the start of the series, the average uses all available
history from row 1 through the current month (expanding window). From month W
onward, the average uses exactly the last W months (standard rolling window).

VERSION: 1.2
LAST UPDATED: 2026-03-19
AUTHOR: Cursor Agent (T2 Factor Timing Fuzzy)

DESCRIPTION:
Takes the residual panel in T60Residual.xlsx and writes smoothed versions used
for analysis or downstream models, without changing calendar alignment or column names.

DEPENDENCIES:
- pandas
- xlsxwriter (ExcelWriter engine)

USAGE:
  python build_T60Residual_moving_averages.py
  python build_T60Residual_moving_averages.py --input /path/T60Residual.xlsx --out-dir /path
=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MA_WINDOWS = (3, 6, 12, 24, 60)


def apply_expanding_then_rolling_mean(
    df: pd.DataFrame, window: int, date_col: str = "Date"
) -> pd.DataFrame:
    """Return copy with numeric columns smoothed; Date unchanged; sorted by Date."""
    out = df.copy()
    if date_col not in out.columns:
        raise ValueError(f"Missing required column {date_col!r}")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.sort_values(date_col).reset_index(drop=True)
    num_cols = [c for c in out.columns if c != date_col]
    for col in num_cols:
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.rolling(window=window, min_periods=1).mean()
    return out


def write_xlsx_with_date_format(df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1") -> None:
    """Write DataFrame to xlsx; format Date column as dd-mmm-yyyy per project standards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        date_format = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        if "Date" not in df.columns:
            return
        date_col_idx = df.columns.get_loc("Date")
        worksheet.set_column(date_col_idx, date_col_idx, 15)
        for row in range(len(df)):
            val = df["Date"].iloc[row]
            if pd.notna(val):
                worksheet.write(row + 1, date_col_idx, val, date_format)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T60Residual moving-average workbooks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "T60Residual.xlsx",
        help="Path to T60Residual.xlsx",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as input)",
    )
    args = parser.parse_args()
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else inp.parent

    raw = pd.read_excel(inp, sheet_name="Sheet1")
    stem = inp.stem
    for w in MA_WINDOWS:
        smoothed = apply_expanding_then_rolling_mean(raw, window=w)
        out_path = out_dir / f"{stem}_MA{w}.xlsx"
        write_xlsx_with_date_format(smoothed, out_path)
        print(f"Wrote {out_path} ({w}-month MA, expanding when history < {w})")


if __name__ == "__main__":
    main()
