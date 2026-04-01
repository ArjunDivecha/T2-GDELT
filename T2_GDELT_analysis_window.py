"""
=============================================================================
MODULE: T2_GDELT_analysis_window.py
=============================================================================

PURPOSE (simple terms):
  All T2 pipeline outputs that should line up with GDELT use the same monthly
  start and end dates. Those dates are read from ``GDELT.xlsx`` so you only
  change the workbook when coverage changes — scripts stay in sync.

HOW THE WINDOW IS DEFINED:
  - **Start month:** First row in sheet ``monthly_metronome`` where at least one
    country column has a real number (not blank). For the current GDELT file
    this is **2015-09-01** (September 2015).
  - **End month:** Last date in column A of ``monthly_metronome`` (e.g. **2026-03-01**).

INPUT FILE:
  - GDELT.xlsx (same folder as this module by default)
    Sheet: monthly_metronome — column A = month stamps, row 1 = country names.

OUTPUT:
  - Helpers may rewrite workbooks (e.g. ``clip_t2_master_excel``) after a full
    build so on-disk data matches the GDELT window.

VERSION: 1.1
LAST UPDATED: 2026-03-31

DEPENDENCIES: pandas, openpyxl (for reading GDELT.xlsx)
=============================================================================
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def _default_gdelt_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "GDELT.xlsx")


def get_gdelt_analysis_window(gdelt_path: str | None = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return inclusive (start, end) month timestamps aligned with GDELT coverage.

    Start = first month where ``monthly_metronome`` has any non-NaN value in
    country columns. End = max date in column A.

    Raises
    ------
    FileNotFoundError
        If GDELT.xlsx is missing.
    ValueError
        If the sheet is empty or no valid date range is found.
    """
    path = gdelt_path or _default_gdelt_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GDELT workbook not found: {path}\n"
            "Place GDELT.xlsx next to this module or pass gdelt_path=..."
        )

    df = pd.read_excel(path, sheet_name="monthly_metronome", engine="openpyxl")
    if df.shape[0] < 1 or df.shape[1] < 2:
        raise ValueError("GDELT sheet monthly_metronome has no data rows or no country columns.")

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    country_block = df.iloc[:, 1:]
    has_data = country_block.notna().any(axis=1)
    if not has_data.any():
        raise ValueError("GDELT monthly_metronome: no row has any non-missing country values.")

    first_row_label = has_data[has_data].index[0]
    start = pd.Timestamp(dates.loc[first_row_label]).normalize()
    end = pd.Timestamp(dates.max()).normalize()

    if pd.isna(start) or pd.isna(end):
        raise ValueError("GDELT monthly_metronome: could not parse start/end dates.")
    if start > end:
        raise ValueError(f"Invalid GDELT window: start {start} > end {end}")

    return start, end


def clip_monthly_index_frame(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Keep rows whose index is between start and end (inclusive), monthly stamps.

    The index is converted to datetime if needed. Returns a copy.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    mask = (out.index >= start) & (out.index <= end)
    clipped = out.loc[mask]
    if clipped.empty:
        raise ValueError(
            f"After clipping to [{start.date()} .. {end.date()}], DataFrame has no rows."
        )
    return clipped


def clip_long_format_dates(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter long/tidy data: keep rows with date in [start, end] inclusive."""
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found.")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    mask = (out[date_col] >= start) & (out[date_col] <= end)
    out = out.loc[mask]
    if out.empty:
        raise ValueError(
            f"After clipping {date_col} to [{start.date()} .. {end.date()}], no rows remain."
        )
    return out


def clip_t2_master_excel(path: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Rewrite ``T2 Master.xlsx`` in place: keep only rows whose **first column**
    parses as a date in [start, end] (inclusive).

    Sheets named ``README`` are copied unchanged. Sheets whose first column has
    no valid dates are copied unchanged (safety).
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"T2 Master not found: {path}")

    xl = pd.ExcelFile(path, engine="openpyxl")
    try:
        buffers: dict[str, pd.DataFrame] = {
            sn: pd.read_excel(xl, sheet_name=sn) for sn in xl.sheet_names
        }
    finally:
        xl.close()

    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in buffers.items():
            if sheet_name.upper() == "README" or df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            col0 = df.iloc[:, 0]
            dt = pd.to_datetime(col0, errors="coerce")
            if not dt.notna().any():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            mask = (dt >= start) & (dt <= end)
            clipped = df.loc[mask]
            if clipped.empty:
                raise ValueError(
                    f"T2 Master sheet {sheet_name!r}: no rows between "
                    f"{start.date()} and {end.date()}."
                )
            clipped.to_excel(writer, sheet_name=sheet_name, index=False)
