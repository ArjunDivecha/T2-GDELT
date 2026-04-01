"""
=============================================================================
SCRIPT NAME: Step Two GDELT Create Tidy.py
=============================================================================

INPUT FILES:
- GDELT.xlsx — all data sheets (README skipped); wide monthly panel, dates column A.
- T2 Master.xlsx — sheet **1MRet** only (monthly country returns; GDELT has no returns).
- Step Factor Categories.xlsx — sheet **Asset Class** copied into the GDELT categories file.

OUTPUT FILES:
- GDELT_Factors_MasterCSV.csv — long format: date, country, variable, value.
  For each GDELT sheet, **Raw** uses the same ``variable`` name as ``Archive/Step Three GDELT.py``
  (the sheet name, truncated if needed). CS/TS are ``<sheet>_CS`` and ``<sheet>_TS``.
  **1MRet** is taken from ``Normalized_T2_MasterCSV.csv`` if present, else ``T2 Master.xlsx``.
- Step Factor Categories GDELT.xlsx — sheets **Factor Categories** (one row per GDELT
  factor variant + Max weight) and **Asset Class** (same as T2).

VERSION: 1.2
LAST UPDATED: 2026-03-31

DESCRIPTION:
Builds Raw / cross-sectional (CS) / time-series (TS) variants for every GDELT sheet
(including z-scored sheets), merges T2 returns, clips to the GDELT analysis window,
and writes the parallel factor-category workbook for Step Five GDELT.

**Risk sheets:** If ``FLIP_GDELT_RISK_SHEETS`` is True (default), sheets whose names
contain ``risk`` are sign-flipped before Raw/CS/TS. Set to **False** to match the
legacy ``Archive/Step Three GDELT.py`` behavior (no flip).

**Parity with Archive Step Three GDELT:** Raw factor names match archive; 1MRet from
Normalized CSV when available; analysis window still clips to ``T2_GDELT_analysis_window``
(unlike archive, which used all overlapping dates).

DEPENDENCIES: pandas, numpy, openpyxl, xlsxwriter
USAGE: python "Step Two GDELT Create Tidy.py"
=============================================================================
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from gdelt_country_map import map_country_label
from T2_GDELT_analysis_window import clip_long_format_dates, get_gdelt_analysis_window

README_SHEET = "README"
GDELT_PATH = "GDELT.xlsx"
T2_MASTER_PATH = "T2 Master.xlsx"
T2_CATEGORIES_PATH = "Step Factor Categories.xlsx"
NORMALIZED_T2_CSV = "Normalized_T2_MasterCSV.csv"
OUTPUT_CSV = "GDELT_Factors_MasterCSV.csv"
OUTPUT_CATEGORIES = "Step Factor Categories GDELT.xlsx"

# Sheets whose values should be multiplied by −1 before ranking so that
# "higher = better" semantics are preserved (e.g. negative tone → flip so
# less-negative countries rank higher).
FLIP_PREFIXES = (
    "country_news_attention",    # country_news_attention
    "country_news_sentiment",    # country_news_sentiment, country_news_sentiment_raw
    "foreign_tone",              # foreign_tone, _fast, _fast_z
    "local_foreign_gap",         # local_foreign_gap
    "lf_gap",                    # lf_gap_z
    "local_tone",                # local_tone, _fast, _fast_z
    "metronome",                 # metronome_rank_pct
    "monthly_metronome",         # monthly_metronome
    "sentiment",                 # sentiment_fast, _fast_z, _slow, _slow_z, _trend, _trend_z, _x_attention
    "tone_dispersion",           # tone_dispersion
    "tone_mean",                 # tone_mean
    "tone_wavg",                 # tone_wavg_wordcount
)
# NOT flipped: country_news_risk, country_news_risk_raw (higher risk = bad, no flip needed)


def _truncate_sheet_key(name: str, max_len: int = 80) -> str:
    """Stable variable prefix from sheet name (Excel sheet names are <=31 chars)."""
    s = str(name).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _infer_category(sheet_name: str) -> str:
    """Rough bucket for Step Factor Categories (first word / segment)."""
    s = sheet_name.strip()
    if "_" in s:
        return s.split("_")[0].title()
    return "GDELT"


def _should_flip(sheet_name: str) -> bool:
    """True if this sheet should be sign-flipped (× −1) before Raw/CS/TS."""
    lower = str(sheet_name).lower().strip()
    return any(lower.startswith(p) for p in FLIP_PREFIXES)


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    cs_means = df.mean(axis=1)
    cs_stds = df.std(axis=1).replace(0, np.nan)
    out = df.subtract(cs_means, axis=0).divide(cs_stds, axis=0)
    return out.fillna(0.0)


def _ts_normalize(df: pd.DataFrame) -> pd.DataFrame:
    ts_norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for country in df.columns:
        country_data = df[country].copy()
        ts_mean = country_data.expanding().mean()
        ts_std = country_data.expanding().std()
        ts_norm[country] = np.where(
            ts_std == 0,
            0.0,
            (country_data - ts_mean) / ts_std,
        )
    return ts_norm.fillna(0.0)


def _sheet_to_long_variants(
    sheet_name: str, df_raw: pd.DataFrame
) -> List[pd.DataFrame]:
    """Return tidy frames for CS and TS only (Raw dropped to reduce redundancy)."""
    key = _truncate_sheet_key(sheet_name)
    parts: List[pd.DataFrame] = []
    variants: dict[str, pd.DataFrame] = {
        "CS": _cs_normalize(df_raw),
        "TS": _ts_normalize(df_raw),
    }
    for suffix, wide in variants.items():
        tidy = wide.reset_index()
        tidy = tidy.rename(columns={tidy.columns[0]: "date"})
        tidy = tidy.melt(id_vars=["date"], var_name="country", value_name="value")
        tidy["variable"] = f"{key}_{suffix}"
        parts.append(tidy[["date", "country", "variable", "value"]])
    return parts


def _load_1mret_long() -> pd.DataFrame:
    """Prefer Normalized_T2_MasterCSV (same 1MRet as Archive Step Three GDELT), else T2 Master."""
    if os.path.isfile(NORMALIZED_T2_CSV):
        csv = pd.read_csv(NORMALIZED_T2_CSV)
        lower = {c.lower(): c for c in csv.columns}
        col_date = lower.get("date", "date")
        if col_date not in csv.columns:
            col_date = "date"
        csv = csv.rename(columns={col_date: "date"})
        csv["date"] = pd.to_datetime(csv["date"], errors="coerce")
        ret = csv[csv["variable"] == "1MRet"][["date", "country", "variable", "value"]].copy()
        if not ret.empty:
            ret["country"] = ret["country"].astype(str)
            ret["date"] = ret["date"].dt.to_period("M").dt.to_timestamp()
            return ret
    if not os.path.isfile(T2_MASTER_PATH):
        raise FileNotFoundError(
            f"Need 1MRet: add {NORMALIZED_T2_CSV} or {T2_MASTER_PATH}"
        )
    df = pd.read_excel(T2_MASTER_PATH, sheet_name="1MRet", engine="openpyxl")
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    value_cols = [c for c in df.columns if c != "date"]
    long = df.melt(
        id_vars=["date"],
        value_vars=value_cols,
        var_name="country",
        value_name="value",
    )
    long["variable"] = "1MRet"
    long["country"] = long["country"].astype(str)
    return long[["date", "country", "variable", "value"]]


def _write_factor_categories_gdelt(sheet_names: List[str]) -> None:
    """One row per ``_CS`` / ``_TS``; copy Asset Class from T2 categories file."""
    rows = []
    for sn in sheet_names:
        key = _truncate_sheet_key(sn)
        cat = _infer_category(sn)
        rows.append({"Factor Name": f"{key}_CS", "Category": cat, "Max": 1.0})
        rows.append({"Factor Name": f"{key}_TS", "Category": cat, "Max": 1.0})
    fc = pd.DataFrame(rows)
    if not os.path.isfile(T2_CATEGORIES_PATH):
        raise FileNotFoundError(f"Missing {T2_CATEGORIES_PATH} (need Asset Class sheet).")
    asset = pd.read_excel(T2_CATEGORIES_PATH, sheet_name="Asset Class", engine="openpyxl")
    with pd.ExcelWriter(OUTPUT_CATEGORIES, engine="openpyxl", mode="w") as writer:
        fc.to_excel(writer, sheet_name="Factor Categories", index=False)
        asset.to_excel(writer, sheet_name="Asset Class", index=False)
    print(f"Wrote {OUTPUT_CATEGORIES} ({len(fc)} factor rows).")


def main() -> None:
    if not os.path.isfile(GDELT_PATH):
        raise FileNotFoundError(GDELT_PATH)

    xl = pd.ExcelFile(GDELT_PATH, engine="openpyxl")
    data_sheet_names = [s for s in xl.sheet_names if s != README_SHEET]

    all_parts: List[pd.DataFrame] = []
    sheets_loaded: List[str] = []
    for sheet in data_sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, engine="openpyxl")
        if df.empty or df.shape[1] < 2:
            continue
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        country_cols = [c for c in df.columns if c != "date"]
        df = df.rename(columns={c: map_country_label(c) for c in country_cols})
        wide = df.set_index("date")
        wide.index.name = "date"
        wide = wide.apply(pd.to_numeric, errors="coerce")
        if _should_flip(sheet):
            wide = wide * -1.0
            print(f"  FLIPPED: {sheet}")
        all_parts.extend(_sheet_to_long_variants(sheet, wide))
        sheets_loaded.append(sheet)

    if not all_parts:
        raise ValueError("No GDELT sheets produced tidy data.")

    factors_long = pd.concat(all_parts, ignore_index=True)
    ret_long = _load_1mret_long()
    combined = pd.concat([factors_long, ret_long], ignore_index=True)

    win_s, win_e = get_gdelt_analysis_window()
    combined = clip_long_format_dates(combined, win_s, win_e, date_col="date")
    combined = combined.sort_values(["date", "variable", "country"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}  rows={len(combined)}  window={win_s.date()}..{win_e.date()}")

    _write_factor_categories_gdelt(sheets_loaded)


if __name__ == "__main__":
    main()
