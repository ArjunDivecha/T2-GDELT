"""
=============================================================================
SCRIPT NAME: Step Two T2 GDELT Combined Create Tidy.py
=============================================================================

INPUT FILES:
- Normalized_T2_MasterCSV.csv — long T2 factors + returns (from Step Two Create Normalized Tidy.py)
- GDELT_Factors_MasterCSV.csv — GDELT CS/TS factors + 1MRet (from Step Two GDELT Create Tidy.py)
- Step Factor Categories.xlsx — sheet **Factor Categories** + **Asset Class** (T2 caps)
- Step Factor Categories GDELT.xlsx — sheet **Factor Categories** (GDELT caps)

OUTPUT FILES:
- Combined_T2_GDELT_Factors_MasterCSV.csv — single long table: all T2 variables (clipped to
  GDELT window) plus GDELT factors with names prefixed ``GDELT_`` (duplicate 1MRet from GDELT
  side is dropped; T2 supplies 1MRet).
- Step Factor Categories T2 GDELT Combined.xlsx — merged **Factor Categories** + T2 **Asset Class**

VERSION: 1.0
LAST UPDATED: 2026-04-01

DESCRIPTION:
Third parallel track: stack T2 normalized factors with GDELT factors so Steps 3–5 can run
one combined optimizer. GDELT variable names get a ``GDELT_`` prefix to avoid collisions with
T2 names.

DEPENDENCIES: pandas, openpyxl
USAGE:
  python "Step Two Create Normalized Tidy.py"   # prerequisite
  python "Step Two GDELT Create Tidy.py"        # prerequisite
  python "Step Two T2 GDELT Combined Create Tidy.py"
=============================================================================
"""

from __future__ import annotations

import os

import pandas as pd

from T2_GDELT_analysis_window import clip_long_format_dates, get_gdelt_analysis_window

T2_CSV = "Normalized_T2_MasterCSV.csv"
GDELT_CSV = "GDELT_Factors_MasterCSV.csv"
T2_CATEGORIES = "Step Factor Categories.xlsx"
GDELT_CATEGORIES = "Step Factor Categories GDELT.xlsx"
OUTPUT_CSV = "Combined_T2_GDELT_Factors_MasterCSV.csv"
OUTPUT_CATEGORIES = "Step Factor Categories T2 GDELT Combined.xlsx"

GDELT_PREFIX = "GDELT_"


def main() -> None:
    if not os.path.isfile(T2_CSV):
        raise FileNotFoundError(f"Missing {T2_CSV} — run Step Two Create Normalized Tidy.py first.")
    if not os.path.isfile(GDELT_CSV):
        raise FileNotFoundError(f"Missing {GDELT_CSV} — run Step Two GDELT Create Tidy.py first.")
    if not os.path.isfile(T2_CATEGORIES):
        raise FileNotFoundError(T2_CATEGORIES)
    if not os.path.isfile(GDELT_CATEGORIES):
        raise FileNotFoundError(GDELT_CATEGORIES)

    win_s, win_e = get_gdelt_analysis_window()

    t2 = pd.read_csv(T2_CSV)
    t2["date"] = pd.to_datetime(t2["date"], errors="coerce")
    t2 = t2.dropna(subset=["date"])
    t2["date"] = t2["date"].dt.to_period("M").dt.to_timestamp()
    t2["country"] = t2["country"].astype(str)
    t2 = clip_long_format_dates(t2, win_s, win_e, date_col="date")

    gdelt = pd.read_csv(GDELT_CSV)
    gdelt["date"] = pd.to_datetime(gdelt["date"], errors="coerce")
    gdelt = gdelt.dropna(subset=["date"])
    gdelt["date"] = gdelt["date"].dt.to_period("M").dt.to_timestamp()
    gdelt["country"] = gdelt["country"].astype(str)
    # GDELT file is already window-clipped; align again for safety
    gdelt = clip_long_format_dates(gdelt, win_s, win_e, date_col="date")

    g_fac = gdelt[gdelt["variable"] != "1MRet"].copy()
    g_fac["variable"] = GDELT_PREFIX + g_fac["variable"].astype(str)

    out = pd.concat([t2, g_fac], ignore_index=True)
    out = out.sort_values(["date", "variable", "country"]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}  rows={len(out)}  window={win_s.date()}..{win_e.date()}")
    print(f"  T2 rows (in window): {len(t2)}  GDELT factor rows: {len(g_fac)}")

    # --- Factor categories ---
    t2_fc = pd.read_excel(T2_CATEGORIES, sheet_name="Factor Categories", engine="openpyxl")
    g_fc = pd.read_excel(GDELT_CATEGORIES, sheet_name="Factor Categories", engine="openpyxl")
    if "Factor Name" not in t2_fc.columns or "Factor Name" not in g_fc.columns:
        raise ValueError("Factor Categories sheets must have a 'Factor Name' column.")

    g_fc = g_fc.copy()
    g_fc["Factor Name"] = GDELT_PREFIX + g_fc["Factor Name"].astype(str)

    merged_fc = pd.concat([t2_fc, g_fc], ignore_index=True)
    asset = pd.read_excel(T2_CATEGORIES, sheet_name="Asset Class", engine="openpyxl")

    with pd.ExcelWriter(OUTPUT_CATEGORIES, engine="openpyxl", mode="w") as writer:
        merged_fc.to_excel(writer, sheet_name="Factor Categories", index=False)
        asset.to_excel(writer, sheet_name="Asset Class", index=False)
    print(f"Wrote {OUTPUT_CATEGORIES}  ({len(merged_fc)} factor rows).")


if __name__ == "__main__":
    main()
