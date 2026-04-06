"""
Step Twelve — multi-horizon summary of factor portfolio returns (from optimizer sheet).

Reads ``Monthly_Net_Returns`` (or first sheet) from the track's optimizer workbook,
computes trailing 3-, 6-, and 12-month mean return per factor at each date, and writes
the **last available row** snapshot to Excel.

This is a lightweight substitute for a full multi-period forecast model.

VERSION: 1.0  LAST UPDATED: 2026-04-02
"""

from __future__ import annotations

import pandas as pd

from gdelt_track_config import PostStep5TrackConfig


def run_multiperiod_summary(cfg: PostStep5TrackConfig) -> None:
    path = cfg.optimizer_xlsx
    try:
        df = pd.read_excel(path, sheet_name="Monthly_Net_Returns", index_col=0)
    except Exception:
        df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric factor columns in {path}")

    out_rows = {}
    for h in (3, 6, 12):
        rolled = numeric.rolling(window=h, min_periods=h).mean()
        out_rows[f"trailing_{h}m_mean"] = rolled.iloc[-1]
    summary = pd.DataFrame(out_rows).T
    summary.index.name = "Horizon"
    out = cfg.multiperiod_summary_xlsx
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Last_Date_Snapshot")
        numeric.tail(24).to_excel(writer, sheet_name="Recent_Monthly_Returns")
    print(f"Saved {out} (last date {df.index.max().date()})")
