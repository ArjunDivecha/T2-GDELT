"""
=============================================================================
SCRIPT NAME: build_factor_return_attribution.py
=============================================================================

INPUT FILES:
- T2_rolling_window_weights.xlsx: Monthly factor weights (sum to 1 per month).
  Sheet Sheet1; first column is month-end (or month-start) date; 82 factor columns.
- T2_Optimizer.xlsx: Monthly factor returns (% or decimal per pipeline).
  Sheet Monthly_Net_Returns; column Date; same 82 factor names as weights.

OUTPUT FILES:
- T2_Factor_Return_Attribution.xlsx:
  - Monthly_Attribution: for each month (inner join on Date), contribution
    per factor = weight * return, plus Monthly_Portfolio_Return (sum of contributions).
  - Monthly_Avg_vs_Variance: portfolio return split into (1) what you would have
    earned using each factor's average weight w_bar times that month's factor
    return, and (2) timing: sum_i (w_{i,t} - w_bar_i) * r_{i,t}. The two sum to
    Monthly_Portfolio_Return each month.
  - Factor_Summary: per-factor totals from average position vs variation around
    it, plus overall contribution metrics and pct of grand total.
  - AvgVar_Portfolio_Summary: one row — sums over all months for the portfolio-
    level split (matches column sums on Monthly_Avg_vs_Variance).

VERSION: 1.1
LAST UPDATED: 2026-03-19

DESCRIPTION:
Creates additive attribution: each month, factor i contributes weight * return.
Also splits each factor and the portfolio using time-mean weights w_bar:
w_t*r_t = w_bar*r_t + (w_t - w_bar)*r_t ("average position" vs "variation / timing").
Months present in only one input file are dropped.

DEPENDENCIES:
- pandas, openpyxl, xlsxwriter

USAGE:
python build_factor_return_attribution.py

NOTES:
- Does not chain-link returns; summary uses sums of monthly contributions.
=============================================================================
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "T2_rolling_window_weights.xlsx")
RETURNS_PATH = os.path.join(BASE_DIR, "T2_Optimizer.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "T2_Factor_Return_Attribution.xlsx")


def main() -> None:
    weights = pd.read_excel(WEIGHTS_PATH, sheet_name="Sheet1")
    weights = weights.rename(columns={weights.columns[0]: "Date"})
    weights["Date"] = pd.to_datetime(weights["Date"])

    rets = pd.read_excel(RETURNS_PATH, sheet_name="Monthly_Net_Returns")
    rets["Date"] = pd.to_datetime(rets["Date"])

    factor_cols = [c for c in weights.columns if c != "Date"]
    missing = [c for c in factor_cols if c not in rets.columns]
    if missing:
        raise ValueError(f"Factor columns missing in returns: {missing[:5]}...")

    merged = pd.merge(
        weights[["Date"] + factor_cols],
        rets[["Date"] + factor_cols],
        on="Date",
        how="inner",
        suffixes=("_w", "_r"),
    )
    if merged.empty:
        raise ValueError("No overlapping dates between weights and returns.")

    # Wide merge duplicate names — merge gives _w and _r suffixes
    contrib = pd.DataFrame({"Date": merged["Date"]})
    for f in factor_cols:
        wcol = f"{f}_w"
        rcol = f"{f}_r"
        contrib[f] = merged[wcol].astype(float) * merged[rcol].astype(float)

    contrib["Monthly_Portfolio_Return"] = contrib[factor_cols].sum(axis=1)

    # Mean weight per factor over the overlapping window (average strategic position).
    w_bar = merged[[f"{f}_w" for f in factor_cols]].astype(float).mean(axis=0)
    w_bar.index = factor_cols

    avg_weight_port: list[float] = []
    variation_port: list[float] = []
    for _, row in merged.iterrows():
        r_vec = np.array([float(row[f"{f}_r"]) for f in factor_cols])
        w_vec = np.array([float(row[f"{f}_w"]) for f in factor_cols])
        wbar_vec = np.array([float(w_bar[f]) for f in factor_cols])
        avg_weight_port.append(float(np.dot(wbar_vec, r_vec)))
        variation_port.append(float(np.dot(w_vec - wbar_vec, r_vec)))

    monthly_avg_var = pd.DataFrame(
        {
            "Date": merged["Date"].values,
            "Portfolio_From_Avg_Weights": avg_weight_port,
            "Portfolio_From_Weight_Variation": variation_port,
        }
    )
    monthly_avg_var["Monthly_Portfolio_Return_Check"] = (
        monthly_avg_var["Portfolio_From_Avg_Weights"]
        + monthly_avg_var["Portfolio_From_Weight_Variation"]
    )

    grand_total = float(contrib["Monthly_Portfolio_Return"].sum())
    summary_rows = []
    for f in factor_cols:
        col_sum = float(contrib[f].sum())
        w_series = merged[f"{f}_w"].astype(float)
        r_series = merged[f"{f}_r"].astype(float)
        wb = float(w_bar[f])
        from_avg_w = float((wb * r_series).sum())
        from_var_w = float(((w_series - wb) * r_series).sum())
        if not np.isclose(from_avg_w + from_var_w, col_sum, rtol=1e-9, atol=1e-6):
            raise RuntimeError(f"Decomposition mismatch for factor {f}")
        summary_rows.append(
            {
                "Factor": f,
                "Total_Contribution_Sum_of_Months": col_sum,
                "Total_From_Avg_Weight": from_avg_w,
                "Total_From_Weight_Variation": from_var_w,
                "Pct_of_Total_From_Avg_Weight": (from_avg_w / col_sum * 100.0)
                if col_sum != 0
                else float("nan"),
                "Avg_Monthly_Contribution": col_sum / len(contrib),
                "Pct_of_Grand_Total": (col_sum / grand_total * 100.0)
                if grand_total != 0
                else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        "Total_Contribution_Sum_of_Months", ascending=False
    )

    port_total_avg = float(monthly_avg_var["Portfolio_From_Avg_Weights"].sum())
    port_total_var = float(monthly_avg_var["Portfolio_From_Weight_Variation"].sum())
    if not np.isclose(port_total_avg + port_total_var, grand_total, rtol=1e-9, atol=1e-4):
        raise RuntimeError("Portfolio avg/variation split does not sum to grand total")

    port_summary = pd.DataFrame(
        [
            {
                "Sum_Monthly_Portfolio_Return": grand_total,
                "Sum_From_Avg_Weights": port_total_avg,
                "Sum_From_Weight_Variation": port_total_var,
                "Pct_From_Avg_Weights": port_total_avg / grand_total * 100.0
                if grand_total != 0
                else float("nan"),
                "Pct_From_Weight_Variation": port_total_var / grand_total * 100.0
                if grand_total != 0
                else float("nan"),
            }
        ]
    )

    # Excel output with date formatting (xlsxwriter)
    monthly = contrib.sort_values("Date").reset_index(drop=True)
    monthly_avg_var = monthly_avg_var.sort_values("Date").reset_index(drop=True)

    with pd.ExcelWriter(OUTPUT_PATH, engine="xlsxwriter") as writer:
        monthly.to_excel(writer, sheet_name="Monthly_Attribution", index=False)
        monthly_avg_var.to_excel(
            writer, sheet_name="Monthly_Avg_vs_Variance", index=False
        )
        summary.to_excel(writer, sheet_name="Factor_Summary", index=False)
        port_summary.to_excel(writer, sheet_name="AvgVar_Portfolio_Summary", index=False)

        workbook = writer.book
        date_fmt = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        for sheet_name, df in [
            ("Monthly_Attribution", monthly),
            ("Monthly_Avg_vs_Variance", monthly_avg_var),
            ("Factor_Summary", summary),
            ("AvgVar_Portfolio_Summary", port_summary),
        ]:
            ws = writer.sheets[sheet_name]
            if "Date" in df.columns:
                d_idx = df.columns.get_loc("Date")
                for row in range(len(df)):
                    ws.write(row + 1, d_idx, df["Date"].iloc[row], date_fmt)
                ws.set_column(d_idx, d_idx, 14)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Months in attribution: {len(monthly)}")
    print(f"  Date range: {monthly['Date'].min().date()} to {monthly['Date'].max().date()}")
    print(f"  Grand total (sum of monthly portfolio returns): {grand_total:.6f}")
    print(f"  From avg weights (sum over months): {port_total_avg:.6f}")
    print(f"  From weight variation (sum over months): {port_total_var:.6f}")


if __name__ == "__main__":
    main()
