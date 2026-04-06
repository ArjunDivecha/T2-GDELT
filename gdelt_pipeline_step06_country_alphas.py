"""
Shared Step Six logic: country alphas = factor exposures × factor alphas (T60).

Used by:
- Step Six GDELT Create Country Alphas from Factor alphas.py
- Step Six T2 GDELT Combined Create Country Alphas from Factor alphas.py

VERSION: 1.0  LAST UPDATED: 2026-04-02
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from gdelt_track_config import PostStep5TrackConfig, T2_MASTER_FILE


def _load_data(
    t60_file: str,
    exposure_file: str,
    t2_master_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list | None]:
    print(f"Loading factor alpha data from {t60_file}...")
    factor_df = pd.read_excel(t60_file, sheet_name=0)
    factor_df = factor_df.rename(columns={factor_df.columns[0]: "Date"})
    factor_df["Date"] = pd.to_datetime(factor_df["Date"])

    print(f"Loading country exposure data from {exposure_file}...")
    exposure_df = pd.read_csv(exposure_file)
    exposure_df["Date"] = pd.to_datetime(exposure_df["Date"])

    print(f"Loading country order from {t2_master_file}...")
    try:
        country_order_df = pd.read_excel(t2_master_file, sheet_name="1MRet")
        country_order = country_order_df.columns[1:].tolist()
        print(f"Found {len(country_order)} countries in reference file")
    except Exception as e:
        print(f"Warning: Could not load country order: {e}")
        country_order = None
    return factor_df, exposure_df, country_order


def _get_available_dates(factor_df: pd.DataFrame, exposure_df: pd.DataFrame) -> list:
    factor_dates = set(factor_df["Date"])
    exposure_dates = set(exposure_df["Date"])
    common_dates = sorted(factor_dates.intersection(exposure_dates))
    return common_dates


def _process_month(
    date_obj,
    factor_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    factor_names: list,
    missing_log: list,
) -> pd.DataFrame:
    factor_row = factor_df[factor_df["Date"] == date_obj]
    if factor_row.empty:
        return pd.DataFrame()
    factor_row = factor_row.iloc[0]
    date_countries = exposure_df[exposure_df["Date"] == date_obj]
    results = []
    factor_means = {}
    for f in factor_names:
        vals = date_countries[f]
        factor_means[f] = vals[vals.notna()].mean()
    for _, row in date_countries.iterrows():
        country = row["Country"]
        total = 0.0
        valid_factors = 0
        for f in factor_names:
            exposure = row[f]
            if pd.isna(exposure):
                exposure = factor_means[f]
                missing_log.append(
                    {"date": date_obj, "country": country, "factor": f, "filled_value": exposure}
                )
            alpha = factor_row[f]
            if pd.isna(alpha):
                continue
            if pd.isna(exposure):
                continue
            total += exposure * alpha
            valid_factors += 1
        results.append(
            {"date": date_obj, "country": country, "total_score": total, "valid_factors": valid_factors}
        )
    return pd.DataFrame(results)


def run_country_alphas(cfg: PostStep5TrackConfig) -> None:
    """Build ``cfg.country_alphas_out`` from T60 and exposure CSV."""
    start_time = time.time()
    factor_df, exposure_df, country_order = _load_data(
        cfg.t60_xlsx, cfg.exposure_csv, T2_MASTER_FILE
    )
    common_dates = _get_available_dates(factor_df, exposure_df)
    if not common_dates:
        raise ValueError("No overlapping dates between T60 and exposure file.")
    exposure_factors = [c for c in exposure_df.columns if c not in ("Date", "Country")]
    factor_file_factors = [c for c in factor_df.columns if c != "Date"]
    factor_names = [f for f in factor_file_factors if f in exposure_factors]
    print(f"Common dates: {len(common_dates)}; factors aligned: {len(factor_names)}")

    missing_log: list = []
    all_results = []
    for date_obj in tqdm(common_dates, desc="Country alphas months"):
        result = _process_month(date_obj, factor_df, exposure_df, factor_names, missing_log)
        all_results.append(result)
    combined_results = pd.concat(all_results, ignore_index=True)

    country_col = "country"
    if "country" not in combined_results.columns:
        for col in combined_results.columns:
            if "country" in col.lower():
                country_col = col
                break

    pivot_table = combined_results.pivot_table(
        index="date", columns=country_col, values="total_score"
    )
    if country_order:
        available_countries = [c for c in country_order if c in pivot_table.columns]
        for country in pivot_table.columns:
            if country not in available_countries:
                available_countries.append(country)
        pivot_table = pivot_table[available_countries]

    missing_data = pivot_table.isna().sum()
    country_quality = pd.DataFrame(
        {
            "country": missing_data.index,
            "missing_months": missing_data.values,
            "completeness_pct": 100 * (1 - missing_data.values / len(common_dates)),
        }
    ).sort_values("completeness_pct")

    factor_count_pivot = combined_results.pivot_table(
        index="date", columns=country_col, values="valid_factors"
    )

    out = cfg.country_alphas_out
    print(f"Writing {out}...")
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        pivot_table.to_excel(writer, sheet_name="Country_Scores")
        country_quality.to_excel(writer, sheet_name="Data_Quality", index=False)
        factor_count_pivot.to_excel(writer, sheet_name="Factor_Counts")
        if missing_log:
            pd.DataFrame(missing_log).to_excel(writer, sheet_name="Missing_Data_Log", index=False)
        workbook = writer.book
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        for sheet in ("Country_Scores", "Factor_Counts"):
            writer.sheets[sheet].set_column(0, 0, 12, date_format)

    print(f"Done in {time.time() - start_time:.1f}s — {out}")
