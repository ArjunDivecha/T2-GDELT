"""
Step Eight — factor weights → country weights (fuzzy 15–25% band).

Shared by GDELT and T2+GDELT Combined tracks.

VERSION: 1.0  LAST UPDATED: 2026-04-02
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from tqdm import tqdm

from gdelt_track_config import PostStep5TrackConfig, T2_MASTER_FILE

SOFT_BAND_TOP = 0.15
SOFT_BAND_CUTOFF = 0.25

# Kept for parity with classic Step Eight (T2); unused in vectorized path.
INVERTED_FEATURES = {
    "BEST Cash Flow",
    "BEST Div Yield",
    "BEST EPS 3 Year",
    "BEST PBK",
    "BEST PE",
    "BEST PS",
    "BEST ROE",
    "EV to EBITDA",
    "Shiller PE",
    "Trailing PE",
    "Positive PE",
    "Best Price Sales",
    "Debt To EV",
    "Currency Change",
    "Debt to GDP",
    "REER",
    "10Yr Bond 12",
    "Bond Yield Change",
    "RSI14",
    "Advance Decline",
    "1MTR",
    "3MTR",
    "Bloom Country Risk",
}


def run_country_weights(cfg: PostStep5TrackConfig) -> None:
    weights_file = cfg.rolling_weights_xlsx
    factor_file = cfg.factors_master_csv
    print(f"Loading {weights_file} and {factor_file}...")
    try:
        feature_weights_df = pd.read_excel(weights_file, sheet_name="Net_Weights", index_col=0)
    except Exception:
        feature_weights_df = pd.read_excel(weights_file, index_col=0)
    factor_df = pd.read_csv(factor_file)
    factor_df["date"] = pd.to_datetime(factor_df["date"])

    all_countries = factor_df["country"].unique()
    all_dates = list(feature_weights_df.index)
    all_weights = pd.DataFrame(index=all_dates, columns=all_countries).fillna(0.0)
    by_date = factor_df.groupby("date")

    def calculate_country_contributions_for_date(date_dt):
        if date_dt not in feature_weights_df.index:
            return None, None
        w = feature_weights_df.loc[date_dt].astype(float)
        w = w[w.abs() > 1e-10]
        if w.empty:
            return None, None
        try:
            slice_df = by_date.get_group(date_dt)
        except KeyError:
            return None, None
        pivot = slice_df.pivot(index="country", columns="variable", values="value")
        common_factors = pivot.columns.intersection(w.index)
        if len(common_factors) == 0:
            return None, None
        V = pivot[common_factors]
        w_vec = w.loc[common_factors]
        rank_desc = V.rank(axis=0, method="first", ascending=False)
        rank_asc = V.rank(axis=0, method="first", ascending=True)
        counts = V.notna().sum(axis=0).replace(0, np.nan)
        counts_mat = np.tile(counts.values, (len(V.index), 1))
        rank_desc_pct = rank_desc.values / counts_mat
        rank_asc_pct = rank_asc.values / counts_mat
        pos_mask = (w_vec.values > 0).astype(float)
        pos_mask_mat = np.tile(pos_mask, (len(V.index), 1))
        rank_pct = np.where(pos_mask_mat > 0, rank_desc_pct, rank_asc_pct)
        full_mask = (rank_pct < SOFT_BAND_TOP).astype(float)
        in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
        taper = 1.0 - (rank_pct - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
        taper = np.where(in_band, taper, 0.0)
        fuzzy = full_mask + taper
        col_sums = fuzzy.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        fuzzy_norm = fuzzy / col_sums
        w_mat = np.tile(w_vec.values, (len(V.index), 1))
        contrib = fuzzy_norm * w_mat
        contributions_df = pd.DataFrame(contrib, index=V.index, columns=common_factors)
        country_weights = contributions_df.sum(axis=1)
        return country_weights, contributions_df

    for date in tqdm(all_dates, desc="Country weights"):
        date_dt = pd.to_datetime(date)
        country_weights, _ = calculate_country_contributions_for_date(date_dt)
        if country_weights is None:
            continue
        all_weights.loc[date, country_weights.index] = country_weights.values

    print("\nWeight sum statistics:")
    print(all_weights.sum(axis=1).describe())

    out_main = cfg.final_country_weights_xlsx
    with pd.ExcelWriter(out_main, engine="xlsxwriter") as writer:
        all_weights.to_excel(writer, sheet_name="All Periods")
        summary_stats = pd.DataFrame(
            {
                "Mean Weight": all_weights.mean(),
                "Std Dev": all_weights.std(),
                "Min Weight": all_weights.min(),
                "Max Weight": all_weights.max(),
                "Days with Weight": (all_weights.abs() > 0).sum(),
            }
        ).sort_values("Mean Weight", ascending=False)
        summary_stats.to_excel(writer, sheet_name="Summary Statistics")
        non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
        if len(non_zero_dates) > 0:
            latest_valid_date = non_zero_dates[-1]
            latest_weights = pd.DataFrame(
                {
                    "Weight": all_weights.loc[latest_valid_date],
                    "Average Weight": all_weights.mean(),
                    "Days with Weight": (all_weights.abs() > 0).sum(),
                    "Latest Date": pd.Series(
                        [latest_valid_date] * len(all_weights.columns), index=all_weights.columns
                    ),
                }
            ).sort_values("Weight", ascending=False)
        else:
            latest_weights = pd.DataFrame(
                {
                    "Weight": all_weights.iloc[-1],
                    "Average Weight": all_weights.mean(),
                    "Days with Weight": (all_weights.abs() > 0).sum(),
                }
            ).sort_values("Weight", ascending=False)
        latest_weights.to_excel(writer, sheet_name="Latest Weights")
    print(f"Saved {out_main}")

    _write_country_final(
        all_weights,
        calculate_country_contributions_for_date,
        cfg.country_final_xlsx,
    )


def _write_country_final(all_weights, calculate_country_contributions_for_date, country_final_path: str) -> None:
    print(f"\nWriting {country_final_path}...")
    non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
    if len(non_zero_dates) == 0:
        print("Error: No date with non-zero weights")
        return
    latest_valid_date = non_zero_dates[-1]
    latest_weights = all_weights.loc[latest_valid_date]
    country_weight_dict = latest_weights.to_dict()
    try:
        master_df = pd.read_excel(T2_MASTER_FILE)
        country_columns = list(master_df.columns[1:])
        all_weights_df = pd.DataFrame({"Country": country_columns, "Weight": 0.0})
        algorithm_to_master_mapping = {}
        for country, weight in country_weight_dict.items():
            match_idx = all_weights_df[all_weights_df["Country"].str.lower() == country.lower()].index
            if len(match_idx) > 0:
                all_weights_df.loc[match_idx[0], "Weight"] = weight
            else:
                mapped_name = algorithm_to_master_mapping.get(country, country)
                match_idx = all_weights_df[all_weights_df["Country"].str.lower() == mapped_name.lower()].index
                if len(match_idx) > 0:
                    all_weights_df.loc[match_idx[0], "Weight"] = weight
                else:
                    new_row = pd.DataFrame({"Country": [country], "Weight": [weight]})
                    all_weights_df = pd.concat([all_weights_df, new_row], ignore_index=True)
        sorted_weights = all_weights_df
    except Exception as e:
        print(f"Error reading T2 Master: {e}; fallback column order")
        sorted_weights = pd.DataFrame(list(country_weight_dict.items()), columns=["Country", "Weight"])

    _, contributions_df = calculate_country_contributions_for_date(latest_valid_date)
    if contributions_df is None:
        print("Error: no factor contributions for latest date")
        return
    contributions_sum = contributions_df.sum(axis=1)
    common_countries = contributions_sum.index.intersection(latest_weights.index)
    if len(common_countries) > 0:
        row_diff = (contributions_sum[common_countries] - latest_weights[common_countries]).abs().max()
        if row_diff > 1e-6:
            print(f"Warning: contrib vs weight max diff = {row_diff:.6f}")
    factor_totals = contributions_df.sum().sort_values(ascending=False)
    contributions_df = contributions_df[factor_totals.index]
    contributions_reset = contributions_df.reset_index().rename(
        columns={"index": "Country", "country": "Country"}
    )
    final_df = pd.merge(sorted_weights, contributions_reset, on="Country", how="left").fillna(0.0)
    final_df = final_df.drop_duplicates(subset="Country", keep="first")

    with pd.ExcelWriter(country_final_path, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, sheet_name="Country Weights", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Country Weights"]
        header_format = workbook.add_format(
            {"bold": True, "text_wrap": True, "valign": "top", "bg_color": "#D9D9D9", "border": 1}
        )
        pct_format = workbook.add_format({"num_format": "0.00%"})
        worksheet.set_column(0, 0, 15)
        last_col = final_df.shape[1] - 1
        worksheet.set_column(1, last_col, 12, pct_format)
        for col_num, value in enumerate(final_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        total_weight = final_df["Weight"].sum()
        last_row = len(final_df) + 1
        bold_format = workbook.add_format({"bold": True})
        total_format = workbook.add_format({"bold": True, "num_format": "0.00%"})
        worksheet.write(last_row, 0, "TOTAL", bold_format)
        worksheet.write(last_row, 1, total_weight, total_format)
        col_sums = final_df.drop(columns=["Country"]).sum()
        for col_idx, col_name in enumerate(final_df.columns[1:], start=1):
            worksheet.write(last_row, col_idx, col_sums[col_name], total_format)
    print(f"Saved {country_final_path}")
