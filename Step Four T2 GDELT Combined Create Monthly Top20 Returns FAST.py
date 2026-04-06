#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Step Four T2 GDELT Combined Create Monthly Top20 Returns FAST.py
=============================================================================

INPUT FILES:
- Combined_T2_GDELT_Factors_MasterCSV.csv
- Portfolio_Data.xlsx (Benchmarks)
- Step Tcost.xlsx (sheet jjunk: country trading costs)

OUTPUT FILES:
- T2_GDELT_Combined_Optimizer.xlsx (Monthly_Net_Returns)
- T2_GDELT_Combined_T60.xlsx (T60)
- T2_GDELT_Combined_Trading_Cost.xlsx

VERSION: 1.0
LAST UPDATED: 2026-04-01

USAGE:
  python "Step Four T2 GDELT Combined Create Monthly Top20 Returns FAST.py"
=============================================================================
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = "Combined_T2_GDELT_Factors_MasterCSV.csv"
BENCHMARK_PATH = "Portfolio_Data.xlsx"
TRADING_COST_PATH = "Step Tcost.xlsx"
OUTPUT_OPTIMIZER = "T2_GDELT_Combined_Optimizer.xlsx"
OUTPUT_T60 = "T2_GDELT_Combined_T60.xlsx"
OUTPUT_TRADING_COST = "T2_GDELT_Combined_Trading_Cost.xlsx"

SOFT_BAND_TOP = 0.15
SOFT_BAND_CUTOFF = 0.25


def analyze_portfolios(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
    trading_costs: pd.Series,
) -> tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    monthly_net_returns: Dict[str, pd.Series] = {}
    monthly_trading_costs: Dict[str, pd.Series] = {}

    print("Pre-indexing data...")
    returns_data = data[data["variable"] == "1MRet"].copy()
    returns_data["value"] = pd.to_numeric(returns_data["value"], errors="coerce")
    returns_data = returns_data.rename(columns={"value": "return_value"})

    returns_by_date = {
        date: group[["country", "return_value"]].copy()
        for date, group in returns_data.groupby("date")
    }

    print("Pre-merging factor and returns data...")
    feature_merged_cache: Dict[str, dict] = {}

    for feature in features:
        feature_data = data[data["variable"] == feature].copy()
        if feature_data.empty:
            continue
        feature_data["value"] = pd.to_numeric(feature_data["value"], errors="coerce")
        feature_data = feature_data.dropna(subset=["value"])
        if feature_data.empty:
            continue
        feature_data = feature_data.rename(columns={"value": "factor_value"})
        feature_merged_cache[feature] = {}
        for date, group in feature_data.groupby("date"):
            if date not in returns_by_date:
                continue
            merged = pd.merge(
                group[["country", "factor_value"]],
                returns_by_date[date],
                on="country",
            )
            merged = merged.dropna(subset=["factor_value", "return_value"])
            if merged.empty:
                continue
            merged = merged.sort_values("factor_value", ascending=False).reset_index(drop=True)
            feature_merged_cache[feature][date] = (
                merged["country"].values,
                merged["factor_value"].values,
                merged["return_value"].values,
            )

    print(f"Processing {len(features)} features...")
    processed = 0
    for feature in features:
        if feature not in feature_merged_cache:
            continue
        feat_by_date = feature_merged_cache[feature]
        feature_dates = sorted(feat_by_date.keys())
        if not feature_dates:
            continue

        portfolio_returns = pd.Series(index=feature_dates, dtype=float)
        portfolio_trading_costs = pd.Series(index=feature_dates, dtype=float)

        for date in feature_dates:
            data_countries, _factor_values, return_values = feat_by_date[date]
            n = len(return_values)
            if n == 0:
                continue
            rank_pct = (np.arange(n) + 1) / n
            weights = np.zeros(n)
            weights[rank_pct < SOFT_BAND_TOP] = 1.0
            in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
            weights[in_band] = 1.0 - (rank_pct[in_band] - SOFT_BAND_TOP) / (
                SOFT_BAND_CUTOFF - SOFT_BAND_TOP
            )
            nonzero_mask = weights > 0
            if not nonzero_mask.any():
                continue
            weights_filtered = weights[nonzero_mask]
            returns_filtered = return_values[nonzero_mask]
            countries_filtered = data_countries[nonzero_mask]
            weights_filtered = weights_filtered / weights_filtered.sum()
            portfolio_returns[date] = float(np.dot(weights_filtered, returns_filtered))

            country_trading_costs = trading_costs.reindex(countries_filtered).to_numpy(dtype=float)
            valid_cost_mask = ~np.isnan(country_trading_costs)
            if valid_cost_mask.any():
                cw = weights_filtered[valid_cost_mask]
                cw = cw / cw.sum()
                portfolio_trading_costs[date] = float(np.dot(cw, country_trading_costs[valid_cost_mask]))

        portfolio_returns = portfolio_returns.dropna()
        portfolio_trading_costs = portfolio_trading_costs.dropna()
        if portfolio_returns.empty:
            continue
        aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index)
        valid_idx = aligned_benchmark.notna()
        if valid_idx.any():
            monthly_net_returns[feature] = portfolio_returns[valid_idx] - aligned_benchmark[valid_idx]
        if not portfolio_trading_costs.empty:
            monthly_trading_costs[feature] = portfolio_trading_costs
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(features)} features...")

    return monthly_net_returns, monthly_trading_costs


def save_net_returns_to_excel(net_returns: Dict[str, pd.Series], optimizer_path: str, t60_path: str):
    net_df = pd.DataFrame(net_returns)
    print(f"\nNet returns shape: {net_df.shape}")

    filled = net_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    next_month = filled.index[-1] + pd.DateOffset(months=1)
    filled.loc[next_month] = np.nan
    t60 = filled.shift(1).rolling(60, min_periods=1).mean() * 100

    with pd.ExcelWriter(t60_path, engine="xlsxwriter") as writer:
        t60.to_excel(writer, sheet_name="T60", index_label="Date")
        wb, ws = writer.book, writer.sheets["T60"]
        ws.set_column(0, 0, 15, wb.add_format({"num_format": "dd-mmm-yyyy"}))
        ws.set_column(1, len(t60.columns), 12, wb.add_format({"num_format": "0.0000"}))
    print(f"{t60_path} saved.")

    net_df = net_df.apply(lambda row: row.fillna(row.mean()), axis=1) * 100
    net_df.sort_index(inplace=True)
    net_df.to_excel(optimizer_path, sheet_name="Monthly_Net_Returns", index_label="Date")
    print(f"{optimizer_path} saved.")


def save_trading_costs_to_excel(trading_costs: Dict[str, pd.Series], output_path: str):
    trading_cost_df = pd.DataFrame(trading_costs).sort_index()
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        trading_cost_df.to_excel(writer, sheet_name="Trading_Costs", index_label="Date")
        workbook = writer.book
        worksheet = writer.sheets["Trading_Costs"]
        date_format = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        number_format = workbook.add_format({"num_format": "0.0000"})
        worksheet.set_column(0, 0, 15, date_format)
        worksheet.set_column(1, len(trading_cost_df.columns), 12, number_format)
    print(f"{output_path} saved.")


def main():
    print("Loading data …")
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()

    bench = pd.read_excel(BENCHMARK_PATH, sheet_name="Benchmarks", index_col=0)
    bench.index = pd.to_datetime(bench.index, errors="coerce").to_period("M").to_timestamp()
    benchmark_returns = bench["equal_weight"]

    trading_cost_df = pd.read_excel(TRADING_COST_PATH, sheet_name="jjunk")
    trading_costs = pd.Series(
        pd.to_numeric(trading_cost_df["Trading Cost"], errors="coerce").values,
        index=trading_cost_df["Country"],
    )
    trading_costs = trading_costs[~trading_costs.index.duplicated(keep="first")]

    features = sorted(set(data["variable"]) - {"1MRet"})
    net_returns, monthly_trading_costs = analyze_portfolios(
        data, features, benchmark_returns, trading_costs
    )
    save_net_returns_to_excel(net_returns, OUTPUT_OPTIMIZER, OUTPUT_T60)
    save_trading_costs_to_excel(monthly_trading_costs, OUTPUT_TRADING_COST)
    print("Done.")


if __name__ == "__main__":
    main()
