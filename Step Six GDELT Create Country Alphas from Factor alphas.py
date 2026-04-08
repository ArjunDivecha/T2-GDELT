#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================================
SCRIPT: Step Six GDELT Create Country Alphas from Factor alphas.py
=============================================================================

INPUT FILES (run from repo root — same folder as script):
- GDELT_T60.xlsx           (from Step Four GDELT)
- GDELT_Top_20_Exposure.csv
- T2 Master.xlsx           (sheet 1MRet — country column order)

OUTPUT:
- GDELT_Country_Alphas.xlsx

PREREQUISITE: Step Five GDELT FAST (rolling weights) is not required; you need
Step Four GDELT T60 + Step Three exposure CSV.

VERSION: 2.0 — standalone (no external module dependencies)
LAST UPDATED: 2026-04-08
USAGE: python "Step Six GDELT Create Country Alphas from Factor alphas.py"
=============================================================================
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GDELT file paths (inlined from former gdelt_track_config)
# ---------------------------------------------------------------------------
T60_FILE = 'GDELT_T60.xlsx'
EXPOSURE_FILE = 'GDELT_Top_20_Exposure.csv'
T2_MASTER_FILE = 'T2 Master.xlsx'
OUTPUT_FILE = 'GDELT_Country_Alphas.xlsx'


def load_data():
    """Load factor alphas, country exposures, and reference country order."""
    print(f"Loading factor alpha data from {T60_FILE}...")
    factor_df = pd.read_excel(T60_FILE, sheet_name=0)
    factor_df = factor_df.rename(columns={factor_df.columns[0]: 'Date'})
    factor_df['Date'] = pd.to_datetime(factor_df['Date'])

    print(f"Loading wide-format country exposure data from {EXPOSURE_FILE}...")
    exposure_df = pd.read_csv(EXPOSURE_FILE)
    exposure_df['Date'] = pd.to_datetime(exposure_df['Date'])

    print(f"Loading country order reference from {T2_MASTER_FILE}...")
    try:
        country_order_df = pd.read_excel(T2_MASTER_FILE, sheet_name='1MRet')
        country_order = country_order_df.columns[1:].tolist()
        print(f"Found {len(country_order)} countries in reference file")
    except Exception as e:
        print(f"Warning: Could not load country order from {T2_MASTER_FILE}: {e}")
        print("Using countries as they appear in the data instead")
        country_order = None
    return factor_df, exposure_df, country_order


def get_available_dates(factor_df, exposure_df):
    """Return sorted dates present in both datasets."""
    common = list(set(factor_df['Date']).intersection(set(exposure_df['Date'])))
    common.sort()
    return common


def process_month(date_obj, factor_df, exposure_df, factor_names, missing_log):
    """Calculate country alphas for a single month."""
    factor_row = factor_df[factor_df['Date'] == date_obj]
    if factor_row.empty:
        return pd.DataFrame()
    factor_row = factor_row.iloc[0]

    date_countries = exposure_df[exposure_df['Date'] == date_obj]
    results = []

    # Precompute factor means for missing-value filling
    factor_means = {}
    for f in factor_names:
        vals = date_countries[f]
        factor_means[f] = vals[vals.notna()].mean()

    for _, row in date_countries.iterrows():
        country = row['Country']
        total = 0.0
        valid_factors = 0
        for f in factor_names:
            exposure = row[f]
            if pd.isna(exposure):
                exposure = factor_means[f]
                missing_log.append({
                    'date': date_obj, 'country': country,
                    'factor': f, 'filled_value': exposure,
                })
            alpha = factor_row[f]
            if pd.isna(alpha) or pd.isna(exposure):
                continue
            total += exposure * alpha
            valid_factors += 1
        results.append({
            'date': date_obj, 'country': country,
            'total_score': total, 'valid_factors': valid_factors,
        })
    return pd.DataFrame(results)


def main():
    start_time = time.time()

    # 1. Load data
    factor_df, exposure_df, country_order = load_data()

    # 2. Align dates and factors
    common_dates = get_available_dates(factor_df, exposure_df)
    print(f"Number of common dates: {len(common_dates)}")

    exposure_factors = [c for c in exposure_df.columns if c not in ['Date', 'Country']]
    factor_file_factors = [c for c in factor_df.columns if c != 'Date']
    factor_names = [f for f in factor_file_factors if f in exposure_factors]
    print(f"Number of factors used: {len(factor_names)}")
    print(f"Date range: {common_dates[0]:%Y-%m-%d} to {common_dates[-1]:%Y-%m-%d}")

    # 3. Process each month
    missing_log = []
    all_results = []
    with tqdm(total=len(common_dates), desc="Processing months") as pbar:
        for date_obj in common_dates:
            result = process_month(date_obj, factor_df, exposure_df, factor_names, missing_log)
            all_results.append(result)
            pbar.update(1)

    combined_results = pd.concat(all_results, ignore_index=True)

    # 4. Pivot to dates × countries
    country_col = 'country'
    if 'country' not in combined_results.columns:
        for col in combined_results.columns:
            if 'country' in col.lower():
                country_col = col
                break

    pivot_table = combined_results.pivot_table(
        index='date', columns=country_col, values='total_score',
    )

    if country_order:
        available = [c for c in country_order if c in pivot_table.columns]
        for c in pivot_table.columns:
            if c not in available:
                available.append(c)
        pivot_table = pivot_table[available]

    # Quality report
    missing_data = pivot_table.isna().sum()
    country_quality = pd.DataFrame({
        'country': missing_data.index,
        'missing_months': missing_data.values,
        'completeness_pct': 100 * (1 - missing_data.values / len(common_dates)),
    }).sort_values('completeness_pct')

    factor_count_pivot = combined_results.pivot_table(
        index='date', columns=country_col, values='valid_factors',
    )

    # 5. Write to Excel
    print(f"Writing results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        pivot_table.to_excel(writer, sheet_name='Country_Scores')
        country_quality.to_excel(writer, sheet_name='Data_Quality', index=False)
        factor_count_pivot.to_excel(writer, sheet_name='Factor_Counts')

        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})

        worksheet = writer.sheets['Country_Scores']
        worksheet.set_column(0, 0, 12, date_format)

        worksheet = writer.sheets['Factor_Counts']
        worksheet.set_column(0, 0, 12, date_format)

    elapsed = time.time() - start_time
    print(f"\nSummary:")
    print(f"- Date range: {common_dates[0]:%Y-%m-%d} to {common_dates[-1]:%Y-%m-%d}")
    print(f"- Months processed: {len(common_dates)}")
    print(f"- Countries: {len(pivot_table.columns)}")
    print(f"- Output: {OUTPUT_FILE}")
    print(f"- Time: {elapsed:.2f}s")

    print("\nPreview (first 5 months, first 5 countries):")
    print(pivot_table[pivot_table.columns[:5]].head())


if __name__ == "__main__":
    main()
