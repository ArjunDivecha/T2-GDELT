"""
=============================================================================
SCRIPT NAME: Step Eight GDELT Write Country Weights.py
=============================================================================

DESCRIPTION:
    Converts factor-level optimised weights into country-level portfolio
    weights using a centralised utility (calculate_country_weights_from_factors
    from gdelt_country_factor_transform). For each date, it loads the
    factor net weights from GDELT_rolling_window_weights.xlsx, retrieves
    the next-month (t+1) factor exposures from GDELT_Factors_MasterCSV.csv
    to match Step Five's timing convention, and computes country weights.
    Results are saved to GDELT_Final_Country_Weights.xlsx with summary
    statistics and latest-weight sheets. A separate GDELT_Country_Final.xlsx
    is written aligned to the T2 Master.xlsx country order.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_rolling_window_weights.xlsx
        Factor net weights per date from Step Five. Read from sheet
        'Net_Weights' (or first sheet).
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Factors_MasterCSV.csv
        Long/tidy format: columns date, country, variable, value. Contains
        factor exposures used to map factor weights to country weights.
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/T2 Master.xlsx
        Reference file whose column headers define the country ordering
        for the final output file.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Final_Country_Weights.xlsx
        Multi-sheet workbook: All Periods (full weight matrix), Summary
        Statistics (mean, std, min, max per country), Latest Weights
        (most recent non-zero date snapshot).
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Country_Final.xlsx
        Single-sheet workbook with non-zero country weights sorted by
        weight descending, aligned to T2 Master country order, plus a
        TOTAL row.

VERSION: 2.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - numpy
    - tqdm
    - xlsxwriter
    - gdelt_country_factor_transform (local module)

USAGE:
    python "Step Eight GDELT Write Country Weights.py"

NOTES:
    - Uses t+1 exposures to align with Step Five's timing convention:
      weights(t) x factor_returns(t+1).
    - GDELT factors are not inverted — sign-flip is handled upstream in
      Step Two.
    - Weight sums are validated; warning printed if total deviates from 1.0.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from gdelt_country_factor_transform import calculate_country_weights_from_factors

# ===============================
# GDELT FILE PATHS
# ===============================
weights_file = "GDELT_rolling_window_weights.xlsx"
factor_file = "GDELT_Factors_MasterCSV.csv"
OUTPUT_FILE = "GDELT_Final_Country_Weights.xlsx"
COUNTRY_FINAL_FILE = "GDELT_Country_Final.xlsx"

# ===============================
# LIQUIDITY (ADV) POSITION CAP  -- shared utility
# ===============================
# At small AUM the dominant trading cost is market impact in thin single-country
# ETFs (Denmark/EDEN etc.). The factor model is blind to liquidity, so it can
# size positions you cannot trade. After building the country weights we cap each
# name so a full rotation is at most LIQ_MAXPART of one day's $ADV, then water-fill
# back to sum=1. Validated in the classic T2 repo to add ~+1.2%/yr net at $7M
# (raises gross AND cuts impact cost); also basic risk management. Same 34-ETF
# universe as the classic repo, so the same IBKR_Liquidity.xlsx ADV data applies.
from step_liquidity_cap import load_adv, apply_liquidity_cap

APPLY_LIQUIDITY_CAP = True          # set False to restore the pre-cap behavior
LIQ_MAXPART = 0.20                  # full rotation <= 20% of one day's ADV
LIQ_AUM = 7_000_000                 # portfolio value (drives the dollar cap)
LIQ_PATH = "Experiments Deep Dive/IBKR_Liquidity.xlsx"  # per-ETF $ADV cache

# ===============================
# DATA LOADING
# ===============================

print("Loading data...")
try:
    feature_weights_df = pd.read_excel(weights_file, sheet_name='Net_Weights', index_col=0)
except Exception:
    feature_weights_df = pd.read_excel(weights_file, index_col=0)

factor_df = pd.read_csv(factor_file)
factor_df['date'] = pd.to_datetime(factor_df['date'])

# GDELT factors are NOT inverted — sign-flip handled upstream in Step Two
INVERTED_FEATURES = set()

all_countries = factor_df['country'].unique()
all_factor_dates = sorted(factor_df['date'].unique())
all_dates = list(feature_weights_df.index)
all_weights = pd.DataFrame(index=all_dates, columns=all_countries).fillna(0.0)

# Group by date for efficient lookup
by_date = factor_df.groupby('date')

# ===============================
# WEIGHT CALCULATION (using centralized utility)
# ===============================

print("\nProcessing all dates using centralized utility...")
print("Using t+1 exposures to match Step 5 timing...")

for i, date in enumerate(tqdm(all_dates)):
    date_dt = pd.to_datetime(date)
    
    if date_dt not in feature_weights_df.index:
        continue
    
    w = feature_weights_df.loc[date_dt].astype(float)
    w = w[w.abs() > 1e-10]
    if w.empty:
        continue
    
    # Get next month's exposures (t+1) to match Step 5's use of factor_returns(t+1)
    # Step 5 uses weights(t) × factor_returns(t+1)
    # factor_returns(t+1) were calculated using country_selection(t+1) from exposures(t+1)
    next_month_idx = all_dates.index(date) + 1 if date in all_dates else None
    if next_month_idx is None or next_month_idx >= len(all_dates):
        continue
    
    next_date = pd.to_datetime(all_dates[next_month_idx])
    
    try:
        slice_df = by_date.get_group(next_date)
    except KeyError:
        continue
    
    # Pivot to get countries × factors matrix
    pivot = slice_df.pivot(index='country', columns='variable', values='value')
    
    # Calculate country weights using centralized utility
    country_weights = calculate_country_weights_from_factors(w, pivot, next_date)
    
    if country_weights is None or country_weights.empty:
        continue
    
    # Store weights at original date t (weights are applied at t, but based on t+1 exposures)
    all_weights.loc[date, country_weights.index] = country_weights.values

# ===============================
# LIQUIDITY (ADV) POSITION CAP
# ===============================
# Cap each country so a full rotation stays within LIQ_MAXPART of one day's ADV,
# then water-fill back to sum=1. Deliberately breaks the exact factor==country
# return identity (the factor model is blind to liquidity); intentional and
# net-positive at small AUM. Disable with APPLY_LIQUIDITY_CAP = False.
if APPLY_LIQUIDITY_CAP:
    print(f"\nApplying liquidity cap (MAXPART={LIQ_MAXPART:.0%}, AUM=${LIQ_AUM:,.0f})...")
    adv = load_adv(LIQ_PATH, list(all_weights.columns))
    to_before = 0.5 * all_weights.diff().abs().sum(axis=1).dropna().mean()
    all_weights, cap_report = apply_liquidity_cap(all_weights, adv, LIQ_AUM, LIQ_MAXPART)
    to_after = 0.5 * all_weights.diff().abs().sum(axis=1).dropna().mean()
    n_capped = int((cap_report["Cap_%"] < 99.9).sum())
    print(f"  {n_capped} names capped; one-way turnover {to_before*100:.1f}% -> "
          f"{to_after*100:.1f}%/mo (raw turnover may rise; COST falls)")
    print(cap_report[cap_report["Bind_Freq_%"] > 0]
          [["ADV_USD", "Cap_%", "Bind_Freq_%"]].round(2).head(8).to_string())

# ===============================
# VALIDATION
# ===============================

weight_sums = all_weights.sum(axis=1)
print("\nWeight sum statistics:")
print(weight_sums.describe())

# ===============================
# SAVE RESULTS
# ===============================

print("\nSaving results...")
with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
    all_weights.to_excel(writer, sheet_name='All Periods')

    summary_stats = pd.DataFrame({
        'Mean Weight': all_weights.mean(),
        'Std Dev': all_weights.std(),
        'Min Weight': all_weights.min(),
        'Max Weight': all_weights.max(),
        'Days with Weight': (all_weights.abs() > 0).sum()
    }).sort_values('Mean Weight', ascending=False)
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')

    non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
    if len(non_zero_dates) > 0:
        latest_valid_date = non_zero_dates[-1]
        latest_weights = pd.DataFrame({
            'Weight': all_weights.loc[latest_valid_date],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights.abs() > 0).sum(),
            'Latest Date': pd.Series([latest_valid_date] * len(all_weights.columns),
                                     index=all_weights.columns)
        }).sort_values('Weight', ascending=False)
        print(f"\nUsing {latest_valid_date} as the latest valid date with non-zero weights")
    else:
        latest_weights = pd.DataFrame({
            'Weight': all_weights.iloc[-1],
            'Average Weight': all_weights.mean(),
            'Days with Weight': (all_weights.abs() > 0).sum()
        }).sort_values('Weight', ascending=False)
    latest_weights.to_excel(writer, sheet_name='Latest Weights')

print(f"Results saved to {OUTPUT_FILE}")

print("\nTop 10 countries by average weight:")
print(summary_stats.head(10))

# ===============================
# WRITE COUNTRY FINAL
# ===============================


def write_final_country_weights():
    print(f"\nWriting country weights to {COUNTRY_FINAL_FILE}...")

    non_zero_dates = all_weights.index[all_weights.sum(axis=1) > 0]
    if len(non_zero_dates) == 0:
        print("Error: No date with non-zero weights found")
        return

    latest_valid_date = non_zero_dates[-1]
    print(f"Using weights from latest date: {latest_valid_date}")
    latest_wts = all_weights.loc[latest_valid_date]
    country_weight_dict = latest_wts.to_dict()
    print(f"Found {len(country_weight_dict)} countries in latest weights")
    print(f"Total net weight: {sum(country_weight_dict.values()):.4f}")

    try:
        print("Reading original country order from T2 Master.xlsx...")
        master_df = pd.read_excel("T2 Master.xlsx")
        country_columns = list(master_df.columns[1:])
        print(f"Found {len(country_columns)} countries in column headers")

        all_countries_list = list(country_columns)
        sorted_weights = pd.DataFrame({
            'Country': all_countries_list,
            'Weight': 0.0
        })

        for country, weight in country_weight_dict.items():
            match_idx = sorted_weights[sorted_weights['Country'].str.lower() == country.lower()].index
            if len(match_idx) > 0:
                sorted_weights.loc[match_idx[0], 'Weight'] = weight
            else:
                print(f"Note: Country '{country}' with weight {weight:.4f} not found in T2 Master.xlsx")
                new_row = pd.DataFrame({'Country': [country], 'Weight': [weight]})
                sorted_weights = pd.concat([sorted_weights, new_row], ignore_index=True)

    except Exception as e:
        print(f"Error reading original country order: {e}")
        sorted_weights = pd.DataFrame(list(country_weight_dict.items()),
                                      columns=['Country', 'Weight'])

    # Simplified output - just country weights without per-factor breakdown
    final_df = sorted_weights.copy()
    final_df = final_df[final_df['Weight'] != 0]
    final_df = final_df.sort_values('Weight', ascending=False)

    total_weight = final_df['Weight'].sum()
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: total country weight = {total_weight:.4f}, should be 1.0")

    with pd.ExcelWriter(COUNTRY_FINAL_FILE, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='Country Weights', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Country Weights']
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'bg_color': '#D9D9D9', 'border': 1
        })
        pct_format = workbook.add_format({'num_format': '0.00%'})

        worksheet.set_column(0, 0, 15)
        worksheet.set_column(1, 1, 12, pct_format)

        for col_num, value in enumerate(final_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        last_row = len(final_df) + 1
        bold_format = workbook.add_format({'bold': True})
        total_format = workbook.add_format({'bold': True, 'num_format': '0.00%'})
        worksheet.write(last_row, 0, 'TOTAL', bold_format)
        worksheet.write(last_row, 1, total_weight, total_format)

    print(f"Final weights saved to {COUNTRY_FINAL_FILE}")


write_final_country_weights()
