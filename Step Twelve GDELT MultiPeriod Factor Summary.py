"""
=============================================================================
SCRIPT NAME: Step Twelve GDELT MultiPeriod Factor Summary.py
=============================================================================

DESCRIPTION:
    Generates a multi-period factor performance summary from the optimizer
    workbook. Reads monthly net returns for each GDELT factor portfolio, then
    calculates annualized mean return, annualized volatility, Sharpe ratio,
    and hit rate for the full sample and trailing 3-month, 6-month, and
    12-month windows. Outputs a single Excel workbook with separate sheets
    per window, each sorted by Sharpe ratio in descending order. Lightweight
    utility — not a full multi-horizon forecast model.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_Optimizer.xlsx
        Monthly net excess returns for each GDELT factor (sheet:
        Monthly_Net_Returns, indexed by date with factor columns.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 GDELT/GDELT_MultiPeriod_Factor_Summary.xlsx
        Excel workbook with sheets: Full_Sample, Trailing_3M, Trailing_6M,
        and Trailing_12M, each containing Mean, Vol, Sharpe, and Hit Rate
        columns sorted by Sharpe ratio.

VERSION: 2.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - numpy
    - xlsxwriter

USAGE:
    python "Step Twelve GDELT MultiPeriod Factor Summary.py"

NOTES:
    - Falls back to first sheet if Monthly_Net_Returns sheet is missing.
    - All input/output files reside in the same directory as the script.
=============================================================================
"""

import pandas as pd
import numpy as np

# ===============================
# FILE PATHS
# ===============================
OPTIMIZER_FILE = 'GDELT_Optimizer.xlsx'
OUTPUT_FILE = 'GDELT_MultiPeriod_Factor_Summary.xlsx'

TRAILING_WINDOWS = [3, 6, 12]


def main():
    print(f"Loading factor returns from {OPTIMIZER_FILE}...")
    try:
        df = pd.read_excel(OPTIMIZER_FILE, sheet_name='Monthly_Net_Returns', index_col=0)
    except Exception:
        df = pd.read_excel(OPTIMIZER_FILE, index_col=0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"Data: {len(df)} months, {len(df.columns)} factors")
    print(f"Date range: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")

    summary_frames = {}

    # Full sample
    full = pd.DataFrame({
        'Mean (ann %)': df.mean() * 12 * 100,
        'Vol (ann %)': df.std() * np.sqrt(12) * 100,
        'Sharpe': (df.mean() * 12) / (df.std() * np.sqrt(12)),
        'Hit Rate (%)': (df > 0).mean() * 100,
    }).sort_values('Sharpe', ascending=False)
    summary_frames['Full_Sample'] = full
    print(f"\nFull sample top 5 by Sharpe:")
    print(full.head().round(2))

    # Trailing windows
    for w in TRAILING_WINDOWS:
        tail = df.iloc[-w:]
        trailing = pd.DataFrame({
            'Mean (ann %)': tail.mean() * 12 * 100,
            'Vol (ann %)': tail.std() * np.sqrt(12) * 100,
            'Sharpe': (tail.mean() * 12) / (tail.std() * np.sqrt(12)),
            'Hit Rate (%)': (tail > 0).mean() * 100,
        }).sort_values('Sharpe', ascending=False)
        sheet_name = f'Trailing_{w}M'
        summary_frames[sheet_name] = trailing
        print(f"\n{sheet_name} top 5 by Sharpe:")
        print(trailing.head().round(2))

    # Write
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        for sheet_name, frame in summary_frames.items():
            frame.to_excel(writer, sheet_name=sheet_name)

    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
