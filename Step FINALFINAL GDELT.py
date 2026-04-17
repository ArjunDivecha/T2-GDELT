"""
=============================================================================
SCRIPT NAME: Step FINALFINAL GDELT.py
=============================================================================

INPUT FILES:
- GDELT_Country_Alphas.xlsx (from Step Six GDELT)
    - Sheet: 'Country_Scores'
    - Contains country alpha scores across dates. Each row is a date; columns are countries.
- GDELT_Optimized_Country_Weights.xlsx (from Step Fourteen GDELT)
    - Sheet: 'Latest_Weights'
    - Contains latest optimized country weights. Columns: 'Country', 'Weight' (or similar).

OUTPUT FILES:
- GDELT_FINAL_T60.xlsx
    - Sheet: 'Latest_Country_Alpha_Weights' (columns: Country, Country Alpha, Country Weight)

VERSION: 1.0
LAST UPDATED: 2026-04-17
AUTHOR: Antigravity

DESCRIPTION:
This script extracts the most recent country alpha scores from the last row of the 'Country_Scores' sheet in GDELT_Country_Alphas.xlsx and the latest weights from the 'Latest_Weights' sheet in GDELT_Optimized_Country_Weights.xlsx, merges them by country, and writes a single Excel sheet with columns: Country, Country Alpha, Country Weight. This is the GDELT-specific version of Step FINALFINAL.py.

=============================================================================
"""

import pandas as pd
import os

# --- File names (GDELT track) ---
ALPHAS_FILE = 'GDELT_Country_Alphas.xlsx'
WEIGHTS_FILE = 'GDELT_Optimized_Country_Weights.xlsx'
OUTPUT_FILE = 'GDELT_FINAL_T60.xlsx'

def main():
    # Check input files exist
    for f in [ALPHAS_FILE, WEIGHTS_FILE]:
        if not os.path.exists(f):
            print(f"Error: Required file '{f}' not found in current directory.")
            return

    print(f"Loading latest alphas from {ALPHAS_FILE}...")
    # Read the last row of country alphas (latest scores)
    alphas_df = pd.read_excel(ALPHAS_FILE, sheet_name='Country_Scores', index_col=0)
    latest_scores = alphas_df.iloc[-1]
    latest_scores = latest_scores.reset_index()
    latest_scores.columns = ['Country', 'Country Alpha']
    
    latest_date = alphas_df.index[-1]
    print(f"  Latest Alpha Date: {latest_date}")

    print(f"Loading latest weights from {WEIGHTS_FILE}...")
    # Read the latest weights
    weights_df = pd.read_excel(WEIGHTS_FILE, sheet_name='Latest_Weights')
    
    # Try to find the right column for weights
    weight_col = 'Weight' if 'Weight' in weights_df.columns else weights_df.columns[-1]
    weights_df = weights_df[['Country', weight_col]].copy()
    weights_df.columns = ['Country', 'Country Weight']

    # Merge on Country
    merged = pd.merge(latest_scores, weights_df, on='Country', how='inner')

    # Save to Excel (single sheet)
    print(f"Writing results to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        merged.to_excel(writer, sheet_name='Latest_Country_Alpha_Weights', index=False)
        worksheet = writer.sheets['Latest_Country_Alpha_Weights']
        for idx, col in enumerate(merged.columns):
            # Calculate column width
            val_len = merged[col].astype(str).map(len).max() if not merged.empty else 0
            max_len = max(val_len, len(str(col)))
            worksheet.set_column(idx, idx, max_len + 2)
            
    print(f"Successfully created '{OUTPUT_FILE}' with columns: Country, Country Alpha, Country Weight.")


if __name__ == "__main__":
    main()
