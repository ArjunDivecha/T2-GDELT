"""
=============================================================================
SCRIPT: Step Eleven GDELT Compare Strategies.py
=============================================================================

Compares country-weight portfolios (from Step Eight outputs) against the equal-weight
benchmark. Missing optional files (e.g. T2_Final_Country_Weights.xlsx) are skipped.

INPUT:
- Portfolio_Data.xlsx
- GDELT_Final_Country_Weights.xlsx (required)
- T2_Final_Country_Weights.xlsx (optional, for side-by-side)

OUTPUT:
- GDELT_Compare_Strategies.pdf
- GDELT_Compare_Return_Results.xlsx

VERSION: 2.0 — standalone (no external module dependencies)
LAST UPDATED: 2026-04-08
USAGE: python "Step Eleven GDELT Compare Strategies.py"
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# FILE PATHS
# ===============================
PORTFOLIO_DATA = 'Portfolio_Data.xlsx'
WEIGHT_FILES = [
    ('GDELT', 'GDELT_Final_Country_Weights.xlsx'),
    ('T2', 'T2_Final_Country_Weights.xlsx'),
]
OUTPUT_PDF = 'GDELT_Compare_Strategies.pdf'
OUTPUT_XLSX = 'GDELT_Compare_Return_Results.xlsx'


def load_strategy_returns(weights_file, returns_df, label):
    """Calculate portfolio returns for a given weight file."""
    try:
        wdf = pd.read_excel(weights_file, sheet_name='All Periods', index_col=0)
    except FileNotFoundError:
        print(f"  Skipping {label}: {weights_file} not found")
        return None
    wdf.index = pd.to_datetime(wdf.index).to_period('M').to_timestamp()
    common = sorted(set(wdf.index).intersection(set(returns_df.index)))
    if len(common) > 1:
        common = common[:-1]
    rets = []
    for date in common:
        w = wdf.loc[date]
        r = returns_df.loc[date]
        cc = w.index.intersection(r.index)
        rets.append(w[cc].fillna(0).mul(r[cc].fillna(0)).sum())
    return pd.Series(rets, index=common, name=label)


def main():
    print("Loading portfolio data...")
    returns_df = pd.read_excel(PORTFOLIO_DATA, sheet_name='Returns', index_col=0)
    returns_df.index = pd.to_datetime(returns_df.index).to_period('M').to_timestamp()

    benchmark_df = pd.read_excel(PORTFOLIO_DATA, sheet_name='Benchmarks', index_col=0)
    benchmark_df.index = pd.to_datetime(benchmark_df.index).to_period('M').to_timestamp()

    strategies = {}
    for label, wf in WEIGHT_FILES:
        print(f"Processing {label}...")
        s = load_strategy_returns(wf, returns_df, label)
        if s is not None:
            strategies[label] = s

    if not strategies:
        print("Error: No strategy weight files found.")
        return

    # Common dates across all strategies
    all_dates = None
    for s in strategies.values():
        if all_dates is None:
            all_dates = set(s.index)
        else:
            all_dates = all_dates.intersection(set(s.index))
    all_dates = sorted(all_dates)

    results = pd.DataFrame(index=all_dates)
    for label, s in strategies.items():
        results[label] = s.loc[all_dates].values
    results['Equal Weight'] = benchmark_df.loc[all_dates, 'equal_weight'].values

    # Cumulative returns
    cum = (1 + results).cumprod()

    # Stats
    def calc_stats(r):
        ann_ret = r.mean() * 12 * 100
        ann_vol = r.std() * np.sqrt(12) * 100
        sharpe = (r.mean() * 12) / (r.std() * np.sqrt(12)) if r.std() > 0 else 0
        max_dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100
        hit = (r > 0).mean() * 100
        return pd.Series({
            'Annual Return (%)': ann_ret,
            'Annual Vol (%)': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_dd,
            'Hit Rate (%)': hit,
        })

    stats = results.apply(calc_stats)

    # PDF
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    for col in cum.columns:
        ax.plot(cum.index, cum[col], label=col, linewidth=2)
    ax.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col in results.columns:
        if col != 'Equal Weight':
            net = results[col] - results['Equal Weight']
            cum_net = (1 + net).cumprod()
            ax.plot(cum_net.index, cum_net, label=f'{col} Net', linewidth=2)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Cumulative Net Returns vs Equal Weight', fontsize=16, fontweight='bold')
    ax.set_ylabel('Growth of $1 (active)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_PDF}")

    # Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine='xlsxwriter') as writer:
        results.to_excel(writer, sheet_name='Monthly Returns')
        cum.to_excel(writer, sheet_name='Cumulative Returns')
        stats.to_excel(writer, sheet_name='Statistics')

        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        for sn in ['Monthly Returns', 'Cumulative Returns']:
            ws = writer.sheets[sn]
            ws.set_column(0, 0, 15, date_format)

    print(f"Saved {OUTPUT_XLSX}")
    print("\nStrategy Statistics:")
    print(stats.round(2).to_string())


if __name__ == "__main__":
    main()
