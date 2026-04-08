"""
=============================================================================
SCRIPT: Step Seven GDELT Visualize Factor Weights.py
=============================================================================

INPUT:  GDELT_rolling_window_weights.xlsx (from Step Five GDELT FAST)
OUTPUT: GDELT_latest_factor_allocation.pdf, GDELT_factor_allocation_grid.pdf

VERSION: 2.0 — standalone (no external module dependencies)
LAST UPDATED: 2026-04-08
USAGE: python "Step Seven GDELT Visualize Factor Weights.py"
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# GDELT file paths
# ---------------------------------------------------------------------------
WEIGHTS_FILE = 'GDELT_rolling_window_weights.xlsx'
OUTPUT_LATEST = 'GDELT_latest_factor_allocation.pdf'
OUTPUT_GRID = 'GDELT_factor_allocation_grid.pdf'


def main():
    print("Starting GDELT Factor Weights Visualization...")

    # Load weights data
    print(f"Loading weight data from {WEIGHTS_FILE}...")
    try:
        weights_df = pd.read_excel(WEIGHTS_FILE, sheet_name='Net_Weights', index_col=0)
    except Exception:
        weights_df = pd.read_excel(WEIGHTS_FILE, index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_df = weights_df.sort_index()

    # Factor selection: max ABS weight > 5%
    max_weights = weights_df.abs().max()
    threshold = 0.05
    significant_factors = max_weights[max_weights > threshold].index.tolist()

    if not significant_factors:
        print(f"Warning: No factors exceeded {threshold:.0%}. Selecting top 5 by avg ABS weight.")
        significant_factors = weights_df.abs().mean().nlargest(5).index.tolist()

    print(f"Identified {len(significant_factors)} factors with max ABS weight > {threshold:.0%}")
    significant_weights = weights_df[significant_factors]

    # Sort by latest weight
    latest_date = significant_weights.index.max()
    latest_weights_all = significant_weights.loc[latest_date]
    sorted_factors = latest_weights_all.abs().sort_values(ascending=False).index.tolist()
    significant_weights = significant_weights[sorted_factors]

    color_map = {f: ('tab:blue' if latest_weights_all[f] >= 0 else 'tab:red')
                 for f in significant_weights.columns}

    # --- Chart 1: Latest Factor Allocation ---
    print("Creating latest factor allocation chart...")
    latest_display = latest_weights_all[sorted_factors]
    latest_display = latest_display[latest_display.abs() > 0.001]

    plt.figure(figsize=(12, 8))
    bar_colors = [color_map[f] for f in latest_display.index]
    bars = plt.barh(latest_display.index, latest_display.values, color=bar_colors, alpha=0.8)
    plt.xlabel('Portfolio Weight (%)', fontsize=14)
    plt.title(f'GDELT Factor Allocation for {latest_date.strftime("%Y-%m-%d")}',
              fontsize=18, pad=15, fontweight='bold')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.xticks(fontsize=12)
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--', axis='x')
    plt.axvline(0, color='gray', linewidth=1, alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        x = width + (0.005 if width >= 0 else -0.005)
        ha = 'left' if width >= 0 else 'right'
        plt.text(x, bar.get_y() + bar.get_height()/2., f'{width:.1%}',
                 ha=ha, va='center', fontsize=10)

    plt.tight_layout()
    print(f"Saving latest allocation chart to {OUTPUT_LATEST}...")
    plt.savefig(OUTPUT_LATEST, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Chart 2: Factor Allocation Through Time (Small Multiples Grid) ---
    print("Creating factor allocation over time chart (small multiples grid)...")

    n_factors = len(significant_weights.columns)
    ncols = int(np.ceil(np.sqrt(n_factors)))
    nrows = int(np.ceil(n_factors / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, factor in enumerate(significant_weights.columns):
        ax = axes[i]
        series = significant_weights[factor]
        color = 'tab:blue' if series.iloc[-1] >= 0 else 'tab:red'
        ax.plot(significant_weights.index, series, color=color, linewidth=2, label=factor)
        ax.set_title(factor, fontsize=12, pad=5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='gray', linewidth=1, alpha=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10, rotation=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(4))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('GDELT Factor Allocation Through Time', fontsize=20, y=1.02, fontweight='bold')
    fig.text(0.5, 0.01, 'Date', ha='center', va='center', fontsize=14)
    fig.text(0.01, 0.5, 'Portfolio Weight (%)', ha='center', va='center',
             rotation='vertical', fontsize=14)

    max_abs = significant_weights.abs().max().max()
    plt.ylim(-max_abs * 1.1, max_abs * 1.1)
    plt.xlim(significant_weights.index.min(), significant_weights.index.max())

    start_date = significant_weights.index.min().strftime('%b %Y')
    end_date = significant_weights.index.max().strftime('%b %Y')
    summary_text = (f"Dataset: {start_date} to {end_date}\n"
                    f"Factors with max ABS weight > {threshold:.0%}")
    fig.text(0.99, 0.01, summary_text, fontsize=10, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    print(f"Saving visualization to {OUTPUT_GRID}...")
    plt.savefig(OUTPUT_GRID, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

    print("Visualization complete!")


if __name__ == "__main__":
    main()
