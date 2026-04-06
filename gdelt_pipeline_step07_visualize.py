"""
Step Seven — factor weight charts (shared for GDELT / Combined tracks).

VERSION: 1.0  LAST UPDATED: 2026-04-02
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from gdelt_track_config import PostStep5TrackConfig

warnings.filterwarnings("ignore")


def run_visualize_factor_weights(cfg: PostStep5TrackConfig) -> None:
    wf = cfg.rolling_weights_xlsx
    print(f"Loading weights: {wf}")
    try:
        weights_df = pd.read_excel(wf, sheet_name="Net_Weights", index_col=0)
    except Exception:
        weights_df = pd.read_excel(wf, index_col=0)
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_df = weights_df.sort_index()

    max_weights = weights_df.abs().max()
    threshold = 0.05
    significant_factors = max_weights[max_weights > threshold].index.tolist()
    if not significant_factors:
        print("Warning: fallback to top 5 factors by mean |weight|")
        significant_factors = weights_df.abs().mean().nlargest(5).index.tolist()

    significant_weights = weights_df[significant_factors]
    latest_date = significant_weights.index.max()
    latest_w = significant_weights.loc[latest_date]
    order = latest_w.abs().sort_values(ascending=False).index.tolist()
    significant_weights = significant_weights[order]
    color_map = {
        f: ("tab:blue" if latest_w[f] >= 0 else "tab:red") for f in significant_weights.columns
    }

    disp = latest_w[order]
    disp = disp[disp.abs() > 0.001]
    plt.figure(figsize=(12, 8))
    bar_colors = [color_map[f] for f in disp.index]
    bars = plt.barh(disp.index, disp.values, color=bar_colors, alpha=0.8)
    plt.xlabel("Portfolio Weight (%)", fontsize=14)
    plt.title(
        f'Factor Allocation for {latest_date.strftime("%Y-%m-%d")}',
        fontsize=18,
        pad=15,
        fontweight="bold",
    )
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: "{:.0%}".format(x)))
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.axvline(0, color="gray", linewidth=1, alpha=0.6)
    for bar in bars:
        width = bar.get_width()
        x = width + (0.005 if width >= 0 else -0.005)
        ha = "left" if width >= 0 else "right"
        plt.text(
            x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{width:.1%}",
            ha=ha,
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    print(f"Saving {cfg.latest_factor_alloc_pdf}")
    plt.savefig(cfg.latest_factor_alloc_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    n_factors = len(significant_weights.columns)
    ncols = int(np.ceil(np.sqrt(n_factors)))
    nrows = int(np.ceil(n_factors / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), sharex=True, sharey=True
    )
    axes = axes.flatten()
    i = -1
    for i, factor in enumerate(significant_weights.columns):
        ax = axes[i]
        series = significant_weights[factor]
        color = "tab:blue" if series.iloc[-1] >= 0 else "tab:red"
        ax.plot(significant_weights.index, series, color=color, linewidth=2)
        ax.set_title(factor, fontsize=12, pad=5)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="gray", linewidth=1, alpha=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Factor Allocation Through Time (Individual Factors)", fontsize=20, y=1.02, fontweight="bold")
    max_abs = significant_weights.abs().max().max()
    plt.ylim(-max_abs * 1.1, max_abs * 1.1)
    plt.xlim(significant_weights.index.min(), significant_weights.index.max())
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    print(f"Saving {cfg.factor_grid_pdf}")
    plt.savefig(cfg.factor_grid_pdf, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Step Seven visualization complete.")
