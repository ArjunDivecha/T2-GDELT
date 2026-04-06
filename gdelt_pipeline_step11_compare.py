"""
Step Eleven — compare multiple country-weight portfolios vs equal-weight benchmark.

Loads each ``*_Final_Country_Weights.xlsx`` file listed in the track config (skips
missing files). Uses matplotlib only (no seaborn).

VERSION: 1.0  LAST UPDATED: 2026-04-02
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gdelt_track_config import PostStep5TrackConfig

PORTFOLIO_DATA = "Portfolio_Data.xlsx"


def run_compare_strategies(cfg: PostStep5TrackConfig) -> None:
    portfolio_files = []
    for name in cfg.compare_weight_files:
        p = Path(name)
        if p.is_file():
            portfolio_files.append(p)
        else:
            print(f"Step 11: skip missing weights file {name}")

    if not portfolio_files:
        raise FileNotFoundError(
            "No compare weight files found. Run Step Eight for this track (and optionally T2) first."
        )

    returns_df = pd.read_excel(PORTFOLIO_DATA, sheet_name="Returns", index_col=0)
    benchmark_df = pd.read_excel(PORTFOLIO_DATA, sheet_name="Benchmarks", index_col=0)
    returns_df.index = pd.to_datetime(returns_df.index).to_period("M").to_timestamp()
    benchmark_df.index = pd.to_datetime(benchmark_df.index).to_period("M").to_timestamp()

    all_results = {}
    all_cumulative = {}
    all_net = {}
    all_turnover = {}

    common_dates = returns_df.index
    for pf in portfolio_files:
        wtmp = pd.read_excel(pf, sheet_name="All Periods", index_col=0)
        wtmp.index = pd.to_datetime(wtmp.index).to_period("M").to_timestamp()
        common_dates = common_dates.intersection(wtmp.index)
    common_dates = common_dates.intersection(benchmark_df.index).sort_values()

    if len(common_dates) < 2:
        raise ValueError("Insufficient overlapping dates for comparison.")

    for pf in portfolio_files:
        label = pf.stem
        weights_df = pd.read_excel(pf, sheet_name="All Periods", index_col=0)
        weights_df.index = pd.to_datetime(weights_df.index).to_period("M").to_timestamp()
        weights_df = weights_df.loc[common_dates]
        pf_returns = []
        pf_turnover = []
        prev_weights = None
        for date in common_dates:
            weights = weights_df.loc[date]
            rets = returns_df.loc[date]
            pf_ret = np.nansum(weights * rets)
            pf_returns.append(pf_ret)
            if prev_weights is not None:
                turnover = np.nansum(np.abs(weights - prev_weights)) / 2
            else:
                turnover = np.nan
            pf_turnover.append(turnover)
            prev_weights = weights
        pf_returns = pd.Series(pf_returns, index=common_dates)
        pf_turnover = pd.Series(pf_turnover, index=common_dates)
        eqw = benchmark_df.loc[common_dates, "equal_weight"]
        net_ret = pf_returns - eqw
        all_results[label] = pf_returns
        all_cumulative[label] = (1 + pf_returns).cumprod()
        all_net[label] = net_ret
        all_turnover[label] = pf_turnover

    all_results["Equal Weight"] = benchmark_df.loc[common_dates, "equal_weight"]
    all_cumulative["Equal Weight"] = (1 + benchmark_df.loc[common_dates, "equal_weight"]).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    for label, series in all_cumulative.items():
        axes[0].plot(series.index, series.values, label=label)
    axes[0].set_title("Cumulative Returns")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for label, net in all_net.items():
        axes[1].plot(net.index, (1 + net).cumprod().values, label=label)
    axes[1].axhline(1, color="r", linestyle="--", alpha=0.3)
    axes[1].set_title("Cumulative Net Returns (vs Equal Weight)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    window = 12
    for label, net in all_net.items():
        axes[2].plot(net.index, net.rolling(window).sum().values, label=label)
    axes[2].axhline(0, color="r", linestyle="--", alpha=0.3)
    axes[2].set_title("Rolling 12-Month Net Returns")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.compare_strategies_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {cfg.compare_strategies_pdf}")

    with pd.ExcelWriter(cfg.compare_strategies_xlsx, engine="xlsxwriter") as writer:
        pd.DataFrame(all_results).to_excel(writer, sheet_name="Monthly Return")
        pd.DataFrame(all_cumulative).to_excel(writer, sheet_name="Cumulative Return")
        stats = {}
        for label, returns in all_results.items():
            if label == "Equal Weight":
                continue
            ann_ret = returns.mean() * 12 * 100
            ann_vol = returns.std() * np.sqrt(12) * 100
            sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
            max_dd = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).min() - 1
            avg_turnover = all_turnover[label].mean() if label in all_turnover else np.nan
            stats[label] = {
                "Annual Return (%)": ann_ret,
                "Volatility (%)": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd,
                "Avg Turnover": avg_turnover,
            }
        pd.DataFrame(stats).T.to_excel(writer, sheet_name="Statistics")
        pd.DataFrame(all_net).to_excel(writer, sheet_name="Net Returns")
    print(f"Saved {cfg.compare_strategies_xlsx}")
