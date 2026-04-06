#!/usr/bin/env python3
"""
=============================================================================
MODULE: step_three_regression_utils.py
=============================================================================
Helper module for Step Three Top20 Portfolios Fast.py.

PURPOSE:
  Provides cross-sectional regression-based factor return functions.
  Imported by Step Three Top20 Portfolios Fast.py — not run directly.

HOW FACTOR RETURNS ARE FORMED (REGRESSION METHOD):
  Each month, for each factor, we run a univariate OLS regression:

      1MRet(country) = alpha + beta * factor_score(country) + error

  across all countries that have both a valid factor score and a valid
  one-month return.  The slope coefficient (beta) is that factor's return
  for that month.  A positive beta means countries with higher factor scores
  tended to earn higher returns — i.e., the factor "worked" that month.
  We collect one beta per month per factor to form a time series, then
  evaluate it using the same performance metrics as the sort-based approach.

INPUT  (consumed by functions below, not read from disk here):
  - data             : long-format DataFrame (date, country, variable, value)
  - features         : list of factor variable names to regress
  - benchmark_returns: pd.Series of monthly equal-weight benchmark returns

OUTPUT (returned by functions, saved by caller):
  - monthly_betas    : dict[factor_name -> pd.Series of monthly betas]
  - results_df       : performance summary DataFrame

VERSION: 1.0
LAST UPDATED: 2026-04-06
=============================================================================
"""

from typing import Dict, Tuple

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------
# Regression-based factor returns (Fama-MacBeth style, univariate)
# ------------------------------------------------------------------
def analyze_portfolios_regression(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
    calculate_performance_metrics,          # passed in from main module
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    For each factor and each month, regress country returns on that factor's
    cross-sectional scores using OLS:

        1MRet(country) = alpha + beta * factor_score(country) + error

    The slope (beta) becomes that factor's return for that month.
    A positive beta means higher-scored countries earned higher returns.

    Parameters
    ----------
    data                        : long-format tidy DataFrame
    features                    : list of factor names
    benchmark_returns           : equal-weight benchmark return series
    calculate_performance_metrics: function imported from the main module

    Returns
    -------
    monthly_betas : dict[str, Series]   — one beta per date per factor
    results_df    : DataFrame           — performance summary table
    """
    results = []
    monthly_betas: Dict[str, pd.Series] = {}

    dates = sorted(data["date"].unique())

    # Build a wide pivot: rows = dates, cols = countries, values = 1MRet
    print("Building returns pivot for regression...")
    ret_wide = (
        data[data["variable"] == "1MRet"]
        .pivot_table(index="date", columns="country", values="value", aggfunc="first")
    )
    ret_wide.index = pd.to_datetime(ret_wide.index)

    print(f"Running regressions for {len(features)} features...")
    for feat_idx, feature in enumerate(features):
        if (feat_idx + 1) % 20 == 0:
            print(f"  Regressed {feat_idx + 1}/{len(features)} features...")

        # Wide pivot for this factor: rows = dates, cols = countries
        feat_wide = (
            data[data["variable"] == feature]
            .pivot_table(index="date", columns="country", values="value", aggfunc="first")
        )
        feat_wide.index = pd.to_datetime(feat_wide.index)

        beta_series = pd.Series(index=pd.to_datetime(dates), dtype=float)

        for date in pd.to_datetime(dates):
            # Pull factor scores and returns for this date
            if date not in feat_wide.index or date not in ret_wide.index:
                continue

            x = feat_wide.loc[date]
            y = ret_wide.loc[date]

            # Keep only countries where both factor score and return are valid
            valid = x.notna() & y.notna()
            x_clean = x[valid].values.astype(float)
            y_clean = y[valid].values.astype(float)

            # Need at least 3 observations to fit a meaningful line
            if len(x_clean) < 3:
                continue

            # OLS: 1MRet = alpha + beta * factor_score
            slope, _intercept, _r, _p, _se = stats.linregress(x_clean, y_clean)
            beta_series[date] = slope

        beta_series = beta_series.reindex(benchmark_returns.index)
        monthly_betas[feature] = beta_series

        metrics = calculate_performance_metrics(beta_series, benchmark_returns)
        results.append({"Feature": feature, **metrics})

    metric_cols = [
        "Feature",
        "Avg Excess Return (%)",
        "Volatility (%)",
        "Information Ratio",
        "Maximum Drawdown (%)",
        "Hit Ratio (%)",
        "Skewness",
        "Kurtosis",
        "Beta",
        "Tracking Error (%)",
        "Calmar Ratio",
    ]
    results_df = (
        pd.DataFrame(results)[metric_cols] if results else pd.DataFrame()
    )
    return monthly_betas, results_df


# ------------------------------------------------------------------
# Visualisation for regression betas
# ------------------------------------------------------------------
def create_regression_charts(
    betas_dict: Dict[str, pd.Series],
    benchmark_returns: pd.Series,
    output_path: str,
) -> None:
    """
    Plot cumulative factor beta (regression slope) minus benchmark return
    over time, one panel per factor — mirroring the sort-based charts.
    """
    n_features, n_cols = len(betas_dict), 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    colors = plt.cm.tab10.colors

    for i, (feature, betas) in enumerate(betas_dict.items()):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        excess = betas - benchmark_returns
        first_valid = excess.first_valid_index()
        if first_valid is not None:
            cum = excess[first_valid:].cumsum() * 100
            ax.plot(
                cum.index, cum,
                label="Cumul. β excess (bps)",
                linewidth=1.5,
                color=colors[1],
            )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(feature, fontsize=12, weight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
