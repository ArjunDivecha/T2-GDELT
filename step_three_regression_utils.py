#!/usr/bin/env python3
"""
=============================================================================
MODULE: step_three_regression_utils.py
=============================================================================
Helper module for Step Three Top20 Portfolios Fast.py.

PURPOSE:
  Provides cross-sectional regression-based factor return functions:
    1. Univariate OLS  — one factor at a time per month
    2. Multivariate LASSO (LassoCV) — all factors simultaneously per month
  Imported by Step Three Top20 Portfolios Fast.py — not run directly.

HOW FACTOR RETURNS ARE FORMED (REGRESSION METHOD):
  Each month, for each factor, we run a simple univariate cross-sectional
  OLS regression of country excess returns on factor scores:

      (1MRet(country) - benchmark_return) = alpha + beta * factor_score(country) + error

  The benchmark_return for a given month is the equal-weight average of
  all country returns that month.  Subtracting it converts raw returns into
  excess returns, so the slope (beta) captures how much *excess* return is
  associated with a unit of factor exposure.
  A positive beta means countries with higher factor scores tended to
  earn higher excess returns — i.e., the factor "worked" that month.
  We collect one beta per month per factor to form a time series, then
  evaluate it as a standalone factor return series.

INPUT  (consumed by functions below, not read from disk here):
  - data              : long-format DataFrame (date, country, variable, value)
  - features          : list of factor variable names to regress
  - benchmark_returns : pd.Series of monthly equal-weight benchmark returns

OUTPUT (returned by functions, saved by caller):
  - monthly_betas : dict[factor_name -> pd.Series of monthly betas]
  - results_df    : performance summary DataFrame

VERSION: 3.1 — removed LASSO; kept univariate OLS + charts
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
# Standalone metrics for a factor return (beta) series
# ------------------------------------------------------------------
def calculate_factor_return_metrics(beta_series: pd.Series) -> Dict[str, float]:
    """
    Compute performance metrics for a factor return series (regression betas).

    The beta series is treated as a standalone return stream — there is no
    benchmark subtraction because the slope from a cross-sectional regression
    is already a pure factor return, not a portfolio return.

    Parameters
    ----------
    beta_series : pd.Series
        Monthly factor returns (one OLS slope per month).

    Returns
    -------
    dict with: Mean Factor Return (% ann.), Volatility (% ann.), t-statistic,
    Sharpe Ratio, Hit Ratio (%), Max Drawdown (%), Skewness, Kurtosis,
    Calmar Ratio.
    """
    clean = pd.to_numeric(beta_series, errors="coerce").dropna()

    zero_metrics = dict.fromkeys(
        [
            "Mean Factor Return (% ann.)",
            "Volatility (% ann.)",
            "t-statistic",
            "Sharpe Ratio",
            "Hit Ratio (%)",
            "Max Drawdown (%)",
            "Skewness",
            "Kurtosis",
            "Calmar Ratio",
        ],
        0,
    )

    if len(clean) < 3:
        return zero_metrics

    mean_m = clean.mean()
    std_m = clean.std()
    n = len(clean)

    mean_ann = mean_m * 12 * 100
    vol_ann = std_m * np.sqrt(12) * 100
    t_stat = mean_m / (std_m / np.sqrt(n)) if std_m > 0 else 0
    sharpe = (mean_m * 12) / (std_m * np.sqrt(12)) if std_m > 0 else 0
    hit = (clean > 0).mean() * 100

    cum = clean.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min() * 100

    skew = clean.skew()
    kurt = clean.kurtosis()
    calmar = -mean_ann / max_dd if max_dd != 0 else 0

    return {
        "Mean Factor Return (% ann.)": round(mean_ann, 2),
        "Volatility (% ann.)": round(vol_ann, 2),
        "t-statistic": round(t_stat, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Hit Ratio (%)": round(hit, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2),
        "Calmar Ratio": round(calmar, 2),
    }


# ------------------------------------------------------------------
# Simple monthly cross-sectional regression factor returns
# ------------------------------------------------------------------
def analyze_portfolios_regression(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """
    For each factor and each month, run a simple cross-sectional OLS of
    country excess returns on factor scores:

        (1MRet(country) - benchmark_return) = alpha + beta * factor_score(country) + error

    The slope (beta) becomes that factor's excess return for that month.
    A positive beta means higher-scored countries earned higher excess returns.

    Parameters
    ----------
    data              : long-format tidy DataFrame (date, country, variable, value)
    features          : list of factor names
    benchmark_returns : equal-weight benchmark return series (indexed by date)

    Returns
    -------
    monthly_betas : dict[str, Series]   — one beta per date per factor
    results_df    : DataFrame           — performance summary table
    """
    results = []
    monthly_betas: Dict[str, pd.Series] = {}

    dates = sorted(data["date"].unique())

    print("Building returns pivot for regression...")
    ret_wide = (
        data[data["variable"] == "1MRet"]
        .pivot_table(index="date", columns="country", values="value", aggfunc="first")
    )
    ret_wide.index = pd.to_datetime(ret_wide.index)

    # Build a benchmark lookup for fast access per date
    bench_lookup = benchmark_returns.to_dict()

    print(f"Running regressions for {len(features)} features...")
    for feat_idx, feature in enumerate(features):
        if (feat_idx + 1) % 20 == 0:
            print(f"  Regressed {feat_idx + 1}/{len(features)} features...")

        feat_wide = (
            data[data["variable"] == feature]
            .pivot_table(index="date", columns="country", values="value", aggfunc="first")
        )
        feat_wide.index = pd.to_datetime(feat_wide.index)

        beta_series = pd.Series(index=pd.to_datetime(dates), dtype=float)

        for date in pd.to_datetime(dates):
            if date not in feat_wide.index or date not in ret_wide.index:
                continue

            x = feat_wide.loc[date]
            y_raw = ret_wide.loc[date]

            # Subtract benchmark to get excess returns
            bm = bench_lookup.get(date, np.nan)
            if np.isnan(bm):
                continue
            y = y_raw - bm

            valid = x.notna() & y.notna()
            x_clean = x[valid].values.astype(float)
            y_clean = y[valid].values.astype(float)

            if len(x_clean) < 3:
                continue

            if np.std(x_clean) < 1e-14:
                continue

            try:
                slope, _intercept, _r, _p, _se = stats.linregress(x_clean, y_clean)
                beta_series[date] = slope
            except ValueError:
                continue

        monthly_betas[feature] = beta_series

        metrics = calculate_factor_return_metrics(beta_series)
        results.append({"Feature": feature, **metrics})

    metric_cols = [
        "Feature",
        "Mean Factor Return (% ann.)",
        "Volatility (% ann.)",
        "t-statistic",
        "Sharpe Ratio",
        "Hit Ratio (%)",
        "Max Drawdown (%)",
        "Skewness",
        "Kurtosis",
        "Calmar Ratio",
    ]
    results_df = (
        pd.DataFrame(results)[metric_cols] if results else pd.DataFrame()
    )
    return monthly_betas, results_df


# ------------------------------------------------------------------
# Visualisation for regression factor returns
# ------------------------------------------------------------------
def create_regression_charts(
    betas_dict: Dict[str, pd.Series],
    output_path: str,
) -> None:
    """
    Plot cumulative factor return (regression slope) over time,
    one panel per factor.
    """
    n_features, n_cols = len(betas_dict), 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    colors = plt.cm.tab10.colors

    for i, (feature, betas) in enumerate(betas_dict.items()):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])

        first_valid = betas.first_valid_index()
        if first_valid is not None:
            cum = betas[first_valid:].cumsum() * 100
            ax.plot(
                cum.index, cum,
                label="Cumul. Factor Return",
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


