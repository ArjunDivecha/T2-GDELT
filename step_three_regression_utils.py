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

VERSION: 3.0 — added multivariate LASSO and combined overlay charts
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
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


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


# ------------------------------------------------------------------
# Multivariate LASSO: all factors in one regression each month
# ------------------------------------------------------------------
def analyze_portfolios_lasso(
    data: pd.DataFrame,
    features: list,
    benchmark_returns: pd.Series,
) -> Tuple[Dict[str, pd.Series], pd.DataFrame, pd.Series]:
    """
    Each month, run a multivariate ElasticNet (ElasticNetCV) regression:

        excess_return(country) = alpha + sum_j( beta_j * factor_j(country) ) + error

    where excess_return = 1MRet - benchmark.  Factor scores are standardised
    cross-sectionally each month before fitting so that coefficients are
    comparable across factors.  ElasticNetCV picks the regularisation strength
    (alpha) via 5-fold CV each month.  l1_ratio=0.5 blends L1 (LASSO) and
    L2 (Ridge) penalties: L1 still zeroes out truly irrelevant factors while
    L2 keeps correlated factors together in the model.

    Parameters
    ----------
    data              : long-format tidy DataFrame
    features          : list of factor names
    benchmark_returns : equal-weight benchmark return series

    Returns
    -------
    monthly_betas  : dict[str, Series]  — one coeff per date per factor
    results_df     : DataFrame          — performance summary table
    selected_alpha : Series             — LassoCV-chosen alpha per date
    """
    dates = sorted(data["date"].unique())
    dt_dates = pd.to_datetime(dates)

    # Build wide pivots once
    print("Building pivots for LASSO...")
    ret_wide = (
        data[data["variable"] == "1MRet"]
        .pivot_table(index="date", columns="country", values="value", aggfunc="first")
    )
    ret_wide.index = pd.to_datetime(ret_wide.index)

    feat_pivots = {}
    for feat in features:
        fp = (
            data[data["variable"] == feat]
            .pivot_table(index="date", columns="country", values="value", aggfunc="first")
        )
        fp.index = pd.to_datetime(fp.index)
        feat_pivots[feat] = fp

    bench_lookup = benchmark_returns.to_dict()

    # Storage: one Series per factor, plus alpha tracker
    beta_store = {f: pd.Series(index=dt_dates, dtype=float) for f in features}
    alpha_series = pd.Series(index=dt_dates, dtype=float)
    scaler = StandardScaler()

    print(f"Running monthly LASSO across {len(features)} factors...")
    for d_idx, date in enumerate(dt_dates):
        if (d_idx + 1) % 20 == 0:
            print(f"  Processed {d_idx + 1}/{len(dt_dates)} months...")

        if date not in ret_wide.index:
            continue
        bm = bench_lookup.get(date, np.nan)
        if np.isnan(bm):
            continue

        y_raw = ret_wide.loc[date]
        y = y_raw - bm

        # Build X matrix: one column per factor, rows = countries
        x_cols = {}
        for feat in features:
            fp = feat_pivots[feat]
            if date in fp.index:
                x_cols[feat] = fp.loc[date]

        if not x_cols:
            continue

        x_df = pd.DataFrame(x_cols)

        # Keep only countries with valid return AND at least one factor
        valid_countries = y.dropna().index.intersection(x_df.dropna(how="all").index)
        if len(valid_countries) < 10:
            continue

        y_clean = y[valid_countries].values.astype(float)
        x_clean = x_df.loc[valid_countries].copy()

        # Fill remaining NaN factors with 0, replace inf, clip extremes
        x_clean = x_clean.fillna(0.0).values.astype(float)
        x_clean = np.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(x_clean, -1e10, 1e10, out=x_clean)

        # Drop any columns (factors) with zero variance this month
        col_std = x_clean.std(axis=0)
        valid_cols = col_std > 1e-14
        if valid_cols.sum() < 2:
            continue
        x_use = x_clean[:, valid_cols]
        feat_indices = np.where(valid_cols)[0]

        # Standardise factors cross-sectionally
        x_scaled = scaler.fit_transform(x_use)

        # Skip if degenerate
        if np.std(y_clean) < 1e-14:
            continue

        try:
            alphas = np.logspace(-6, -2, 50)
            model = ElasticNetCV(
                l1_ratio=0.3, cv=5, alphas=alphas,
                max_iter=5000, n_jobs=-1,
            )
            model.fit(x_scaled, y_clean)
            alpha_series[date] = model.alpha_

            for coef_idx, orig_col_idx in enumerate(feat_indices):
                beta_store[features[orig_col_idx]][date] = model.coef_[coef_idx]
        except Exception:
            continue

    # Build results table
    results = []
    for feat in features:
        metrics = calculate_factor_return_metrics(beta_store[feat])
        # Add sparsity: % of months where coefficient was exactly zero
        clean = beta_store[feat].dropna()
        pct_zero = (clean == 0).mean() * 100 if len(clean) > 0 else 100.0
        results.append({
            "Feature": feat,
            **metrics,
            "% Months Zero": round(pct_zero, 1),
        })

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
        "% Months Zero",
    ]
    results_df = (
        pd.DataFrame(results)[metric_cols] if results else pd.DataFrame()
    )
    return beta_store, results_df, alpha_series


# ------------------------------------------------------------------
# Combined overlay chart: any two methods on dual y-axes
# ------------------------------------------------------------------
def create_combined_charts(
    series_left: Dict[str, pd.Series],
    series_right: Dict[str, pd.Series],
    label_left: str,
    label_right: str,
    output_path: str,
    color_left: str = "#1f77b4",
    color_right: str = "#d62728",
) -> None:
    """
    Plot two cumulative return series per factor on dual y-axes.
    Used to overlay sort-based vs regression or sort-based vs LASSO.
    """
    common = [f for f in series_left if f in series_right
              and series_left[f].notna().sum() > 3
              and series_right[f].notna().sum() > 3]

    n_cols = 3
    n_rows = (len(common) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(18, n_rows * 4.5))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.4)

    for i, feat in enumerate(common):
        row, col = divmod(i, n_cols)
        ax1 = fig.add_subplot(gs[row, col])

        left = series_left[feat]
        fv1 = left.first_valid_index()
        if fv1 is not None:
            cum1 = left[fv1:].cumsum() * 100
            ax1.plot(cum1.index, cum1, color=color_left, linewidth=1.4,
                     label=label_left)
        ax1.set_ylabel(label_left, fontsize=8, color=color_left)
        ax1.tick_params(axis="y", labelcolor=color_left, labelsize=7)

        ax2 = ax1.twinx()
        right = series_right[feat]
        fv2 = right.first_valid_index()
        if fv2 is not None:
            cum2 = right[fv2:].cumsum() * 100
            ax2.plot(cum2.index, cum2, color=color_right, linewidth=1.4,
                     label=label_right)
        ax2.set_ylabel(label_right, fontsize=8, color=color_right)
        ax2.tick_params(axis="y", labelcolor=color_right, labelsize=7)

        # Correlation
        combined = pd.DataFrame({"a": left, "b": right}).dropna()
        if len(combined) >= 3:
            corr = combined["a"].corr(combined["b"])
            ax1.text(0.02, 0.95, f"\u03c1 = {corr:.2f}",
                     transform=ax1.transAxes, fontsize=8,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="wheat", alpha=0.7))

        ax1.set_title(feat, fontsize=11, weight="bold")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax1.grid(True, linestyle="--", alpha=0.4)

        lines = ax1.get_lines() + ax2.get_lines()
        if lines:
            ax1.legend(lines, [l.get_label() for l in lines],
                       fontsize=7, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
