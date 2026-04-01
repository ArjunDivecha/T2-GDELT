"""
Factor Timing Strategy Implementation — smooth_v69__long_only_conc_p10

INPUT FILES:
1. T60.xlsx
   - Excel file with dates in first column and T60 factor forecasts in subsequent columns
   - Format: Excel workbook with single sheet
   - Index: Dates in datetime format

2. T2_Optimizer.xlsx
   - Excel file with dates in first column and realized factor returns (as percentages)
   - Format: Excel workbook with single sheet
   - Index: Dates in datetime format

OUTPUT FILES:
1. T2_Top3_Factor_Weights.xlsx
   - Excel file containing portfolio weights over time
   - Format: Excel workbook with single sheet
   - Index: Dates (monthly)
   - Columns: Factor weights from smooth_v69 long-only concentrated strategy

2. T2_Top3_Monthly_Returns.xlsx
   - Excel file containing monthly returns of the strategy
   - Format: Excel workbook with single sheet
   - Index: Dates (monthly)
   - Columns: Monthly returns

STRATEGY OVERVIEW (smooth_v69__long_only_conc_p10):
This module implements an adaptive factor rotation strategy with three stages:

  Stage 1 — Signal Construction:
    - Triple-smooth T60 forecasts using 70/15/15 weights (current + 2 lags)
    - Blend 88% smoothed forecast with 12% 3-month momentum (trailing realized returns)
    - All features cross-sectionally z-scored within each month

  Stage 2 — Volatility Adjustment & Re-Standardization:
    - Compute inverse 6-month realized vol for each factor
    - Multiplicative tilt: signal × (1 + 0.2 × clipped_inv_vol_zscore)
    - Re-standardize signal cross-sectionally

  Stage 3 — Portfolio Construction (long_only_conc_p10):
    - Select top-12 factors by final signal
    - Shift signals so minimum = 0, raise to power 10 for concentration
    - Normalize weights to sum to 1
    - Long-only, monthly rebalancing

Performance (Out-of-Sample):
  Sharpe: 1.0476 | Return: 5.93% ann. | Vol: 5.66% ann. | Max DD: -8.71% | Turnover: 0.30

Key Parameters:
  Smoothing weights:    70 / 15 / 15
  Momentum weight:      12%
  Signal weight:        88%
  Vol tilt strength:    0.2
  Vol tilt clipping:    ±2σ
  Concentration power:  10
  Top-K factors:        12
  Long-only:            Yes
  Rebalance:            Monthly

Version: 2.0
Last Updated: 2026
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helper: cross-sectional z-score (across factors for a single date)
# ---------------------------------------------------------------------------
def _cs_zscore(row: pd.Series) -> pd.Series:
    """Cross-sectional z-score: (x - mean) / std across factors for one date.
    Returns 0 if std < 1e-12."""
    mu = row.mean()
    sigma = row.std(ddof=0)
    if sigma < 1e-12:
        return row * 0.0
    return (row - mu) / sigma


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------
def factor_timing_strategy(
    t60_path: str,
    optimizer_path: str,
    top_k: int = 12,
    power: float = 10.0,
    smooth_weights: tuple = (0.70, 0.15, 0.15),
    momentum_blend: float = 0.12,
    vol_tilt_strength: float = 0.2,
    vol_tilt_clip: float = 2.0,
    momentum_window: int = 3,
    vol_window: int = 6,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Implements the smooth_v69__long_only_conc_p10 factor timing strategy.

    Parameters
    ----------
    t60_path : str
        Path to T60.xlsx (factor forecasts).
    optimizer_path : str
        Path to T2_Optimizer.xlsx (realized factor returns as percentages).
    top_k : int, default=12
        Number of top factors to hold long each month.
    power : float, default=10.0
        Concentration exponent for portfolio construction.
    smooth_weights : tuple, default=(0.70, 0.15, 0.15)
        Weights for current / lag-1 / lag-2 T60 z-scores.
    momentum_blend : float, default=0.12
        Weight on 3-month momentum; (1 - momentum_blend) on smoothed forecast.
    vol_tilt_strength : float, default=0.2
        Multiplicative inverse-vol adjustment strength.
    vol_tilt_clip : float, default=2.0
        Clipping bound (±) for normalized inverse-vol.
    momentum_window : int, default=3
        Trailing months for momentum calculation.
    vol_window : int, default=6
        Trailing months for volatility calculation.

    Returns
    -------
    strategy_returns : pd.Series
        Monthly returns of the strategy.
    positions : pd.DataFrame
        Factor weights/positions over time.
    final_signals : pd.DataFrame
        Final signal values for each factor over time.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t60_raw = pd.read_excel(t60_path)
    t60_raw.index = pd.to_datetime(t60_raw.iloc[:, 0])
    t60_df = t60_raw.iloc[:, 1:].astype(float)

    opt_raw = pd.read_excel(optimizer_path)
    opt_raw.index = pd.to_datetime(opt_raw.iloc[:, 0])
    returns_pct = opt_raw.iloc[:, 1:].astype(float)      # percentage returns
    returns_df = returns_pct / 100.0                       # decimal returns

    # Align factors (intersection)
    common_factors = t60_df.columns.intersection(returns_df.columns)
    t60_df = t60_df[common_factors]
    returns_df = returns_df[common_factors]
    returns_pct = returns_pct[common_factors]

    # ------------------------------------------------------------------
    # 2. Build features
    # ------------------------------------------------------------------

    # 2a. Cross-sectional z-score of T60 (current and lagged)
    t60_cs_z = t60_df.apply(_cs_zscore, axis=1)
    t60_lag1_cs_z = t60_df.shift(1).apply(_cs_zscore, axis=1)
    t60_lag2_cs_z = t60_df.shift(2).apply(_cs_zscore, axis=1)

    # 2b. Triple-smooth (v69 weights: 70/15/15)
    w0, w1, w2 = smooth_weights
    smooth = w0 * t60_cs_z + w1 * t60_lag1_cs_z + w2 * t60_lag2_cs_z

    # 2c. 3-month trailing mean of realized returns, cross-sectionally z-scored
    #     Use shift(1) to avoid lookahead: looks at t-1 through t-3
    returns_shifted = returns_df.shift(1)
    target_mean_3 = returns_shifted.rolling(window=momentum_window, min_periods=momentum_window).mean()
    target_mean_3_cs_z = target_mean_3.apply(_cs_zscore, axis=1)

    # 2d. Inverse vol (6-month trailing vol of realized returns, shifted by 1)
    target_vol_6 = returns_shifted.rolling(window=vol_window, min_periods=vol_window).std()
    inv_vol_6 = 1.0 / target_vol_6.clip(lower=0.01)

    # ------------------------------------------------------------------
    # 3. Signal construction
    # ------------------------------------------------------------------
    signal_weight = 1.0 - momentum_blend  # 0.88
    base_signal = signal_weight * smooth + momentum_blend * target_mean_3_cs_z

    # 3a. Cross-sectional normalize inverse vol & clip
    inv_vol_norm = inv_vol_6.apply(_cs_zscore, axis=1)
    inv_vol_norm = inv_vol_norm.clip(-vol_tilt_clip, vol_tilt_clip)

    # 3b. Multiplicative vol tilt
    signal_adj = base_signal * (1.0 + vol_tilt_strength * inv_vol_norm)

    # 3c. Re-standardize across factors
    final_signal = signal_adj.apply(_cs_zscore, axis=1)

    # ------------------------------------------------------------------
    # 4. Portfolio construction — long_only_conc_p10
    # ------------------------------------------------------------------
    # Determine valid dates: need enough history for all rolling features
    min_lookback = max(2, momentum_window, vol_window) + 1  # lags + rolling
    all_dates = final_signal.index
    valid_mask = final_signal.notna().all(axis=1)
    valid_dates = all_dates[valid_mask]

    positions = pd.DataFrame(0.0, index=returns_df.index, columns=common_factors)

    for date in valid_dates:
        sig = final_signal.loc[date].dropna()
        if len(sig) < top_k:
            continue

        # Rank descending, pick top-K
        top_factors = sig.nlargest(top_k)

        # Shift so minimum = 0
        v = top_factors - top_factors.min()

        # Raise to power for concentration
        w_raw = v ** power

        # Normalize
        total = w_raw.sum()
        if total > 0:
            weights = w_raw / total
            positions.loc[date, weights.index] = weights.values

    # ------------------------------------------------------------------
    # 5. Calculate strategy returns (use previous month's weights)
    # ------------------------------------------------------------------
    dates_idx = returns_df.index
    # First valid position date
    first_pos_date_idx = positions.index.get_indexer(valid_dates)[0]
    ret_start = first_pos_date_idx + 1  # return starts one month after first weights

    strategy_returns = pd.Series(dtype=float)

    for i in range(ret_start, len(dates_idx)):
        date = dates_idx[i]
        prev_date = dates_idx[i - 1]

        w = positions.loc[prev_date]
        if w.sum() == 0:
            continue

        ret = returns_df.loc[date] if date in returns_df.index else None
        if ret is None:
            continue

        strategy_returns[date] = (w * ret).sum()

    return strategy_returns, positions, final_signal


# ---------------------------------------------------------------------------
# Evaluation helpers (unchanged interface)
# ---------------------------------------------------------------------------
def calculate_turnover(weights_df: pd.DataFrame) -> tuple:
    """
    Calculate average monthly turnover and turnover series.

    Args:
        weights_df (pd.DataFrame): Portfolio weights over time.

    Returns:
        tuple: Average monthly turnover and monthly turnover series.
    """
    turnover_series = weights_df.diff().abs().sum(axis=1)
    avg_turnover = turnover_series.mean()
    return avg_turnover, turnover_series


def evaluate_strategy(returns: pd.Series, positions: pd.DataFrame) -> dict:
    """
    Calculates key performance metrics for a trading strategy.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns of the strategy
    positions : pd.DataFrame
        Factor weights/positions over time

    Returns
    -------
    dict with: Annualized Return, Annualized Vol, Sharpe Ratio, Max Drawdown, Turnover
    """
    positions_aligned = positions.loc[
        positions.index.isin(returns.index)
        | positions.index.isin(returns.index - pd.DateOffset(months=1))
    ]

    monthly_mean = returns.mean()
    monthly_vol = returns.std()

    ann_return = (1 + monthly_mean) ** 12 - 1
    ann_vol = monthly_vol * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0

    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    avg_turnover, _ = calculate_turnover(positions_aligned)

    return {
        "Annualized Return": f"{ann_return:.2%}",
        "Annualized Vol": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Turnover": f"{avg_turnover:.2%}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Main execution block that:
    1. Runs the smooth_v69__long_only_conc_p10 strategy
    2. Prints performance metrics
    3. Saves factor weights and monthly returns to Excel

    Output files:
    - T2_Top3_Factor_Weights.xlsx  (portfolio weights over time)
    - T2_Top3_Monthly_Returns.xlsx (monthly strategy returns)
    """

    T60_PATH = "T60.xlsx"
    OPT_PATH = "T2_Optimizer.xlsx"

    print("Running smooth_v69__long_only_conc_p10 strategy...")
    print(f"  T60 forecasts:     {T60_PATH}")
    print(f"  Realized returns:  {OPT_PATH}")
    print()

    strategy_returns, positions, final_signals = factor_timing_strategy(
        t60_path=T60_PATH,
        optimizer_path=OPT_PATH,
    )

    # Evaluate
    perf = evaluate_strategy(strategy_returns, positions)
    print("Strategy Performance:")
    for k, v in perf.items():
        print(f"  {k:20s}: {v}")
    print()

    # --- Save weights ---
    orig_df = pd.read_excel(OPT_PATH)
    dates_col_name = orig_df.columns[0]

    output_df = pd.DataFrame(index=positions.index)
    output_df.insert(0, dates_col_name, output_df.index)
    output_df = pd.concat([output_df, positions], axis=1)

    output_path = "T2_Top3_Factor_Weights.xlsx"
    output_df.to_excel(output_path, index=False)
    print(f"Factor weights saved to: {output_path}")

    # --- Save monthly returns ---
    returns_out = pd.DataFrame(strategy_returns)
    returns_out.columns = ["Monthly_Return"]
    returns_out["Monthly_Return"] = returns_out["Monthly_Return"] * 100  # percentage
    returns_path = "T2_Top3_Monthly_Returns.xlsx"
    returns_out.to_excel(returns_path)
    print(f"Monthly returns saved to: {returns_path}")
