"""
=============================================================================
SCRIPT NAME: Step Five GDELT Sweep.py
=============================================================================

INPUT FILES:
- GDELT_Optimizer.xlsx: Monthly net returns for GDELT factor portfolios
  (sheet: Monthly_Net_Returns), used as the optimization universe
- Step Factor Categories GDELT.xlsx: Factor definitions and max-weight
  constraints (sheet: Factor Categories; columns: Factor Name, Category, Max)

OUTPUT FILES:
- None (console output only: three summary tables)

VERSION: 2.0
LAST UPDATED: 2026-04-10
AUTHOR: Arjun Divecha

DESCRIPTION:
Runs a parameter sweep over LAMBDA, HHI_PENALTY, and WINDOW_SIZE to find
the best combination for the GDELT rolling-window portfolio optimization.
For each of the 64 combinations (4x4x4), it runs the full rolling-window
backtest and collects three performance metrics:

  1. Annualized Return (geometric)
  2. Sharpe Ratio (annualized)
  3. Mean Monthly Turnover

Results are printed to the console as three table groups, one per metric.
Each group has one table per WINDOW_SIZE, with LAMBDA as rows and
HHI_PENALTY as columns.

SWEEP PARAMETERS:
  LAMBDA:       [0, 10, 100, 1000]
  HHI_PENALTY:  [0, 0.005, 0.1, 1]
  WINDOW_SIZE:  [12, 36, 60, 90]

DEPENDENCIES:
- pandas, numpy, cvxpy, openpyxl

USAGE:
python "Step Five GDELT Sweep.py"
=============================================================================
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import time
import os

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZER_FILE = os.path.join(BASE_DIR, "GDELT_Optimizer.xlsx")
FACTOR_CATS_FILE = os.path.join(BASE_DIR, "Step Factor Categories GDELT.xlsx")

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
LAMBDAS = [0, 10, 100, 1000]
HHI_PENALTIES = [0, 0.005, 0.1, 1]
WINDOW_SIZES = [12, 36, 60, 90]

# ---------------------------------------------------------------------------
# Fixed settings (not swept)
# ---------------------------------------------------------------------------
USE_COVARIANCE = True    # Enable lambda impact through covariance penalty
RISK_FREE_RATE = 0.0

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 76)
print("STEP FIVE GDELT SWEEP - Parameter Grid Search")
print("=" * 76)

print("\nLoading GDELT_Optimizer.xlsx ...")
df_raw = pd.read_excel(OPTIMIZER_FILE, sheet_name="Monthly_Net_Returns")
df_raw.dropna(how="all", inplace=True)

# First column is the date
date_col = df_raw.columns[0]
df_raw[date_col] = pd.to_datetime(df_raw[date_col])
df_raw.sort_values(date_col, inplace=True)
df_raw.reset_index(drop=True, inplace=True)

dates = df_raw[date_col].values
# Step Four writes monthly net returns in percentage points (decimal * 100).
# Convert to decimal returns so backtest math is on the correct scale.
returns_data = df_raw.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce").values / 100.0
factor_names = df_raw.drop(columns=[date_col]).columns.tolist()

print(f"  Loaded {returns_data.shape[0]} months x {returns_data.shape[1]} factors")
print(f"  Date range: {pd.Timestamp(dates[0]).strftime('%Y-%m')} to "
      f"{pd.Timestamp(dates[-1]).strftime('%Y-%m')}")

# ---------------------------------------------------------------------------
# Load factor categories and max weights
# ---------------------------------------------------------------------------
print("\nLoading Step Factor Categories GDELT.xlsx ...")
factor_cats_df = pd.read_excel(FACTOR_CATS_FILE, sheet_name="Factor Categories")

# Build max-weight dictionary: factor_name -> max_weight
max_weight_dict = dict(zip(factor_cats_df["Factor Name"], factor_cats_df["Max"]))

# Default max weight for factors not in the workbook
DEFAULT_MAX_WEIGHT = 1.0

# Build category-group dictionary from Factor Categories:
# category_name -> list of factor indices
category_groups = {}
for i, fname in enumerate(factor_names):
    if fname in factor_cats_df["Factor Name"].values:
        row_match = factor_cats_df[factor_cats_df["Factor Name"] == fname]
        if not row_match.empty and "Category" in factor_cats_df.columns:
            cat_name = str(row_match.iloc[0]["Category"]).strip()
        else:
            cat_name = "Uncategorized"
    else:
        cat_name = "Uncategorized"
    if cat_name not in category_groups:
        category_groups[cat_name] = []
    category_groups[cat_name].append(i)

# Category cap dictionary (optional; only used if present)
category_caps = {}
if "Category_Cap" in factor_cats_df.columns:
    for _, row in factor_cats_df.iterrows():
        cat_name = str(row["Category"]).strip()
        cap_val = row.get("Category_Cap", 1.0)
        if pd.notna(cap_val):
            category_caps[cat_name] = float(cap_val)

# Build max-weight array aligned to factor_names order
max_weights = np.array([max_weight_dict.get(f, DEFAULT_MAX_WEIGHT) for f in factor_names])

print(f"  {len(max_weight_dict)} factor max-weights loaded")
print(f"  {len(category_groups)} category groups identified")
print(f"  Category caps: {category_caps}")

# ---------------------------------------------------------------------------
# Core optimization + backtest function
# ---------------------------------------------------------------------------
def run_rolling_backtest(lambda_val, hhi_penalty, window_size):
    """
    Run the full rolling-window optimization backtest for one parameter combo.

    Args:
        lambda_val:   Risk-aversion parameter (0 = no covariance penalty)
        hhi_penalty:  Herfindahl concentration penalty coefficient
        window_size:  Rolling window length in months

    Returns:
        dict with keys: 'ann_return', 'sharpe', 'turnover'
        Returns None if backtest fails or produces invalid results.
    """
    n_months, n_factors = returns_data.shape
    if window_size >= n_months:
        return None

    # Storage for portfolio weights and returns
    all_weights = []
    all_port_returns = []

    for t in range(window_size, n_months):
        # Trailing window of returns
        window_returns = returns_data[t - window_size:t, :]

        # Mean return in the window (match Step Five FAST scaling)
        mu = 8.0 * np.nanmean(window_returns, axis=0)

        # Skip if too many NaN
        if np.sum(np.isnan(mu)) > n_factors * 0.5:
            all_weights.append(None)
            all_port_returns.append(np.nan)
            continue

        # Fill remaining NaN with 0
        mu = np.nan_to_num(mu, nan=0.0)

        # Covariance matrix (annualized, ddof=0 to match Step Five)
        window_cov = np.cov(window_returns.T, ddof=0) * 12.0
        if np.ndim(window_cov) == 0:
            window_cov = np.array([[float(window_cov)]], dtype=float)
        window_cov = np.asarray(window_cov, dtype=float)
        window_cov = (window_cov + window_cov.T) / 2.0
        try:
            eigvals, eigvecs = np.linalg.eigh(window_cov)
            eigvals = np.maximum(eigvals, 1e-6)
            window_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            avg_var = float(np.nanmean(np.diag(window_cov))) if window_cov.size else 1e-6
            if not np.isfinite(avg_var) or avg_var <= 0:
                avg_var = 1e-6
            window_cov = np.eye(n_factors) * avg_var

        # --- CVXPY optimization ---
        w = cp.Variable(n_factors)

        # Objective: maximize expected return - risk - concentration
        objective_expr = mu @ w

        if USE_COVARIANCE and lambda_val > 0:
            objective_expr -= lambda_val * cp.quad_form(w, cp.psd_wrap(window_cov))

        # Add HHI concentration penalty (minimize concentration)
        if hhi_penalty > 0:
            objective_expr -= hhi_penalty * cp.sum_squares(w)

        # Constraints
        constraints = [
            cp.sum(w) == 1,       # Fully invested
            w >= 0,               # Long only
        ]

        # Per-factor max weight constraints
        for i in range(n_factors):
            constraints.append(w[i] <= max_weights[i])

        # Category cap constraints
        for cat_name, factor_indices in category_groups.items():
            if cat_name in category_caps and len(factor_indices) > 0:
                cap = category_caps[cat_name]
                constraints.append(cp.sum(w[factor_indices]) <= cap)

        # Build and solve - use OSQP like Step Five FAST; fall back to SCS
        prob = cp.Problem(cp.Maximize(objective_expr), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            except Exception:
                all_weights.append(None)
                all_port_returns.append(np.nan)
                continue

        if w.value is None or np.any(np.isnan(w.value)):
            all_weights.append(None)
            all_port_returns.append(np.nan)
            continue

        opt_weights = np.maximum(w.value, 0)
        wsum = opt_weights.sum()
        if wsum <= 0:
            all_weights.append(None)
            all_port_returns.append(np.nan)
            continue
        opt_weights /= wsum  # Re-normalize to sum=1

        all_weights.append(opt_weights)

        # Portfolio return for this month = weight * actual return
        month_return = np.nansum(opt_weights * returns_data[t, :])
        all_port_returns.append(month_return)

    # --- Compute metrics ---
    port_returns = np.array(all_port_returns)
    valid_mask = ~np.isnan(port_returns)

    if valid_mask.sum() < 12:
        return None

    valid_returns = port_returns[valid_mask]

    # Annualized return + volatility + Sharpe (match Step Five FAST)
    ann_return = (1 + np.mean(valid_returns)) ** 12 - 1
    ann_vol = np.std(valid_returns, ddof=1) * np.sqrt(12)

    # Sharpe ratio (annualized)
    if ann_vol < 1e-10:
        sharpe = 0.0
    else:
        sharpe = ann_return / ann_vol

    # Turnover: mean absolute month-to-month weight change
    valid_weights = [w for w in all_weights if w is not None]
    if len(valid_weights) < 2:
        turnover = 0.0
    else:
        turnover_vals = []
        for i in range(1, len(valid_weights)):
            diff = np.abs(valid_weights[i] - valid_weights[i - 1])
            turnover_vals.append(np.sum(diff) / 2.0)
        turnover = np.mean(turnover_vals)

    return {
        'ann_return': ann_return,
        'sharpe': sharpe,
        'turnover': turnover
    }


# ---------------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------------
total_combos = len(LAMBDAS) * len(HHI_PENALTIES) * len(WINDOW_SIZES)
print(f"\nRunning {total_combos} parameter combinations ...")
print(f"  LAMBDA values:       {LAMBDAS}")
print(f"  HHI_PENALTY values:  {HHI_PENALTIES}")
print(f"  WINDOW_SIZE values:  {WINDOW_SIZES}")
print()

# Results storage: results[(lambda, hhi, window)] = {ann_return, sharpe, turnover}
results = {}
combo_count = 0
start_time = time.time()

for lam in LAMBDAS:
    for hhi in HHI_PENALTIES:
        for win in WINDOW_SIZES:
            combo_count += 1
            elapsed = time.time() - start_time
            print(f"  [{combo_count:3d}/{total_combos}] "
                  f"LAMBDA={lam:>6}, HHI={hhi:>7}, WINDOW={win:>3}  "
                  f"({elapsed:.1f}s elapsed)", end="", flush=True)

            result = run_rolling_backtest(lam, hhi, win)

            if result is not None:
                results[(lam, hhi, win)] = result
                print(f"  => AnnRet={result['ann_return']:+.4f}  "
                      f"Sharpe={result['sharpe']:+.4f}  "
                      f"Turn={result['turnover']:.4f}")
            else:
                print(f"  => SKIPPED (insufficient data)")

total_elapsed = time.time() - start_time
print(f"\nSweep complete in {total_elapsed:.1f}s  ({len(results)} valid combos)\n")

# ---------------------------------------------------------------------------
# Print three table groups
# ---------------------------------------------------------------------------
def print_metric_tables(metric_name, metric_key, fmt="+.4f"):
    """Print one table per WINDOW_SIZE for the given metric."""
    print("=" * 76)
    print(f"  {metric_name}")
    print("=" * 76)

    hhi_labels = [f"{h:.3f}" if h > 0 else "0" for h in HHI_PENALTIES]

    for win in WINDOW_SIZES:
        print(f"\n  WINDOW_SIZE = {win}")
        print("-" * 76)

        # Header
        header = f"{'LAMBDA':>8} |"
        for lbl in hhi_labels:
            header += f" {lbl:>10}"
        print(header)
        print("-" * 76)

        for lam in LAMBDAS:
            row = f"{lam:>8} |"
            for hhi in HHI_PENALTIES:
                key = (lam, hhi, win)
                if key in results:
                    val = results[key][metric_key]
                    row += f" {format(val, fmt):>10}"
                else:
                    row += f" {'N/A':>10}"
            print(row)

        print()

    print()


# Print the three table groups
print("\n")
print_metric_tables("ANNUALIZED RETURN", "ann_return", fmt="+.4f")
print_metric_tables("SHARPE RATIO (annualized)", "sharpe", fmt="+.4f")
print_metric_tables("MEAN MONTHLY TURNOVER", "turnover", fmt=".4f")

print("=" * 76)
print("  SWEEP COMPLETE")
print("=" * 76)
