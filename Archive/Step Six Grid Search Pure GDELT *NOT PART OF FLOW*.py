"""
Factor Timing Strategy Grid Search — Pure GDELT (not part of main flow)

=============================================================================
INPUT FILES:
- GDELT_Optimizer.xlsx (default): sheet ``Monthly_Net_Returns`` from
  Step Four GDELT Create Monthly Top20 Returns FAST.py.
- Step Factor Categories GDELT.xlsx: columns ``Factor Name`` and ``Max``.
  Only factors with ``Max == 1`` are used (same rule as the classic T2 grid search).

OUTPUT FILES:
- Terminal only: three grids (annualized return, Sharpe, monthly turnover) plus best combos.

WHAT THIS DOES:
Same logic as ``Step Six Grid Search *NOT PART OF FLOW*.py`` and the combined
``Step Six Grid Search T2 GDELT Combined *NOT PART OF FLOW*.py``: grid over
(number of top factors) × (lookback months), score factors by momentum × hit rate × (1+Sharpe),
weight selected names by 12-month trailing return.

Version: 1.0
Last updated: 2026-04-03
=============================================================================
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

# -----------------------------------------------------------------------------
# Paths — point OPTIMIZER_PATH at any GDELT optimizer workbook with the same sheet layout
# -----------------------------------------------------------------------------
OPTIMIZER_PATH = "GDELT_Optimizer.xlsx"

FACTOR_CATEGORIES_PATH = "Step Factor Categories GDELT.xlsx"
OPTIMIZER_SHEET = "Monthly_Net_Returns"


def factor_timing_strategy(
    excel_path: str,
    n_top_factors: int = 3,
    lookback: int = 36,
    allowed_factors: Optional[list] = None,
    sheet_name: str | int = 0,
    verbose: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    returns_df = df.apply(pd.to_numeric, errors="coerce").astype(float) / 100.0

    if allowed_factors is not None:
        available_factors = set(returns_df.columns)
        filtered_factors = [f for f in allowed_factors if f in available_factors]
        if len(filtered_factors) == 0:
            raise ValueError(
                f"None of the allowed factors found in {excel_path}. "
                "Check factor names match the optimizer columns."
            )
        returns_df = returns_df[filtered_factors]
        if verbose:
            print(f"Filtered returns data to {len(filtered_factors)} allowed factors.")

    factor_scores = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    for date in returns_df.index[lookback:]:
        hist_data = returns_df.loc[:date]
        for factor in returns_df.columns:
            momentum = hist_data[factor].tail(lookback).mean()
            hit_rate = (hist_data[factor].tail(lookback) > 0).mean()
            vol = hist_data[factor].tail(lookback).std()
            sharpe = momentum / vol if vol != 0 else 0
            factor_scores.loc[date, factor] = momentum * hit_rate * (1 + sharpe)

    positions = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    for date in factor_scores.index[lookback:]:
        positive_factors = factor_scores.loc[date][factor_scores.loc[date] > 0]
        if len(positive_factors) > 0:
            top_factors = positive_factors.nlargest(min(n_top_factors, len(positive_factors)))
            trailing_returns = {}
            for factor in top_factors.index:
                hist_data = returns_df[factor].loc[:date].tail(12)
                twelve_month_return = (1 + hist_data).prod() - 1
                trailing_returns[factor] = max(twelve_month_return, 0)
            total_trailing = sum(trailing_returns.values())
            if total_trailing > 0:
                for factor, trailing_ret in trailing_returns.items():
                    positions.loc[date, factor] = trailing_ret / total_trailing

    strategy_returns = pd.Series(index=returns_df.index[lookback + 1 :], dtype=float)
    dates = returns_df.index

    for i, date in enumerate(dates[lookback + 1 :], lookback + 1):
        prev_date = dates[dates < date][-1]
        strategy_returns[date] = float(np.sum(positions.loc[prev_date] * returns_df.loc[date]))

    return strategy_returns, positions, factor_scores


def calculate_performance_metrics(
    returns: pd.Series, positions: pd.DataFrame
) -> tuple[float, float, float]:
    monthly_mean = returns.mean()
    monthly_vol = returns.std()
    ann_return = (1 + monthly_mean) ** 12 - 1
    ann_vol = monthly_vol * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    turnover = positions.diff().abs().sum(axis=1).mean()
    return ann_return, sharpe, turnover


def run_grid_search(
    excel_path: str,
    n_factors_list: list[int],
    lookbacks_list: list[int],
    allowed_factors: Optional[list] = None,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns_grid = pd.DataFrame(
        index=[f"{lb}m" for lb in lookbacks_list],
        columns=[f"{n}f" for n in n_factors_list],
    )
    sharpe_grid = pd.DataFrame(
        index=[f"{lb}m" for lb in lookbacks_list],
        columns=[f"{n}f" for n in n_factors_list],
    )
    turnover_grid = pd.DataFrame(
        index=[f"{lb}m" for lb in lookbacks_list],
        columns=[f"{n}f" for n in n_factors_list],
    )

    total_combinations = len(n_factors_list) * len(lookbacks_list)
    completed = 0
    print(f"Running grid search across {total_combinations} combinations...")
    start_time = time.time()

    for lookback in lookbacks_list:
        for n_factors in n_factors_list:
            returns, positions, _ = factor_timing_strategy(
                excel_path,
                n_top_factors=n_factors,
                lookback=lookback,
                allowed_factors=allowed_factors,
                sheet_name=sheet_name,
                verbose=False,
            )
            ann_return, sharpe, turnover = calculate_performance_metrics(returns, positions)
            returns_grid.loc[f"{lookback}m", f"{n_factors}f"] = ann_return
            sharpe_grid.loc[f"{lookback}m", f"{n_factors}f"] = sharpe
            turnover_grid.loc[f"{lookback}m", f"{n_factors}f"] = turnover

            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = avg_time * (total_combinations - completed)
            print(
                f"Completed {completed}/{total_combinations} combinations. "
                f"Estimated time remaining: {remaining:.1f}s",
                end="\r",
            )

    print(f"\nGrid search completed in {time.time() - start_time:.1f} seconds.")
    return returns_grid, sharpe_grid, turnover_grid


def format_grid(grid: pd.DataFrame, is_return: bool = True) -> pd.DataFrame:
    if is_return:
        return grid.map(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")
    return grid.map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")


if __name__ == "__main__":
    n_factors_list = [1, 3, 5, 7, 10, 15]
    lookbacks_list = [24, 36, 60, 72, 90]

    print(f"Loading factor categories from {FACTOR_CATEGORIES_PATH}...")
    factor_categories = pd.read_excel(FACTOR_CATEGORIES_PATH)
    allowed_factors = factor_categories[factor_categories["Max"] == 1]["Factor Name"].tolist()

    print(f"Using optimizer: {OPTIMIZER_PATH} (sheet: {OPTIMIZER_SHEET})")
    print(f"{len(allowed_factors)} factors with Max=1 will be included.")
    print()

    returns_grid, sharpe_grid, turnover_grid = run_grid_search(
        OPTIMIZER_PATH,
        n_factors_list,
        lookbacks_list,
        allowed_factors=allowed_factors,
        sheet_name=OPTIMIZER_SHEET,
    )

    returns_formatted = format_grid(returns_grid, is_return=True)
    sharpe_formatted = format_grid(sharpe_grid, is_return=False)
    turnover_formatted = format_grid(turnover_grid, is_return=True)

    print("\n" + "=" * 80)
    print("PURE GDELT — ANNUALIZED RETURNS GRID")
    print("=" * 80)
    print("Rows: lookback (months), Columns: number of top factors")
    print(tabulate(returns_formatted, headers="keys", tablefmt="grid"))

    print("\n" + "=" * 80)
    print("PURE GDELT — SHARPE RATIOS GRID")
    print("=" * 80)
    print(tabulate(sharpe_formatted, headers="keys", tablefmt="grid"))

    print("\n" + "=" * 80)
    print("PURE GDELT — MONTHLY TURNOVER GRID")
    print("=" * 80)
    print(tabulate(turnover_formatted, headers="keys", tablefmt="grid"))

    max_return_idx = returns_grid.stack().idxmax()
    max_sharpe_idx = sharpe_grid.stack().idxmax()
    min_turnover_idx = turnover_grid.stack().idxmin()

    print("\n" + "=" * 80)
    print("BEST COMBINATIONS (Pure GDELT)")
    print("=" * 80)
    print(
        f"Best Return: {returns_grid.stack().max() * 100:.2f}% with "
        f"{max_return_idx[1]} factors and {max_return_idx[0]} lookback"
    )
    print(
        f"Best Sharpe: {sharpe_grid.stack().max():.2f} with "
        f"{max_sharpe_idx[1]} factors and {max_sharpe_idx[0]} lookback"
    )
    print(
        f"Lowest Turnover: {turnover_grid.stack().min() * 100:.2f}% with "
        f"{min_turnover_idx[1]} factors and {min_turnover_idx[0]} lookback"
    )
