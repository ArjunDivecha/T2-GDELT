#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: Archive/Portfolio Construction.py
=============================================================================

INPUT FILES (run from project root: T2 Factor Timing Fuzzy folder):

- T2_Optimizer.xlsx
  What it is: Monthly factor returns used to build portfolios.
  Layout: First column (or index) = dates; other columns = one column per factor
  (same names as in Step Factor Categories). If typical values look like
  percentages (absolute mean > 1), the script divides by 100. Column
  "Monthly Return_CS" is dropped if present.

- Step Factor Categories.xlsx
  What it is: Per-factor maximum weights for the optimizer.
  Required columns: "Factor Name" (text) and "Max" (number, cap per factor).
  Any factor in T2_Optimizer that is missing here gets Max = 1.0.

OUTPUT FILES:

- Portfolio_Construction_parameter_sweep.xlsx
  What it is: Results of a full grid search over risk aversion, HHI penalty,
  estimation window length, and mean type (arithmetic vs EMA).
  Sheets:
    "Summary" — every combination, sorted by Sharpe then return then drawdown.
    "Top 25 Sharpe" — first 25 rows of that sort (best Sharpe ideas).
    "Top 25 Return" — top 25 by annualized return.
    "Top 25 Drawdown" — best (least severe) max drawdowns first (higher %, closer to zero).
    "Metadata" — grids, scaling rules, and total runtime in plain language.

- T2_processing.log (append)
  Console-style log of progress and top combinations when the script finishes.

VERSION HISTORY:
- 1.0 — Initial archived script (parameter sweep for portfolio construction).
- 1.1 — 2026-03-19 — Expanded file header (inputs/outputs, usage) per project docs.

LAST UPDATED: 2026-03-19

DESCRIPTION (simple terms):
This program does not pull Bloomberg data. It reads your saved factor returns
and asks: "If we optimized long-only weights every month using a mean–variance
style rule (with optional concentration penalty), which settings look best?"
It tries many combinations of:
  • lambda — how much you penalize risk in the objective,
  • HHI penalty — extra penalty so weights do not get too concentrated,
  • window — how many months of history feed the mean and covariance,
  • mean mode — plain average of the window vs an EMA over that window.

Expected returns use cross-sectional z-scores each month inside the window, then
the z-score signal is rescaled so its spread matches the raw expected-return
spread (see EXPECTED_RETURN_INPUT in code). Covariance is annualized from the
same trailing window. No future data is used at any rebalance date.

This file lives under Archive/ — it is an experiment / research tool, not the
main numbered pipeline step.

DEPENDENCIES:
- Python 3.10+
- numpy, pandas, scipy, osqp, xlsxwriter

USAGE:
  cd "/Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy"
  python "Archive/Portfolio Construction.py"

NOTES:
- Country-level missing-data fill rules do not apply here; inputs are factor
  returns and category caps, not country panels.
- If OSQP cannot solve a sub-problem, the code logs a warning and falls back to
  equal weights for that step (see OSQPPortfolioOptimizer.optimize_weights).
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import logging
import time
import warnings

import numpy as np
import osqp
import pandas as pd
from scipy import sparse

warnings.filterwarnings("ignore")


LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]
HHI_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
WINDOW_GRID = [36, 60, 72]
MEAN_MODES = ["arithmetic", "ema"]

RETURN_SCALING = 8.0
EPS = 1e-12
EXPECTED_RETURN_INPUT = "cross_sectional_zscore_rescaled_to_raw_dispersion"

SUMMARY_OUTPUT_FILE = "Portfolio_Construction_parameter_sweep.xlsx"


@dataclass(frozen=True)
class SweepConfig:
    lambda_param: float
    hhi_penalty: float
    window_size: int
    mean_mode: str

    @property
    def combo_key(self) -> str:
        return (
            f"lambda={self.lambda_param:g}|"
            f"hhi={self.hhi_penalty:g}|"
            f"window={self.window_size}|"
            f"mean={self.mean_mode}"
        )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("T2_processing.log", mode="a"),
            logging.StreamHandler(),
        ],
    )


class OSQPPortfolioOptimizer:
    """Reusable long-only quadratic optimizer with warm starts."""

    SOLVED_STATUS_VALUES = {1, 2}

    def __init__(self, max_weights_array: np.ndarray) -> None:
        self.max_weights_array = np.asarray(max_weights_array, dtype=float)
        self.n_assets = len(self.max_weights_array)
        self.prev_weights: np.ndarray | None = None

        self.p_template = sparse.triu(
            sparse.csc_matrix(np.ones((self.n_assets, self.n_assets), dtype=float)),
            format="csc",
        )
        p_rows: list[int] = []
        p_cols: list[int] = []
        for col in range(self.n_assets):
            start_ptr = self.p_template.indptr[col]
            end_ptr = self.p_template.indptr[col + 1]
            p_rows.extend(self.p_template.indices[start_ptr:end_ptr].tolist())
            p_cols.extend([col] * (end_ptr - start_ptr))
        self.p_rows = np.array(p_rows, dtype=int)
        self.p_cols = np.array(p_cols, dtype=int)

        constraint_matrix = sparse.vstack(
            [
                sparse.csc_matrix(np.ones((1, self.n_assets), dtype=float)),
                sparse.eye(self.n_assets, format="csc"),
            ],
            format="csc",
        )
        lower_bounds = np.concatenate(([1.0], np.zeros(self.n_assets, dtype=float)))
        upper_bounds = np.concatenate(([1.0], self.max_weights_array))
        linear_objective = np.zeros(self.n_assets, dtype=float)

        self.solver = osqp.OSQP()
        self.solver.setup(
            P=self.p_template,
            q=linear_objective,
            A=constraint_matrix,
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            warm_start=True,
            polish=True,
            eps_abs=1e-10,
            eps_rel=1e-10,
            max_iter=50000,
            adaptive_rho=True,
        )

    def reset_warm_start(self) -> None:
        self.prev_weights = None

    def optimize_weights(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        lambda_param: float,
        hhi_penalty: float,
    ) -> np.ndarray:
        quad_matrix = 2.0 * (lambda_param * covariance_matrix + hhi_penalty * np.eye(self.n_assets))
        quad_matrix = (quad_matrix + quad_matrix.T) / 2.0
        quad_values = quad_matrix[self.p_rows, self.p_cols]
        linear_objective = -np.asarray(expected_returns, dtype=float)

        self.solver.update(Px=quad_values, q=linear_objective)
        if self.prev_weights is not None:
            self.solver.warm_start(x=self.prev_weights)

        result = self.solver.solve()
        status_value = getattr(result.info, "status_val", None)
        if status_value in self.SOLVED_STATUS_VALUES and result.x is not None:
            optimal_weights = np.asarray(result.x, dtype=float)
            self.prev_weights = optimal_weights.copy()
            return optimal_weights

        logging.warning(
            "OSQP failed with status %s for lambda=%s hhi=%s. Falling back to equal weights.",
            getattr(result.info, "status", "unknown"),
            lambda_param,
            hhi_penalty,
        )
        fallback = np.ones(self.n_assets, dtype=float) / self.n_assets
        self.prev_weights = fallback.copy()
        return fallback


def load_and_prepare_data() -> tuple[pd.DataFrame, np.ndarray]:
    logging.info("Loading input data...")

    returns = pd.read_excel("T2_Optimizer.xlsx", index_col=0)
    returns.index = pd.to_datetime(returns.index)

    factor_categories = pd.read_excel("Step Factor Categories.xlsx")
    max_weights = dict(zip(factor_categories["Factor Name"], factor_categories["Max"]))

    if returns.abs().mean().mean() > 1:
        returns = returns / 100.0

    if "Monthly Return_CS" in returns.columns:
        returns = returns.drop(columns=["Monthly Return_CS"])

    returns = returns.sort_index()
    max_weights_array = np.array([max_weights.get(name, 1.0) for name in returns.columns], dtype=float)

    logging.info("Loaded returns data: %s periods, %s factors", returns.shape[0], returns.shape[1])
    return returns, max_weights_array


def determine_window_values(
    returns_values: np.ndarray,
    date_position: int,
    window_size: int,
) -> np.ndarray:
    if date_position <= window_size:
        return returns_values[:date_position]
    return returns_values[date_position - window_size : date_position]


def aggregate_window_signal(window_values: np.ndarray, mean_mode: str) -> np.ndarray:
    if mean_mode == "arithmetic":
        return window_values.mean(axis=0)

    if mean_mode == "ema":
        alpha = 2.0 / (window_values.shape[0] + 1.0)
        ema = window_values[0].astype(float).copy()
        for row in window_values[1:]:
            ema = alpha * row + (1.0 - alpha) * ema
        return ema

    raise ValueError(f"Unknown mean mode: {mean_mode}")


def cross_sectional_zscore(window_values: np.ndarray) -> np.ndarray:
    row_means = window_values.mean(axis=1, keepdims=True)
    row_stds = window_values.std(axis=1, ddof=0, keepdims=True)
    safe_stds = np.where(row_stds > EPS, row_stds, 1.0)
    zscored = (window_values - row_means) / safe_stds
    zscored[row_stds[:, 0] <= EPS] = 0.0
    return zscored


def compute_expected_returns(window_values: np.ndarray, mean_mode: str) -> np.ndarray:
    raw_expected_returns = RETURN_SCALING * aggregate_window_signal(window_values, mean_mode)
    zscored_window = cross_sectional_zscore(window_values)
    zscore_signal = aggregate_window_signal(zscored_window, mean_mode)

    raw_dispersion = float(raw_expected_returns.std(ddof=0))
    zscore_dispersion = float(zscore_signal.std(ddof=0))
    if raw_dispersion <= EPS or zscore_dispersion <= EPS:
        return np.zeros(window_values.shape[1], dtype=float)

    scale_factor = raw_dispersion / zscore_dispersion
    return zscore_signal * scale_factor


def stabilize_covariance(covariance_matrix: np.ndarray) -> np.ndarray:
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    except np.linalg.LinAlgError:
        avg_var = float(np.diag(covariance_matrix).mean()) if covariance_matrix.size else 1e-6
        covariance_matrix = np.eye(covariance_matrix.shape[0], dtype=float) * max(avg_var, 1e-6)

    return covariance_matrix


def build_covariance(window_values: np.ndarray, n_assets: int) -> np.ndarray:
    if len(window_values) <= 1:
        return np.eye(n_assets, dtype=float) * 1e-6
    covariance_matrix = np.cov(window_values.T, ddof=0) * 12.0
    return stabilize_covariance(covariance_matrix)


def precompute_window_statistics(returns_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    logging.info("Pre-computing expected returns and covariance matrices...")

    returns_values = returns_df.to_numpy(dtype=float)
    n_periods = len(returns_df.index)
    n_assets = returns_values.shape[1]
    optimization_periods = n_periods - 1
    output_periods = optimization_periods + 1

    precomputed: dict[int, dict[str, np.ndarray]] = {}
    for window_size in WINDOW_GRID:
        covariance_stack = np.zeros((output_periods, n_assets, n_assets), dtype=float)
        arithmetic_returns = np.zeros((output_periods, n_assets), dtype=float)
        ema_returns = np.zeros((output_periods, n_assets), dtype=float)
        effective_lookbacks = np.zeros(output_periods, dtype=int)

        for date_position in range(1, n_periods):
            period_index = date_position - 1
            window_values = determine_window_values(returns_values, date_position, window_size)
            covariance_stack[period_index] = build_covariance(window_values, n_assets)
            arithmetic_returns[period_index] = compute_expected_returns(window_values, "arithmetic")
            ema_returns[period_index] = compute_expected_returns(window_values, "ema")
            effective_lookbacks[period_index] = len(window_values)

        extra_window_values = returns_values[max(0, n_periods - window_size) :]
        covariance_stack[-1] = build_covariance(extra_window_values, n_assets)
        arithmetic_returns[-1] = compute_expected_returns(extra_window_values, "arithmetic")
        ema_returns[-1] = compute_expected_returns(extra_window_values, "ema")
        effective_lookbacks[-1] = len(extra_window_values)

        precomputed[window_size] = {
            "covariances": covariance_stack,
            "arithmetic": arithmetic_returns,
            "ema": ema_returns,
            "effective_lookbacks": effective_lookbacks,
        }

    logging.info("Finished pre-computing statistics for %s window settings", len(WINDOW_GRID))
    return precomputed


def normalize_weights(weights: np.ndarray, max_weights_array: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    weight_sum = float(weights.sum())
    if weight_sum > EPS and abs(weight_sum - 1.0) > 1e-6:
        scaled_weights = weights / weight_sum
        if not np.any(scaled_weights > max_weights_array + 1e-8):
            weights = scaled_weights
    return weights


def calculate_strategy_performance(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    returns_values = returns_df.to_numpy(dtype=float)
    aligned_dates = returns_df.index[2:]
    portfolio_returns = np.sum(weights[: len(aligned_dates)] * returns_values[2:], axis=1)
    monthly_returns = pd.Series(portfolio_returns, index=aligned_dates, dtype=float)

    ann_return = (1.0 + monthly_returns.mean()) ** 12 - 1.0 if not monthly_returns.empty else np.nan
    ann_vol = monthly_returns.std(ddof=1) * np.sqrt(12.0) if len(monthly_returns) > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and abs(ann_vol) > EPS else np.nan

    cumulative = (1.0 + monthly_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max

    turnover_values = np.abs(np.diff(weights, axis=0)).sum(axis=1) / 2.0
    turnover_dates = list(returns_df.index[2:]) + [returns_df.index[-1] + pd.DateOffset(months=1)]
    monthly_turnover = pd.Series(turnover_values, index=turnover_dates, dtype=float)
    portfolio_hhi = np.sum(np.square(weights), axis=1)

    statistics = {
        "Annualized Return (%)": float(ann_return * 100.0) if pd.notna(ann_return) else np.nan,
        "Annualized Volatility (%)": float(ann_vol * 100.0) if pd.notna(ann_vol) else np.nan,
        "Sharpe Ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Maximum Drawdown (%)": float(drawdowns.min() * 100.0) if not drawdowns.empty else np.nan,
        "Average Monthly Turnover (%)": float(monthly_turnover.mean() * 100.0) if not monthly_turnover.empty else np.nan,
        "Average Portfolio HHI": float(portfolio_hhi.mean()) if len(portfolio_hhi) else np.nan,
        "Positive Months (%)": float((monthly_returns > 0).mean() * 100.0) if not monthly_returns.empty else np.nan,
        "Skewness": float(monthly_returns.skew()) if not monthly_returns.empty else np.nan,
        "Kurtosis": float(monthly_returns.kurtosis()) if not monthly_returns.empty else np.nan,
    }

    return statistics, monthly_returns, monthly_turnover


def run_single_configuration(
    config: SweepConfig,
    precomputed: dict[int, dict[str, np.ndarray]],
    optimizer: OSQPPortfolioOptimizer,
    returns_df: pd.DataFrame,
    max_weights_array: np.ndarray,
) -> dict[str, float | str | int]:
    combo_start = time.time()
    statistics_bundle = precomputed[config.window_size]
    expected_returns_stack = statistics_bundle[config.mean_mode]
    covariance_stack = statistics_bundle["covariances"]

    weights = np.zeros((expected_returns_stack.shape[0], expected_returns_stack.shape[1]), dtype=float)
    optimizer.reset_warm_start()
    for period_index in range(expected_returns_stack.shape[0]):
        optimal_weights = optimizer.optimize_weights(
            expected_returns=expected_returns_stack[period_index],
            covariance_matrix=covariance_stack[period_index],
            lambda_param=config.lambda_param,
            hhi_penalty=config.hhi_penalty,
        )
        weights[period_index] = normalize_weights(optimal_weights, max_weights_array)

    statistics, _, _ = calculate_strategy_performance(weights, returns_df)
    return {
        "Combo Key": config.combo_key,
        "Lambda": config.lambda_param,
        "HHI Penalty": config.hhi_penalty,
        "Window Size": config.window_size,
        "Mean Mode": config.mean_mode,
        **statistics,
        "Runtime (s)": round(time.time() - combo_start, 4),
    }


def run_parameter_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_start = time.time()
    returns_df, max_weights_array = load_and_prepare_data()
    precomputed = precompute_window_statistics(returns_df)
    optimizer = OSQPPortfolioOptimizer(max_weights_array=max_weights_array)

    configs = [
        SweepConfig(
            lambda_param=lambda_param,
            hhi_penalty=hhi_penalty,
            window_size=window_size,
            mean_mode=mean_mode,
        )
        for lambda_param, hhi_penalty, window_size, mean_mode in product(
            LAMBDA_GRID,
            HHI_GRID,
            WINDOW_GRID,
            MEAN_MODES,
        )
    ]

    logging.info("Running parameter sweep across %s combinations...", len(configs))
    results: list[dict[str, float | str | int]] = []
    for combo_index, config in enumerate(configs, start=1):
        if combo_index == 1 or combo_index % 20 == 0 or combo_index == len(configs):
            elapsed = time.time() - overall_start
            logging.info(
                "Running combo %s/%s after %.1fs: %s",
                combo_index,
                len(configs),
                elapsed,
                config.combo_key,
            )
        results.append(
            run_single_configuration(
                config=config,
                precomputed=precomputed,
                optimizer=optimizer,
                returns_df=returns_df,
                max_weights_array=max_weights_array,
            )
        )

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(
        by=["Sharpe Ratio", "Annualized Return (%)", "Maximum Drawdown (%)"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    metadata_df = pd.DataFrame(
        {
            "Field": [
                "total_combinations",
                "lambda_grid",
                "hhi_grid",
                "window_grid",
                "mean_modes",
                "expected_return_input",
                "expected_return_scaling",
                "signal_rescaling",
                "ema_definition",
                "covariance_definition",
                "runtime_seconds",
            ],
            "Value": [
                len(configs),
                ",".join(str(value) for value in LAMBDA_GRID),
                ",".join(str(value) for value in HHI_GRID),
                ",".join(str(value) for value in WINDOW_GRID),
                ",".join(MEAN_MODES),
                EXPECTED_RETURN_INPUT,
                RETURN_SCALING,
                "Signal std matched to the raw expected-return cross-sectional std at each rebalance",
                "EMA over the active trailing window with span equal to active lookback",
                "Annualized sample covariance over the active trailing window",
                round(time.time() - overall_start, 4),
            ],
        }
    )

    return summary_df, metadata_df


def save_results(summary_df: pd.DataFrame, metadata_df: pd.DataFrame) -> None:
    logging.info("Saving parameter sweep summary to %s", SUMMARY_OUTPUT_FILE)

    with pd.ExcelWriter(SUMMARY_OUTPUT_FILE, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        summary_df.head(25).to_excel(writer, sheet_name="Top 25 Sharpe", index=False)
        summary_df.sort_values(by="Annualized Return (%)", ascending=False).head(25).to_excel(
            writer,
            sheet_name="Top 25 Return",
            index=False,
        )
        summary_df.sort_values(by="Maximum Drawdown (%)", ascending=False).head(25).to_excel(
            writer,
            sheet_name="Top 25 Drawdown",
            index=False,
        )
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        workbook = writer.book
        percent_format = workbook.add_format({"num_format": "0.00"})
        runtime_format = workbook.add_format({"num_format": "0.0000"})
        summary_sheet_names = ["Summary", "Top 25 Sharpe", "Top 25 Return", "Top 25 Drawdown"]
        for sheet_name in summary_sheet_names:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column("A:A", 40)
            worksheet.set_column("B:E", 14)
            worksheet.set_column("F:N", 20, percent_format)
            worksheet.set_column("O:O", 14, runtime_format)

        metadata_sheet = writer.sheets["Metadata"]
        metadata_sheet.set_column("A:A", 28)
        metadata_sheet.set_column("B:B", 80)


def main() -> None:
    setup_logging()

    logging.info("=" * 80)
    logging.info("PORTFOLIO CONSTRUCTION PARAMETER SWEEP")
    logging.info("=" * 80)
    logging.info("Lambda grid: %s", LAMBDA_GRID)
    logging.info("HHI grid: %s", HHI_GRID)
    logging.info("Window grid: %s", WINDOW_GRID)
    logging.info("Mean modes: %s", MEAN_MODES)
    logging.info("=" * 80)

    summary_df, metadata_df = run_parameter_sweep()
    save_results(summary_df, metadata_df)

    logging.info("=" * 80)
    logging.info("TOP 10 COMBINATIONS BY SHARPE")
    logging.info("=" * 80)
    for _, row in summary_df.head(10).iterrows():
        logging.info(
            "%s | Return=%6.2f | Vol=%6.2f | Sharpe=%5.2f | MDD=%7.2f",
            row["Combo Key"],
            row["Annualized Return (%)"],
            row["Annualized Volatility (%)"],
            row["Sharpe Ratio"],
            row["Maximum Drawdown (%)"],
        )
    logging.info("=" * 80)
    logging.info("PORTFOLIO CONSTRUCTION SWEEP COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
