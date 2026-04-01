#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: nudge.py
=============================================================================

PRIMARY INPUT FILES (default: same folder as this script; override with --input-dir):
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/T60.xlsx
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/T2_Optimizer.xlsx
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/Extrernal Data.xlsx
  (Filenames are fixed in code: T60_FILENAME, T2_FILENAME, EXTERNAL_FILENAME.)

  T60.xlsx
    - What it is: Wide table of T60 factor timing forecasts by month.
    - Required: Column "Date" (monthly timestamps). One column per factor; names
      must overlap with T2_Optimizer.xlsx. Values are numeric forecast levels.
    - Format: Excel .xlsx (read with pandas/openpyxl).

  T2_Optimizer.xlsx
    - What it is: Target or "truth" series used to score forecasts (same shape
      as T60 for overlapping factor columns).
    - Required: "Date" plus the same factor column names as in T60 for columns
      you want in the panel. Used as `target`; realized residual = target - t60.
    - Format: Excel .xlsx.

  Extrernal Data.xlsx  (spelling "Extrernal" matches the file on disk)
    - What it is: Monthly macro / market features that may help predict the
      residual (target minus T60).
    - Required columns (before rename): Date, !MTR, #MTR, 3MTR,
      Bond Yield Change, Advance Decline, Dollar Index, GDP Growth, Inflation, Vix.
      These are renamed internally to: mtr_1m, mtr_2m, mtr_3m, bond_yield_change,
      advance_decline, dollar_index, gdp_growth, inflation, vix.
    - Format: Excel .xlsx. Cells may be NaN; see MISSING DATA below.

  OPTIONAL RESEARCH HARNESS (rare):
    If --input-dir is exactly this script's directory, and you do not pass
    --start-date / --end-date, the loader first tries `import train` and
    `train.load_inputs(...)` so the long panel matches a shared backtest harness
    when a compatible `train` module exists on the import path. If that fails,
    the three Excel files above are loaded instead.

PRIMARY OUTPUT FOLDER (default; override with --output-root):
  /Users/arjundivecha/Dropbox/AAA Backup/A Complete/T2 Factor Timing Fuzzy/output/nudge_analysis/<UTC_TIMESTAMP>/
  Example timestamp folder: 20260319T162514Z (UTC time when the run started).

  Per run, the program writes:

  nudge_rows.csv   OR   nudge_rows.parquet   (--format csv|parquet)
    - Long panel: one row per factor per month in the test window. Includes t60,
      target, realized/predicted residual, tilt weight, applied nudge, final
      prediction, ranks, portfolio weights and contributions, sign-correct flags,
      and whether the nudge reduced absolute error vs raw t60.

  factor_summary.csv
    - One row per factor: counts, average nudge, sign-correct rate, error
      improvement rate, etc.

  factor_style_summary.csv
    - Same style of aggregates grouped by factor_style (suffix after the last
      underscore in the factor name, e.g. _CS / _TS).

  regime_summary.csv
    - Each numeric macro feature split into quantile buckets (up to 5); stats
      on nudges and rank IC deltas within each bucket.

  month_summary.csv
    - One row per month: Spearman rank IC (t60 vs target, final vs target),
      portfolio returns (power-weighted long book), average nudge, etc.

  global_findings.md
    - Short markdown report: headline metrics, top factors/styles, regime
      highlights, and plain-language conclusions.

  run_config.json
    - Run timestamp, paths, full RunConfig, validation constants, data window,
      aggregate Sharpe/rank IC/error improvement, and paths to all artifacts.

VERSION: 1.1
LAST UPDATED: 2026-03-19
AUTHOR: Arjun Divecha

VERSION HISTORY:
  1.2 (2026-03-19) — Defaults updated to full-history `T60 old.xlsx`,
    next-month alignment, current external column names, and restored
    external diff family.
  1.1 (2026-03-19) — Documentation aligned with T2 Factor Timing Fuzzy paths,
    inputs/outputs, optional train harness, and missing-data behavior.
  1.0 — Initial nudge analysis tool (residual Ridge + walk-forward).

DESCRIPTION (simple terms):
  You have monthly factor forecasts (T60) and a target series (optimizer).
  The "error" each month is roughly target minus T60. This program asks:
  can we use public macro columns plus which factor we are looking at to
  predict that error, and then add a small correction ("nudge") to T60?

  For each test month, it uses only *past* months to train a Ridge regression
  (walk-forward). The model predicts the residual; the nudge is
  tilt_weight × predicted residual, added to t60. It then builds a concentrated
  long-only portfolio (top names by forecast, power-weighted) using both raw
  t60 and nudged forecasts, and compares Sharpe, rank IC, and how often the
  nudge shrinks forecast error. Results go to a new timestamped folder so runs
  do not overwrite each other.

MISSING DATA AND IMPUTATION:
  - Country-style "fill missing country with cross-country mean" is not used
    here; the panel is factor × month, not a country matrix.
  - Macro numeric inputs: sklearn SimpleImputer(strategy="median") inside the
    Ridge pipeline fills missing values when fitting/predicting. Infinite
    values are replaced with NaN before imputation. There is no separate log
    file of which cells were imputed (see NOTES).
  - Rows with missing t60 or target are dropped when building the panel.

DATA QUALITY AND ERRORS:
  - Missing input files, missing required columns after rename, or an empty
    panel after date filters cause a clear error (no silent fallback).
  - If no test rows are produced (e.g. range too short), the run fails with an
    explicit message.

VALIDATION / PARITY (default full-sample repo run only):
  When input_dir, output_root, and date filters match the defaults, raw T60
  baseline Sharpe and mean rank IC are checked against locked-in reference
  numbers (RAW_BASELINE_VALIDATION_TARGET) so research results stay reproducible.
  Custom directories or date windows skip this check.

DEPENDENCIES:
  - pandas, numpy
  - scikit-learn (Pipeline, ColumnTransformer, SimpleImputer, StandardScaler,
    OneHotEncoder, Ridge)
  - openpyxl (read .xlsx)
  - pyarrow or fastparquet (only if --format parquet)

USAGE:
  python nudge.py [--input-dir PATH] [--output-root PATH] [--format csv|parquet]
                  [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

KEY SETTINGS (see constants below; not all are CLI flags):
  - Portfolio mode: long_only_conc_p6_k7 → power 6 on top forecasts, top 12 names.
  - TARGET_SHIFT_MONTHS: 1 (T60[t] scored on the following realized month).
  - MACRO_LAG_MONTHS: 1 (macro table shifted so features are known with lag).
  - MIN_TRAIN_MONTHS: 1 so the full history is used in expanding-window mode.
  - RIDGE_ALPHA, TILT_WEIGHT, SEED: fixed for reproducibility.

NOTES:
  - Validation display targets in VALIDATION_TARGET are for reporting; strict
    parity uses RAW_BASELINE_VALIDATION_TARGET on the default repo run.
  - Hardware: CPU-only; suitable for Apple Silicon; no GPU used.
=============================================================================
"""
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "nudge_analysis"

T60_FILENAME = "T60 old.xlsx"
T2_FILENAME = "T2_Optimizer.xlsx"
EXTERNAL_FILENAME = "Extrernal Data.xlsx"

MODEL_NAME = "residual_tilt_core_w01"
PORTFOLIO_MODE = "long_only_conc_p6_k7"
TARGET_SHIFT_MONTHS = 1
MACRO_LAG_MONTHS = 1
TRAIN_WINDOW = 0
MIN_TRAIN_MONTHS = 1
TOP_K = 12
SEED = 7
RIDGE_ALPHA = 10.0
TILT_WEIGHT = 0.1

EXTERNAL_CORE_FEATURES = [
    "1MTR",
    "mtr_3m",
    "12MTR",
    "bond_yield_change",
    "advance_decline",
    "dollar_index",
    "gdp_growth",
    "inflation",
    "vix",
]
EXTERNAL_DIFF_FEATURES = [
    "1MTR_diff_1",
    "1MTR_diff_3",
    "mtr_3m_diff_1",
    "mtr_3m_diff_3",
    "12MTR_diff_1",
    "12MTR_diff_3",
    "bond_yield_change_diff_1",
    "bond_yield_change_diff_3",
    "advance_decline_diff_1",
    "advance_decline_diff_3",
    "dollar_index_diff_1",
    "dollar_index_diff_3",
    "gdp_growth_diff_1",
    "gdp_growth_diff_3",
    "inflation_diff_1",
    "inflation_diff_3",
    "vix_diff_1",
    "vix_diff_3",
]
EXTERNAL_FULL_FEATURES = [*EXTERNAL_CORE_FEATURES, *EXTERNAL_DIFF_FEATURES]
NUMERIC_FEATURES = [*EXTERNAL_CORE_FEATURES]
CATEGORICAL_FEATURES = ["factor", "factor_base", "factor_style"]
OUTPUT_NUMERIC_COLUMNS = [
    "t60",
    "target",
    "realized_residual",
    "predicted_residual",
    "tilt_weight",
    "applied_nudge",
    "final_prediction",
    "prediction_rank_t60",
    "prediction_rank_final",
    "target_rank",
    "nudge_sign",
    "nudge_abs",
    "weight_t60",
    "weight_final",
    "portfolio_return_t60",
    "portfolio_return_final",
]
VALIDATION_TARGET = {
    "sharpe_ann": 0.8537,
    "mean_rank_ic": 0.0214,
}
RAW_BASELINE_VALIDATION_TARGET = {
    "sharpe_ann": 0.8513046701233,
    "mean_rank_ic": 0.02128394976137994,
}
EPS = 1e-12

# =============================================================================
# CONFIGURATION — defaults for one run (see RunConfig; not all exposed on CLI)
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    model_name: str = MODEL_NAME
    portfolio_mode: str = PORTFOLIO_MODE
    target_shift_months: int = TARGET_SHIFT_MONTHS
    macro_lag_months: int = MACRO_LAG_MONTHS
    train_window: int = TRAIN_WINDOW
    min_train_months: int = MIN_TRAIN_MONTHS
    top_k: int = TOP_K
    seed: int = SEED
    ridge_alpha: float = RIDGE_ALPHA
    tilt_weight: float = TILT_WEIGHT
    numeric_features: Tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical_features: Tuple[str, ...] = tuple(CATEGORICAL_FEATURES)


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze month-by-month external nudges for residual-tilt T60.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args()


def make_one_hot() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


# =============================================================================
# INPUT FILES → LONG PANEL (factor × month + macro features)
# =============================================================================

def read_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t60_path = input_dir / T60_FILENAME
    t2_path = input_dir / T2_FILENAME
    external_path = input_dir / EXTERNAL_FILENAME
    for path in (t60_path, t2_path, external_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    t60 = pd.read_excel(t60_path)
    t2 = pd.read_excel(t2_path)
    external = pd.read_excel(external_path)
    return t60, t2, external


def normalize_external_columns(external: pd.DataFrame) -> pd.DataFrame:
    renamed = external.rename(
        columns={
            "!MTR": "1MTR",
            "3MTR": "mtr_3m",
            "#MTR": "12MTR",
            "Bond Yield Change": "bond_yield_change",
            "Advance Decline": "advance_decline",
            "Dollar Index": "dollar_index",
            "GDP Growth": "gdp_growth",
            "Inflation": "inflation",
            "Vix": "vix",
        }
    )
    missing = [column for column in NUMERIC_FEATURES if column not in renamed.columns]
    if missing:
        raise RuntimeError(f"External workbook is missing required columns after normalization: {missing}")
    return renamed


def load_panel(
    input_dir: Path,
    config: RunConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    # When using the repo's default data directory with the full sample,
    # defer to the shared harness loader so this analyzer stays byte-for-byte
    # aligned with the research backtest's panel construction.
    if input_dir.resolve() == ROOT.resolve() and start_date is None and end_date is None and (ROOT / "train.py").exists():
        try:
            import train

            panel = train.load_inputs(config.target_shift_months, config.macro_lag_months).copy()
            required_columns = ["Date", "factor", "factor_base", "factor_style", "t60", "target", *NUMERIC_FEATURES]
            missing_columns = [column for column in required_columns if column not in panel.columns]
            if missing_columns:
                raise RuntimeError(f"Shared harness panel is missing required columns: {missing_columns}")
            return panel[required_columns].copy().sort_values(["factor", "Date"]).reset_index(drop=True)
        except Exception:
            # Fall back to the standalone loader below if the shared harness
            # is unavailable or changes shape.
            pass

    t60, t2, external = read_inputs(input_dir)
    t60["Date"] = pd.to_datetime(t60["Date"])
    t2["Date"] = pd.to_datetime(t2["Date"])
    external["Date"] = pd.to_datetime(external["Date"])

    factor_cols = [column for column in t60.columns if column != "Date" and column in set(t2.columns)]
    if not factor_cols:
        raise RuntimeError("No overlapping factor columns were found between T60 and T2_Optimizer.")

    if config.target_shift_months:
        t2 = t2.copy()
        t2["Date"] = t2["Date"] - pd.DateOffset(months=config.target_shift_months)

    external = normalize_external_columns(external)
    external = external.sort_values("Date").reset_index(drop=True)
    if config.macro_lag_months:
        external[NUMERIC_FEATURES] = external[NUMERIC_FEATURES].shift(config.macro_lag_months)
    for column in EXTERNAL_CORE_FEATURES:
        external[f"{column}_diff_1"] = external[column].diff(1)
        external[f"{column}_diff_3"] = external[column].diff(3)

    t60_long = t60[["Date"] + factor_cols].melt("Date", var_name="factor", value_name="t60")
    t2_long = t2[["Date"] + factor_cols].melt("Date", var_name="factor", value_name="target")
    panel = (
        t60_long.merge(t2_long, on=["Date", "factor"], how="inner")
        .merge(external[["Date"] + EXTERNAL_FULL_FEATURES], on="Date", how="left")
        .sort_values(["Date", "factor"])
        .reset_index(drop=True)
    )

    panel["factor_style"] = panel["factor"].str.extract(r"_([^_]+)$", expand=False).fillna("UNK")
    panel["factor_base"] = panel["factor"].str.replace(r"_([^_]+)$", "", regex=True).str.strip()
    panel = panel.dropna(subset=["t60", "target"]).reset_index(drop=True)

    if start_date:
        panel = panel[panel["Date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        panel = panel[panel["Date"] <= pd.Timestamp(end_date)].copy()

    if panel.empty:
        raise RuntimeError("Panel is empty after applying date filters.")

    return panel.sort_values(["Date", "factor"]).reset_index(drop=True)


# =============================================================================
# PORTFOLIO WEIGHTS AND PERFORMANCE METRICS
# =============================================================================

def parse_concentration_mode(mode: str, fallback_top_k: int) -> Tuple[float, int]:
    if not mode.startswith("long_only_conc_p"):
        raise ValueError(f"Unsupported portfolio mode for this tool: {mode}")
    suffix = mode[len("long_only_conc_p") :]
    if "_k" in suffix:
        power_text, top_k_text = suffix.split("_k", 1)
        power = float(power_text)
        effective_top_k = int(top_k_text)
    else:
        power = float(suffix)
        effective_top_k = fallback_top_k
    return power, effective_top_k


def portfolio_weights(predictions: pd.Series, mode: str, top_k: int) -> np.ndarray:
    power, effective_top_k = parse_concentration_mode(mode, top_k)
    values = predictions.to_numpy(dtype=float)
    if len(values) == 0:
        return np.array([], dtype=float)
    order = np.argsort(values)[::-1]
    picks = order[: min(effective_top_k, len(order))]
    top_values = values[picks]
    top_values = top_values - np.nanmin(top_values)
    if np.all(np.abs(top_values) < EPS):
        top_values = np.ones_like(top_values)
    top_values = top_values ** power
    weights = np.zeros(len(values), dtype=float)
    weights[picks] = top_values
    total = weights.sum()
    if total <= EPS:
        weights[picks] = 1.0 / len(picks)
        return weights
    return weights / total


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method="spearman"))


def annualized_sharpe(monthly_returns: pd.Series) -> float:
    mean_monthly = monthly_returns.mean()
    vol_monthly = monthly_returns.std(ddof=1)
    if pd.isna(vol_monthly) or abs(vol_monthly) <= EPS:
        return float("nan")
    return float(np.sqrt(12.0) * mean_monthly / vol_monthly)


def compute_aggregate_results(month_summary: pd.DataFrame, rows: pd.DataFrame) -> Dict[str, float]:
    return {
        "raw_t60_sharpe": annualized_sharpe(month_summary["portfolio_return_t60"]),
        "nudged_sharpe": annualized_sharpe(month_summary["portfolio_return_final"]),
        "raw_t60_mean_rank_ic": float(month_summary["rank_ic_t60"].mean()),
        "nudged_mean_rank_ic": float(month_summary["rank_ic_final"].mean()),
        "mean_error_improvement_rate": float(rows["nudge_helped_error"].mean()),
    }


def validate_reference_parity(
    aggregate_results: Dict[str, float],
    *,
    input_dir: Path,
    output_root: Path,
    start_date: Optional[str],
    end_date: Optional[str],
) -> None:
    # Only enforce parity on the default full-history repo run. Custom input
    # paths or date filters are allowed to diverge.
    if input_dir.resolve() != ROOT.resolve():
        return
    if start_date is not None or end_date is not None:
        return
    if output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        return

    sharpe_gap = abs(aggregate_results["raw_t60_sharpe"] - RAW_BASELINE_VALIDATION_TARGET["sharpe_ann"])
    rank_ic_gap = abs(
        aggregate_results["raw_t60_mean_rank_ic"] - RAW_BASELINE_VALIDATION_TARGET["mean_rank_ic"]
    )
    if sharpe_gap > 1e-9 or rank_ic_gap > 1e-9:
        raise RuntimeError(
            "Raw T60 baseline parity failed. "
            f"Expected sharpe={RAW_BASELINE_VALIDATION_TARGET['sharpe_ann']:.10f}, "
            f"rank_ic={RAW_BASELINE_VALIDATION_TARGET['mean_rank_ic']:.10f}; "
            f"got sharpe={aggregate_results['raw_t60_sharpe']:.10f}, "
            f"rank_ic={aggregate_results['raw_t60_mean_rank_ic']:.10f}."
        )


# =============================================================================
# RIDGE MODEL — predict residual (target − t60); nudge = tilt_weight × prediction
# =============================================================================

class ResidualTiltRidgeModel:
    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        alpha: float,
        tilt_weight: float,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.alpha = float(alpha)
        self.tilt_weight = float(tilt_weight)
        self.active_numeric_features = list(self.numeric_features)
        self.active_categorical_features = list(self.categorical_features)
        self.pipeline: Pipeline | None = None

    def _build_pipeline(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
    ) -> Pipeline:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, list(numeric_features)),
                ("categorical", make_one_hot(), list(categorical_features)),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )

    def _select_columns(
        self,
        frame: pd.DataFrame,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
    ) -> pd.DataFrame:
        columns = list(categorical_features) + list(numeric_features)
        selected = frame[columns].copy()
        if numeric_features:
            selected.loc[:, list(numeric_features)] = selected[list(numeric_features)].replace([np.inf, -np.inf], np.nan)
        return selected

    def fit(self, train_frame: pd.DataFrame) -> "ResidualTiltRidgeModel":
        available_numeric = [
            column
            for column in self.numeric_features
            if column in train_frame.columns and train_frame[column].replace([np.inf, -np.inf], np.nan).notna().any()
        ]
        available_categorical = [column for column in self.categorical_features if column in train_frame.columns]
        if not available_numeric and not available_categorical:
            raise RuntimeError("No usable model features are available for this training window.")

        self.active_numeric_features = available_numeric
        self.active_categorical_features = available_categorical
        self.pipeline = self._build_pipeline(self.active_numeric_features, self.active_categorical_features)

        residual = train_frame["target"].to_numpy(dtype=float) - train_frame["t60"].to_numpy(dtype=float)
        self.pipeline.fit(
            self._select_columns(train_frame, self.active_numeric_features, self.active_categorical_features),
            residual,
        )
        return self

    def predict_components(self, test_frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Model must be fit before predict.")
        predicted_residual = np.asarray(
            self.pipeline.predict(
                self._select_columns(test_frame, self.active_numeric_features, self.active_categorical_features)
            ),
            dtype=float,
        )
        applied_nudge = self.tilt_weight * predicted_residual
        final_prediction = test_frame["t60"].to_numpy(dtype=float) + applied_nudge
        return predicted_residual, applied_nudge, final_prediction


# =============================================================================
# WALK-FORWARD BACKTEST — train on past months, score current month
# =============================================================================

def walk_forward_dates(dates: Sequence[pd.Timestamp], train_window: int) -> List[Tuple[pd.Timestamp, List[pd.Timestamp]]]:
    schedule = []
    ordered = list(dates)
    for index, test_date in enumerate(ordered):
        train_end = index
        train_start = 0 if train_window <= 0 else max(0, train_end - train_window)
        schedule.append((test_date, ordered[train_start:train_end]))
    return schedule


def run_analysis(panel: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    np.random.seed(config.seed)
    dates = sorted(pd.to_datetime(panel["Date"].unique()))
    rows: List[pd.DataFrame] = []

    for index, (test_date, train_dates) in enumerate(walk_forward_dates(dates, config.train_window)):
        if index < config.min_train_months:
            continue
        if not train_dates:
            continue

        train_frame = panel[panel["Date"].isin(train_dates)].copy()
        test_frame = panel[panel["Date"] == test_date].copy()
        if train_frame.empty or test_frame.empty:
            continue

        model = ResidualTiltRidgeModel(
            numeric_features=config.numeric_features,
            categorical_features=config.categorical_features,
            alpha=config.ridge_alpha,
            tilt_weight=config.tilt_weight,
        )
        model.fit(train_frame)
        predicted_residual, applied_nudge, final_prediction = model.predict_components(test_frame)

        month_rows = test_frame[
            ["Date", "factor", "factor_base", "factor_style", "t60", "target"] + list(config.numeric_features)
        ].copy()
        month_rows["realized_residual"] = month_rows["target"] - month_rows["t60"]
        month_rows["predicted_residual"] = predicted_residual
        month_rows["tilt_weight"] = config.tilt_weight
        month_rows["applied_nudge"] = applied_nudge
        month_rows["final_prediction"] = final_prediction
        month_rows["prediction_rank_t60"] = month_rows["t60"].rank(method="average", ascending=False)
        month_rows["prediction_rank_final"] = month_rows["final_prediction"].rank(method="average", ascending=False)
        month_rows["target_rank"] = month_rows["target"].rank(method="average", ascending=False)
        month_rows["nudge_sign"] = np.sign(month_rows["applied_nudge"]).astype(int)
        month_rows["nudge_abs"] = month_rows["applied_nudge"].abs()
        month_rows["nudge_sign_correct"] = (
            np.sign(month_rows["predicted_residual"]).astype(int)
            == np.sign(month_rows["realized_residual"]).astype(int)
        )
        month_rows["nudge_helped_error"] = (
            (month_rows["target"] - month_rows["final_prediction"]).abs()
            < (month_rows["target"] - month_rows["t60"]).abs()
        )

        month_rows["weight_t60"] = portfolio_weights(month_rows["t60"], config.portfolio_mode, config.top_k)
        month_rows["weight_final"] = portfolio_weights(month_rows["final_prediction"], config.portfolio_mode, config.top_k)
        month_rows["selected_in_portfolio_t60"] = month_rows["weight_t60"] > 0.0
        month_rows["selected_in_portfolio_final"] = month_rows["weight_final"] > 0.0
        month_rows["portfolio_return_t60"] = month_rows["weight_t60"] * month_rows["target"]
        month_rows["portfolio_return_final"] = month_rows["weight_final"] * month_rows["target"]
        rows.append(month_rows)

    if not rows:
        raise RuntimeError("No test rows were generated. Check the date range and minimum train-month setting.")

    return pd.concat(rows, ignore_index=True).sort_values(["Date", "factor"]).reset_index(drop=True)


# =============================================================================
# AGGREGATIONS — by factor, style, month, and macro regime bucket
# =============================================================================

def summarize_by_group(rows: pd.DataFrame, group_column: str) -> pd.DataFrame:
    summary = (
        rows.groupby(group_column, dropna=False)
        .agg(
            observation_count=("factor", "size"),
            avg_nudge=("applied_nudge", "mean"),
            median_nudge=("applied_nudge", "median"),
            avg_abs_nudge=("nudge_abs", "mean"),
            pct_positive_nudges=("applied_nudge", lambda series: float((series > 0).mean())),
            pct_negative_nudges=("applied_nudge", lambda series: float((series < 0).mean())),
            sign_correct_rate=("nudge_sign_correct", "mean"),
            error_improvement_rate=("nudge_helped_error", "mean"),
            avg_realized_residual=("realized_residual", "mean"),
            avg_predicted_residual=("predicted_residual", "mean"),
        )
        .reset_index()
        .sort_values(["avg_abs_nudge", "observation_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def summarize_months(rows: pd.DataFrame) -> pd.DataFrame:
    month_rows = []
    for date, frame in rows.groupby("Date", sort=True):
        rank_ic_t60 = spearman_corr(frame["t60"], frame["target"])
        rank_ic_final = spearman_corr(frame["final_prediction"], frame["target"])
        month_rows.append(
            {
                "Date": date,
                "rank_ic_t60": rank_ic_t60,
                "rank_ic_final": rank_ic_final,
                "rank_ic_delta": rank_ic_final - rank_ic_t60 if pd.notna(rank_ic_t60) and pd.notna(rank_ic_final) else np.nan,
                "portfolio_return_t60": float(frame["portfolio_return_t60"].sum()),
                "portfolio_return_final": float(frame["portfolio_return_final"].sum()),
                "portfolio_return_delta": float(frame["portfolio_return_final"].sum() - frame["portfolio_return_t60"].sum()),
                "avg_nudge": float(frame["applied_nudge"].mean()),
                "avg_abs_nudge": float(frame["nudge_abs"].mean()),
                "pct_positive_nudges": float((frame["applied_nudge"] > 0).mean()),
                "pct_negative_nudges": float((frame["applied_nudge"] < 0).mean()),
                "sign_correct_rate": float(frame["nudge_sign_correct"].mean()),
                "error_improvement_rate": float(frame["nudge_helped_error"].mean()),
            }
        )

    summary = pd.DataFrame(month_rows).sort_values("Date").reset_index(drop=True)
    return summary


def build_regime_summary(rows: pd.DataFrame, month_summary: pd.DataFrame) -> pd.DataFrame:
    month_summary = month_summary.copy()
    month_summary["Date"] = pd.to_datetime(month_summary["Date"])
    month_values = rows.groupby("Date", sort=True)[NUMERIC_FEATURES].first().reset_index()
    all_rows = []

    for variable in NUMERIC_FEATURES:
        variable_months = month_values[["Date", variable]].dropna().copy()
        if variable_months.empty:
            continue

        unique_count = int(variable_months[variable].nunique())
        bucket_count = min(5, unique_count)
        if bucket_count <= 1:
            variable_months["bucket"] = "Q1"
        else:
            variable_months["bucket_interval"] = pd.qcut(variable_months[variable], q=bucket_count, duplicates="drop")
            interval_codes = variable_months["bucket_interval"].cat.codes + 1
            interval_total = int(interval_codes.max())
            variable_months["bucket"] = interval_codes.map(lambda code: f"Q{int(code)}")
            bucket_count = interval_total

        for bucket, bucket_months in variable_months.groupby("bucket", sort=True):
            dates = set(bucket_months["Date"])
            bucket_rows = rows[rows["Date"].isin(dates)]
            bucket_month_summary = month_summary[month_summary["Date"].isin(dates)]
            all_rows.append(
                {
                    "regime_variable": variable,
                    "bucket": bucket,
                    "bucket_count": bucket_count,
                    "bucket_min": float(bucket_months[variable].min()),
                    "bucket_max": float(bucket_months[variable].max()),
                    "month_count": int(bucket_months.shape[0]),
                    "observation_count": int(bucket_rows.shape[0]),
                    "avg_nudge": float(bucket_rows["applied_nudge"].mean()),
                    "median_nudge": float(bucket_rows["applied_nudge"].median()),
                    "avg_abs_nudge": float(bucket_rows["nudge_abs"].mean()),
                    "sign_correct_rate": float(bucket_rows["nudge_sign_correct"].mean()),
                    "error_improvement_rate": float(bucket_rows["nudge_helped_error"].mean()),
                    "avg_realized_residual": float(bucket_rows["realized_residual"].mean()),
                    "avg_portfolio_delta_return": float(bucket_month_summary["portfolio_return_delta"].mean()),
                    "avg_rank_ic_delta": float(bucket_month_summary["rank_ic_delta"].mean()),
                }
            )

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "regime_variable",
                "bucket",
                "bucket_count",
                "bucket_min",
                "bucket_max",
                "month_count",
                "observation_count",
                "avg_nudge",
                "median_nudge",
                "avg_abs_nudge",
                "sign_correct_rate",
                "error_improvement_rate",
                "avg_realized_residual",
                "avg_portfolio_delta_return",
                "avg_rank_ic_delta",
            ]
        )

    summary = pd.DataFrame(all_rows)
    summary["bucket_index"] = summary["bucket"].str.extract(r"(\d+)").astype(int)
    return summary.sort_values(["regime_variable", "bucket_index"]).drop(columns=["bucket_index"]).reset_index(drop=True)


# =============================================================================
# REPORT HELPERS — number/percent formatting and auto-generated conclusions
# =============================================================================

def pct_text(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def num_text(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_list(df: pd.DataFrame, label_col: str, value_col: str, count: int, digits: int = 4) -> List[str]:
    if df.empty:
        return ["- none"]
    lines = []
    for _, row in df.head(count).iterrows():
        lines.append(f"- `{row[label_col]}`: {num_text(float(row[value_col]), digits)}")
    return lines


def build_conclusions(month_summary: pd.DataFrame, rows: pd.DataFrame) -> Dict[str, str]:
    sharpe_t60 = annualized_sharpe(month_summary["portfolio_return_t60"])
    sharpe_final = annualized_sharpe(month_summary["portfolio_return_final"])
    sharpe_delta = sharpe_final - sharpe_t60 if pd.notna(sharpe_t60) and pd.notna(sharpe_final) else float("nan")
    rank_ic_t60 = float(month_summary["rank_ic_t60"].mean())
    rank_ic_final = float(month_summary["rank_ic_final"].mean())
    rank_ic_delta = rank_ic_final - rank_ic_t60
    error_help_rate = float(rows["nudge_helped_error"].mean())

    if rank_ic_delta > 0.0005:
        ranking = "Nudges appear to add some cross-sectional ranking value beyond raw T60."
    elif rank_ic_delta < -0.0005:
        ranking = "Nudges appear to hurt cross-sectional ranking relative to raw T60."
    else:
        ranking = "Nudges leave cross-sectional ranking close to raw T60."

    if error_help_rate > 0.52:
        magnitude = "Nudges improve forecast magnitude more often than they hurt it."
    elif error_help_rate < 0.48:
        magnitude = "Nudges hurt forecast magnitude more often than they help it."
    else:
        magnitude = "Nudges have only a small and mixed effect on forecast magnitude."

    if pd.notna(sharpe_delta) and sharpe_delta > 0.05 and rank_ic_delta <= 0.0005:
        risk = "Most of the benefit looks like conviction or risk management, not better rank ordering."
    elif pd.notna(sharpe_delta) and sharpe_delta > 0.0 and rank_ic_delta > 0.0005:
        risk = "Nudges improve both portfolio quality and ranking, suggesting usable external conditioning."
    elif pd.notna(sharpe_delta) and sharpe_delta < 0.0:
        risk = "The nudges reduce portfolio quality overall, so the external overlay is not helping this fixed model."
    else:
        risk = "The portfolio impact is modest, so the overlay mainly acts as a small correction layer."

    stable_factor_count = int(
        rows.groupby("factor")["nudge_helped_error"].mean().pipe(lambda s: ((s > 0.55) | (s < 0.45)).sum())
    )
    if stable_factor_count >= 3:
        generalization = "There are repeated factor-level patterns worth inspecting for generalization."
    else:
        generalization = "The nudge patterns look weak or sparse, so strong generalization claims are not supported yet."

    if pd.notna(sharpe_delta) and sharpe_delta > 0.0 and abs(rank_ic_delta) <= 0.0005:
        what_it_does = "External data is mostly modulating conviction around T60 rather than replacing or reordering it."
    elif rank_ic_delta > 0.0005:
        what_it_does = "External data is contributing some directional ranking information beyond raw T60."
    else:
        what_it_does = "In this linear residual form, external data behaves like a small and inconsistent correction to T60."

    return {
        "ranking": ranking,
        "magnitude": magnitude,
        "risk": risk,
        "generalization": generalization,
        "what_it_does": what_it_does,
    }


# =============================================================================
# global_findings.md — ranked tables and narrative sections
# =============================================================================

def write_report(
    output_dir: Path,
    config: RunConfig,
    panel: pd.DataFrame,
    rows: pd.DataFrame,
    factor_summary: pd.DataFrame,
    factor_style_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    month_summary: pd.DataFrame,
) -> None:
    aggregate_results = compute_aggregate_results(month_summary, rows)
    sharpe_t60 = aggregate_results["raw_t60_sharpe"]
    sharpe_final = aggregate_results["nudged_sharpe"]
    rank_ic_t60 = aggregate_results["raw_t60_mean_rank_ic"]
    rank_ic_final = aggregate_results["nudged_mean_rank_ic"]
    conclusions = build_conclusions(month_summary, rows)

    positive_factors = factor_summary.sort_values("avg_nudge", ascending=False).head(5)
    negative_factors = factor_summary.sort_values("avg_nudge", ascending=True).head(5)
    helpful_factors = factor_summary.sort_values("error_improvement_rate", ascending=False).head(5)
    harmful_factors = factor_summary.sort_values("error_improvement_rate", ascending=True).head(5)
    strong_regimes = regime_summary.sort_values("avg_abs_nudge", ascending=False).head(5)
    accurate_regimes = regime_summary[regime_summary["month_count"] >= 8].sort_values(
        "sign_correct_rate", ascending=False
    ).head(5)
    positive_months = month_summary.sort_values("avg_nudge", ascending=False).head(5)
    negative_months = month_summary.sort_values("avg_nudge", ascending=True).head(5)

    lines = [
        "# Residual-Tilt Nudge Analysis",
        "",
        "## Objective",
        "Analyze one fixed residual-tilt model that starts from raw `t60` and applies a small external-data nudge.",
        "",
        "## Fixed Model",
        f"- Model name: `{config.model_name}`",
        "- Base forecast: raw `t60`",
        "- Residual target: `target - t60`",
        f"- Residual model: Ridge(alpha={config.ridge_alpha})",
        f"- Tilt weight: `{config.tilt_weight}`",
        f"- Final forecast: `t60 + {config.tilt_weight} * predicted_residual`",
        f"- Portfolio readout: `{config.portfolio_mode}`",
        "",
        "## Walk-Forward Configuration",
        f"- target_shift_months: `{config.target_shift_months}`",
        f"- macro_lag_months: `{config.macro_lag_months}`",
        f"- train_window: `{config.train_window}`",
        f"- min_train_months: `{config.min_train_months}`",
        f"- seed: `{config.seed}`",
        "",
        "## Data Window",
        f"- Start date: `{panel['Date'].min().strftime('%Y-%m-%d')}`",
        f"- End date: `{panel['Date'].max().strftime('%Y-%m-%d')}`",
        f"- Test months: `{len(month_summary)}`",
        f"- Factor-month test observations: `{len(rows)}`",
        f"- Unique factors: `{rows['factor'].nunique()}`",
        "",
        "## Aggregate Results",
        f"- Raw T60 Sharpe: `{num_text(sharpe_t60)}`",
        f"- Nudged Sharpe: `{num_text(sharpe_final)}`",
        f"- Raw T60 mean Rank IC: `{num_text(rank_ic_t60)}`",
        f"- Nudged mean Rank IC: `{num_text(rank_ic_final)}`",
        f"- Raw T60 validation Sharpe: `{num_text(RAW_BASELINE_VALIDATION_TARGET['sharpe_ann'])}`",
        f"- Raw T60 validation mean Rank IC: `{num_text(RAW_BASELINE_VALIDATION_TARGET['mean_rank_ic'])}`",
        f"- Validation target Sharpe: `{num_text(VALIDATION_TARGET['sharpe_ann'])}`",
        f"- Validation target mean Rank IC: `{num_text(VALIDATION_TARGET['mean_rank_ic'])}`",
        "",
        "## Top Positive-Nudge Factors",
    ]
    lines.extend(markdown_list(positive_factors, "factor", "avg_nudge", 5))
    lines.extend(
        [
            "",
            "## Top Negative-Nudge Factors",
        ]
    )
    lines.extend(markdown_list(negative_factors, "factor", "avg_nudge", 5))
    lines.extend(
        [
            "",
            "## Factors Where Nudges Help Most Often",
        ]
    )
    lines.extend(
        [
            f"- `{row['factor']}`: error improvement {pct_text(float(row['error_improvement_rate']))}"
            for _, row in helpful_factors.iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Factors Where Nudges Hurt Most Often",
        ]
    )
    lines.extend(
        [
            f"- `{row['factor']}`: error improvement {pct_text(float(row['error_improvement_rate']))}"
            for _, row in harmful_factors.iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Factor-Style View",
        ]
    )
    lines.extend(
        [
            f"- `{row['factor_style']}`: avg nudge {num_text(float(row['avg_nudge']))}, error improvement {pct_text(float(row['error_improvement_rate']))}"
            for _, row in factor_style_summary.head(5).iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Regimes With Largest Nudges",
        ]
    )
    lines.extend(
        [
            f"- `{row['regime_variable']} {row['bucket']}`: avg abs nudge {num_text(float(row['avg_abs_nudge']))}, avg rank IC delta {num_text(float(row['avg_rank_ic_delta']))}"
            for _, row in strong_regimes.iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Regimes With Most Accurate Nudge Direction",
        ]
    )
    lines.extend(
        [
            f"- `{row['regime_variable']} {row['bucket']}`: sign-correct rate {pct_text(float(row['sign_correct_rate']))}, month count {int(row['month_count'])}"
            for _, row in accurate_regimes.iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Months With Largest Positive Average Nudge",
        ]
    )
    lines.extend(
        [
            f"- `{row['Date'].strftime('%Y-%m-%d')}`: avg nudge {num_text(float(row['avg_nudge']))}, portfolio delta {num_text(float(row['portfolio_return_delta']))}"
            for _, row in positive_months.iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Months With Largest Negative Average Nudge",
        ]
    )
    lines.extend(
        [
            f"- `{row['Date'].strftime('%Y-%m-%d')}`: avg nudge {num_text(float(row['avg_nudge']))}, portfolio delta {num_text(float(row['portfolio_return_delta']))}"
            for _, row in negative_months.iterrows()
        ]
        or ["- none"]
    )
    lines.extend(
        [
            "",
            "## Conclusion",
            f"- Ranking: {conclusions['ranking']}",
            f"- Magnitude: {conclusions['magnitude']}",
            f"- Risk control: {conclusions['risk']}",
            f"- Generalization: {conclusions['generalization']}",
            f"- What external data seems to do: {conclusions['what_it_does']}",
            "",
        ]
    )

    (output_dir / "global_findings.md").write_text("\n".join(lines))


# =============================================================================
# WRITE ARTIFACTS — panel export (csv/parquet) and run_config.json
# =============================================================================

def write_rows(rows: pd.DataFrame, output_dir: Path, file_format: str) -> Path:
    if file_format == "parquet":
        path = output_dir / "nudge_rows.parquet"
        rows.to_parquet(path, index=False)
        return path
    path = output_dir / "nudge_rows.csv"
    rows.to_csv(path, index=False)
    return path


def serialize_config(
    output_dir: Path,
    input_dir: Path,
    config: RunConfig,
    panel: pd.DataFrame,
    rows: pd.DataFrame,
    month_summary: pd.DataFrame,
    row_output_path: Path,
) -> None:
    aggregate_results = compute_aggregate_results(month_summary, rows)
    payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_files": {
            "t60": str(input_dir / T60_FILENAME),
            "t2_optimizer": str(input_dir / T2_FILENAME),
            "external": str(input_dir / EXTERNAL_FILENAME),
        },
        "config": asdict(config),
        "validation_target": VALIDATION_TARGET,
        "raw_baseline_validation_target": RAW_BASELINE_VALIDATION_TARGET,
        "data_window": {
            "panel_start": panel["Date"].min().strftime("%Y-%m-%d"),
            "panel_end": panel["Date"].max().strftime("%Y-%m-%d"),
            "test_month_count": int(month_summary.shape[0]),
            "factor_month_count": int(rows.shape[0]),
            "factor_count": int(rows["factor"].nunique()),
        },
        "aggregate_results": aggregate_results,
        "artifacts": {
            "nudge_rows": str(row_output_path),
            "factor_summary": str(output_dir / "factor_summary.csv"),
            "factor_style_summary": str(output_dir / "factor_style_summary.csv"),
            "regime_summary": str(output_dir / "regime_summary.csv"),
            "month_summary": str(output_dir / "month_summary.csv"),
            "global_findings": str(output_dir / "global_findings.md"),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig()
    panel = load_panel(input_dir, config, start_date=args.start_date, end_date=args.end_date)
    rows = run_analysis(panel, config)

    month_summary = summarize_months(rows)
    factor_summary = summarize_by_group(rows, "factor")
    factor_style_summary = summarize_by_group(rows, "factor_style")
    regime_summary = build_regime_summary(rows, month_summary)
    aggregate_results = compute_aggregate_results(month_summary, rows)
    validate_reference_parity(
        aggregate_results,
        input_dir=input_dir,
        output_root=output_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    row_output_path = write_rows(rows, output_dir, args.format)
    factor_summary.to_csv(output_dir / "factor_summary.csv", index=False)
    factor_style_summary.to_csv(output_dir / "factor_style_summary.csv", index=False)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    month_summary.to_csv(output_dir / "month_summary.csv", index=False)

    write_report(
        output_dir=output_dir,
        config=config,
        panel=panel,
        rows=rows,
        factor_summary=factor_summary,
        factor_style_summary=factor_style_summary,
        regime_summary=regime_summary,
        month_summary=month_summary,
    )
    serialize_config(output_dir, input_dir, config, panel, rows, month_summary, row_output_path)

    print(
        "Wrote nudge analysis to "
        f"{output_dir} "
        f"(raw_sharpe={aggregate_results['raw_t60_sharpe']:.4f}, "
        f"nudged_sharpe={aggregate_results['nudged_sharpe']:.4f}, "
        f"raw_rank_ic={aggregate_results['raw_t60_mean_rank_ic']:.4f}, "
        f"nudged_rank_ic={aggregate_results['nudged_mean_rank_ic']:.4f})"
    )


if __name__ == "__main__":
    main()
