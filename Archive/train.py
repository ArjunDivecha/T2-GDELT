#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import inspect
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


ROOT = Path(__file__).resolve().parent
T60_PATH = ROOT / "T60.xlsx"
T2_PATH = ROOT / "T2_Optimizer.xlsx"
EXTERNAL_PATH = ROOT / "Extrernal Data.xlsx"
OUTPUT_DIR = ROOT / "output"

PRIMARY_PORTFOLIO = "long_short"
DEFAULT_TARGET_SHIFT_MONTHS = 0
DEFAULT_MACRO_LAG_MONTHS = 1
DEFAULT_TRAIN_WINDOW = 0
DEFAULT_MIN_TRAIN_MONTHS = 60
DEFAULT_TOP_K = 12
DEFAULT_SEED = 7
DEFAULT_MODEL_PROFILE = "fast"
DEFAULT_N_JOBS = max(1, os.cpu_count() or 1)

EPS = 1e-12


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    portfolio: str
    builder: Callable[[], Pipeline | object | None] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor-timing Karpathy Loop harness")
    parser.add_argument("--target-shift-months", type=int, default=DEFAULT_TARGET_SHIFT_MONTHS)
    parser.add_argument("--macro-lag-months", type=int, default=DEFAULT_MACRO_LAG_MONTHS)
    parser.add_argument(
        "--train-window",
        type=int,
        default=DEFAULT_TRAIN_WINDOW,
        help="Number of prior months to train on. Use 0 to train on all available prior history.",
    )
    parser.add_argument("--min-train-months", type=int, default=DEFAULT_MIN_TRAIN_MONTHS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Parallel workers for model fitting/scoring. Use 1 for deterministic single-thread runs.",
    )
    parser.add_argument(
        "--model-profile",
        choices=["fast", "full"],
        default=DEFAULT_MODEL_PROFILE,
        help="fast keeps the loop responsive; full adds slower tree/boosting candidates",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    return parser.parse_args()


def make_one_hot() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:  # pragma: no cover - compatibility path
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def load_inputs(target_shift_months: int, macro_lag_months: int) -> pd.DataFrame:
    t60 = pd.read_excel(T60_PATH)
    t2 = pd.read_excel(T2_PATH)
    external = pd.read_excel(EXTERNAL_PATH)

    t60["Date"] = pd.to_datetime(t60["Date"])
    t2["Date"] = pd.to_datetime(t2["Date"])
    external["Date"] = pd.to_datetime(external["Date"])

    factor_cols = [col for col in t60.columns if col != "Date" and col in set(t2.columns)]
    if not factor_cols:
        raise RuntimeError("No overlapping factor columns were found between T60 and T2_Optimizer.")

    if target_shift_months:
        t2 = t2.copy()
        t2["Date"] = t2["Date"] - pd.DateOffset(months=target_shift_months)

    external = external.rename(
        columns={
            "!MTR": "mtr_1m",
            "#MTR": "mtr_2m",
            "3MTR": "mtr_3m",
            "Bond Yield Change": "bond_yield_change",
            "Advance Decline": "advance_decline",
            "Dollar Index": "dollar_index",
            "GDP Growth": "gdp_growth",
            "Inflation": "inflation",
            "Vix": "vix",
        }
    )

    macro_cols = [col for col in external.columns if col != "Date"]
    external = external.sort_values("Date").reset_index(drop=True)
    if macro_lag_months:
        external[macro_cols] = external[macro_cols].shift(macro_lag_months)
    for col in macro_cols:
        external[f"{col}_diff_1"] = external[col].diff()
        external[f"{col}_diff_3"] = external[col].diff(3)

    t60_long = t60[["Date"] + factor_cols].melt("Date", var_name="factor", value_name="t60")
    t2_long = t2[["Date"] + factor_cols].melt("Date", var_name="factor", value_name="target")
    panel = (
        t60_long.merge(t2_long, on=["Date", "factor"], how="inner")
        .merge(external, on="Date", how="left")
        .sort_values(["factor", "Date"])
        .reset_index(drop=True)
    )

    panel["factor_style"] = panel["factor"].str.extract(r"_([^_]+)$", expand=False).fillna("UNK")
    panel["factor_base"] = panel["factor"].str.replace(r"_([^_]+)$", "", regex=True).str.strip()
    panel["is_cs"] = (panel["factor_style"] == "CS").astype(float)
    panel["is_ts"] = (panel["factor_style"] == "TS").astype(float)
    panel["month"] = panel["Date"].dt.month
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12.0)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12.0)

    add_group_features(panel, "factor", "t60", [3, 6, 12], [6, 12])
    add_group_features(panel, "factor", "target", [3, 6, 12], [6, 12])

    # Longer lookback features for deeper momentum
    for w in [24, 36]:
        col_name = f"target_mean_{w}"
        panel[col_name] = panel.groupby("factor")["target"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(6, w // 3)).mean()
        )

    # Additional lags of t60
    panel["t60_lag_2"] = panel.groupby("factor")["t60"].shift(2)
    panel["t60_lag_3"] = panel.groupby("factor")["t60"].shift(3)
    panel["t60_lag_4"] = panel.groupby("factor")["t60"].shift(4)
    panel["t60_delta_1"] = panel["t60"] - panel["t60_lag_1"]
    panel["t60_target_gap"] = panel["t60"] - panel["target_lag_1"]
    panel["target_sign_12"] = np.sign(panel["target_mean_12"])

    panel["t60_cs_rank"] = panel.groupby("Date")["t60"].rank(pct=True)
    panel["t60_cs_z"] = group_zscore(panel.groupby("Date")["t60"])
    panel["target_lag_1_cs_z"] = group_zscore(panel.groupby("Date")["target_lag_1"])
    panel["target_mean_3_cs_z"] = group_zscore(panel.groupby("Date")["target_mean_3"])
    panel["target_mean_12_cs_z"] = group_zscore(panel.groupby("Date")["target_mean_12"])
    panel["t60_x_vix"] = panel["t60_cs_z"] * panel["vix"]
    panel["t60_x_bond_yield_change"] = panel["t60_cs_z"] * panel["bond_yield_change"]
    panel["t60_x_dollar_index"] = panel["t60_cs_z"] * panel["dollar_index"]

    # t60 squared (captures nonlinear signal strength within linear model)
    panel["t60_cs_z_sq"] = panel["t60_cs_z"] ** 2
    # t60 * momentum interaction (signal confirms momentum)
    panel["t60_x_mom3"] = panel["t60_cs_z"] * panel["target_mean_3_cs_z"].fillna(0.0)
    # t60 acceleration: change in t60 z-score
    panel["t60_cs_z_accel"] = panel["t60_cs_z"] - panel.groupby("factor")["t60_cs_z"].shift(1)
    # Cross-sectional dispersion of t60 (high dispersion = more differentiation opportunity)
    panel["t60_cs_dispersion"] = panel.groupby("Date")["t60"].transform("std")
    # Target dispersion (lagged) - captures regime volatility
    panel["target_cs_dispersion"] = panel.groupby("Date")["target_lag_1"].transform("std")
    # t60 sign agreement with lagged momentum
    panel["t60_mom_agree"] = (np.sign(panel["t60"]) == np.sign(panel["target_mean_3"].fillna(0.0))).astype(float)
    # Expanding mean of target per factor (factor-specific long-run return)
    panel["target_expanding_mean"] = panel.groupby("factor")["target"].transform(
        lambda s: s.shift(1).expanding(min_periods=6).mean()
    )
    panel["target_expanding_mean_cs_z"] = group_zscore(panel.groupby("Date")["target_expanding_mean"])

    # Lagged t60 cross-sectional z-score (for smoothed signal)
    panel["t60_lag_1_cs_z"] = group_zscore(panel.groupby("Date")["t60_lag_1"])

    # Smoothed t60 signal: blend current with lagged to reduce selection noise
    panel["t60_smooth_z"] = (
        0.80 * panel["t60_cs_z"].fillna(0.0) + 0.20 * panel["t60_lag_1_cs_z"].fillna(0.0)
    )

    # Lag-2, lag-3, lag-4 cross-sectional z-scores for smoothing
    panel["t60_lag_2_cs_z"] = group_zscore(panel.groupby("Date")["t60_lag_2"])
    panel["t60_lag_3_cs_z"] = group_zscore(panel.groupby("Date")["t60_lag_3"])
    panel["t60_lag_4_cs_z"] = group_zscore(panel.groupby("Date")["t60_lag_4"])

    # Rank-based signal rescaled to [0,1] then z-scored within date
    panel["t60_rank_z"] = group_zscore(panel.groupby("Date")["t60_cs_rank"])

    # Rolling mean of t60 z-scores (3-month) for smoother signal
    panel["t60_mean_3_cs_z"] = group_zscore(panel.groupby("Date")["t60_mean_3"])

    # Target mean 6 cross-sectional z-score for medium-term momentum
    panel["target_mean_6_cs_z"] = group_zscore(panel.groupby("Date")["target_mean_6"])

    # Inverse-volatility weight for each factor (using 6-month realized vol)
    # Higher vol -> lower weight, to improve risk-adjusted returns
    panel["inv_vol_6"] = 1.0 / panel["target_vol_6"].clip(lower=0.01)
    panel["inv_vol_6_cs_z"] = group_zscore(panel.groupby("Date")["inv_vol_6"])

    # Inverse-volatility weight using 12-month realized vol
    panel["inv_vol_12"] = 1.0 / panel["target_vol_12"].clip(lower=0.01)
    panel["inv_vol_12_cs_z"] = group_zscore(panel.groupby("Date")["inv_vol_12"])

    # Cross-sectional z-scores for longer momentum
    panel["target_mean_24_cs_z"] = group_zscore(panel.groupby("Date")["target_mean_24"])
    panel["target_mean_36_cs_z"] = group_zscore(panel.groupby("Date")["target_mean_36"])

    # Expanding-window rank IC of t60 signal per factor (for adaptive weighting)
    # Compute rolling IC of t60 vs realized target, lagged to avoid lookahead
    panel["t60_hit"] = (np.sign(panel["t60"]) == np.sign(panel["target"])).astype(float)
    panel["t60_hit_expanding"] = panel.groupby("factor")["t60_hit"].transform(
        lambda s: s.shift(1).expanding(min_periods=6).mean()
    )
    panel["t60_hit_expanding_cs_z"] = group_zscore(panel.groupby("Date")["t60_hit_expanding"])

    # Rolling Sharpe ratio per factor (risk-adjusted momentum) - 6m and 12m
    for w in [6, 12]:
        _mean = panel.groupby("factor")["target"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(3, w // 2)).mean()
        )
        _std = panel.groupby("factor")["target"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(3, w // 2)).std()
        )
        panel[f"target_sharpe_{w}"] = _mean / _std.clip(lower=0.01)
        panel[f"target_sharpe_{w}_cs_z"] = group_zscore(panel.groupby("Date")[f"target_sharpe_{w}"])

    # Signal-momentum interaction: t60 * lagged target (signal confirms recent return)
    panel["t60_x_target_lag1"] = panel["t60_cs_z"].fillna(0.0) * panel["target_lag_1_cs_z"].fillna(0.0)

    # Target return persistence: autocorrelation proxy (lag1 * lag1 of lag1 = lag1 * lag2)
    panel["target_lag_2"] = panel.groupby("factor")["target"].shift(2)
    panel["target_persistence"] = panel["target_lag_1"].fillna(0.0) * panel["target_lag_2"].fillna(0.0)
    panel["target_persistence_cs_z"] = group_zscore(panel.groupby("Date")["target_persistence"])

    # Cross-factor mean target (market-level momentum, lagged)
    panel["market_mom_3"] = panel.groupby("Date")["target_mean_3"].transform("mean")
    panel["market_mom_12"] = panel.groupby("Date")["target_mean_12"].transform("mean")

    # Factor return relative to market (idiosyncratic momentum)
    panel["target_mean_3_rel"] = panel["target_mean_3"] - panel["market_mom_3"]
    panel["target_mean_3_rel_cs_z"] = group_zscore(panel.groupby("Date")["target_mean_3_rel"])

    # t60 change magnitude (larger changes = stronger new information)
    panel["t60_delta_abs"] = panel["t60_delta_1"].abs()
    panel["t60_delta_abs_cs_z"] = group_zscore(panel.groupby("Date")["t60_delta_abs"])

    # Longer lookback features: 48-month and 60-month momentum
    for w in [48, 60]:
        col_name = f"target_mean_{w}"
        panel[col_name] = panel.groupby("factor")["target"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=max(12, w // 3)).mean()
        )
        panel[f"{col_name}_cs_z"] = group_zscore(panel.groupby("Date")[col_name])

    # VIX level relative to its own expanding history (regime indicator)
    # Use lagged VIX (already lagged by macro_lag_months)
    panel["vix_expanding_mean"] = panel.groupby("factor")["vix"].transform(
        lambda s: s.shift(1).expanding(min_periods=6).mean()
    )
    panel["vix_expanding_std"] = panel.groupby("factor")["vix"].transform(
        lambda s: s.shift(1).expanding(min_periods=6).std()
    )
    panel["vix_regime_z"] = (panel["vix"] - panel["vix_expanding_mean"]) / panel["vix_expanding_std"].clip(lower=EPS)
    # High VIX regime: vix_regime_z > 0.5; Low VIX regime: vix_regime_z < -0.5
    panel["high_vix"] = (panel["vix_regime_z"] > 0.5).astype(float)
    panel["low_vix"] = (panel["vix_regime_z"] < -0.5).astype(float)

    panel = panel.dropna(subset=["t60", "target"]).reset_index(drop=True)
    return panel


def add_group_features(
    frame: pd.DataFrame, group_col: str, value_col: str, mean_windows: list[int], vol_windows: list[int]
) -> None:
    grouped = frame.groupby(group_col)[value_col]
    frame[f"{value_col}_lag_1"] = grouped.shift(1)
    for window in mean_windows:
        frame[f"{value_col}_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=max(2, min(window, 3))).mean()
        )
    for window in vol_windows:
        frame[f"{value_col}_vol_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=max(2, min(window, 3))).std()
        )


def group_zscore(grouped: pd.core.groupby.SeriesGroupBy) -> pd.Series:
    def _zscore(series: pd.Series) -> pd.Series:
        sigma = series.std(ddof=0)
        if pd.isna(sigma) or sigma < EPS:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / sigma

    return grouped.transform(_zscore)


def numeric_features(panel: pd.DataFrame) -> list[str]:
    excluded = {"Date", "factor", "factor_base", "factor_style", "target"}
    return [col for col in panel.columns if col not in excluded]


def categorical_features() -> list[str]:
    return ["factor", "factor_base", "factor_style"]


def make_linear_pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", make_one_hot(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def make_tree_pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(), NUMERIC_FEATURES),
            ("categorical", make_one_hot(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def parse_concentration_portfolio(mode: str) -> tuple[float | None, int | None]:
    if not mode.startswith("long_only_conc_p"):
        return None, None
    suffix = mode[len("long_only_conc_p") :]
    k_override = None
    if "_k" in suffix:
        power_part, k_part = suffix.split("_k", 1)
        suffix = power_part
        try:
            k_override = int(k_part)
        except ValueError:
            return None, None
    try:
        power = float(suffix)
    except ValueError:
        return None, None
    return power, k_override


def is_fast_portfolio(mode: str) -> bool:
    if mode in {"equal_weight", "long_only", "long_only_ew", "long_short", "long_only_conc", "long_only_rank"}:
        return True
    power, k_override = parse_concentration_portfolio(mode)
    if power is None:
        return False
    if not (4.0 <= power <= 25.0):
        return False
    return k_override is None or k_override <= 12


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    deduped: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.name, candidate.portfolio)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def select_fast_candidates(candidates: list[Candidate], max_total: int = 300) -> list[Candidate]:
    selected: list[Candidate] = []
    baseline_count_by_name: dict[str, int] = {}
    for candidate in candidates:
        if candidate.kind == "equal_weight":
            selected.append(candidate)
            continue
        if candidate.kind == "fitted":
            if candidate.name in {"ridge", "ridge_lo", "ridge_hi", "ridge_mid", "ridge_v2", "ridge_v3", "ridge_v4", "ridge_v5", "ridge_v6", "ridge_v7", "ridge_v8", "ridge_v9", "ridge_v10", "ridge_v11", "ridge_v12", "ridge_v13", "ridge_v14", "ridge_v15", "ridge_v16", "ridge_v17", "ridge_v18", "elastic_net", "elastic_net_v2", "elastic_net_v3", "elastic_net_v4", "elastic_net_v5", "elastic_net_v6", "elastic_net_v7", "elastic_net_v8", "elastic_net_v9", "elastic_net_v10", "elastic_net_v11", "elastic_net_v12", "elastic_net_v13", "elastic_net_v14", "elastic_net_v15", "elastic_net_v16", "elastic_net_v17", "elastic_net_v18", "elastic_net_v19", "elastic_net_v20", "elastic_net_v21", "elastic_net_v22", "elastic_net_v23", "elastic_net_v24", "elastic_net_v25", "elastic_net_v26", "elastic_net_v27", "elastic_net_v28", "elastic_net_v29", "elastic_net_v30", "elastic_net_v31", "elastic_net_v32", "elastic_net_v33", "elastic_net_v34", "elastic_net_v35", "elastic_net_v36", "elastic_net_v37", "elastic_net_v38", "elastic_net_v39", "elastic_net_v40", "elastic_net_v41", "elastic_net_v42", "elastic_net_v43", "elastic_net_v44", "elastic_net_v45", "elastic_net_v46", "elastic_net_v47", "elastic_net_v48", "elastic_net_v49", "elastic_net_v50", "elastic_net_v51", "elastic_net_v52", "elastic_net_v53", "elastic_net_v54", "elastic_net_v55", "elastic_net_v56", "elastic_net_v57", "elastic_net_v58", "elastic_net_v59", "elastic_net_v60", "elastic_net_v61", "elastic_net_v62", "elastic_net_v63"} and candidate.portfolio.startswith("long_only"):
                selected.append(candidate)
            continue
        if not candidate.kind.startswith("baseline"):
            continue
        if not is_fast_portfolio(candidate.portfolio):
            continue
        baseline_count = baseline_count_by_name.get(candidate.name, 0)
        if baseline_count >= 14:
            continue
        baseline_count_by_name[candidate.name] = baseline_count + 1
        selected.append(candidate)
        if len(selected) >= max_total:
            break
    return selected


def build_candidates(seed: int, profile: str, n_jobs: int) -> list[Candidate]:
    candidates = [
        Candidate("equal_weight", "equal_weight", "equal_weight"),
        Candidate("raw_t60", "baseline_raw", "long_only"),
        Candidate("raw_t60", "baseline_raw", "long_only_ew"),
        Candidate("raw_t60", "baseline_raw", "long_short"),
        Candidate("raw_t60_clipped", "baseline_raw_clipped", "long_only"),
        Candidate("raw_t60_clipped", "baseline_raw_clipped", "long_only_ew"),
        Candidate("combo_forecast", "baseline_combo", "long_only"),
        Candidate("combo_forecast", "baseline_combo", "long_only_ew"),
        Candidate("combo_forecast", "baseline_combo", "long_short"),
        Candidate("combo_v2", "baseline_combo_v2", "long_only"),
        Candidate("combo_v2", "baseline_combo_v2", "long_only_ew"),
        # Parameterized concentration powers
        Candidate("raw_t60", "baseline_raw", "long_only_conc_p5"),
        Candidate("raw_t60", "baseline_raw", "long_only_conc_p6"),
        Candidate("raw_t60", "baseline_raw", "long_only_conc_p7"),
        Candidate("raw_t60", "baseline_raw", "long_only_conc_p7.5"),
        Candidate("raw_t60", "baseline_raw", "long_only_conc_p8"),
        # Smooth v24: triple smoothing (current + lag1 + lag2) with momentum
        Candidate("smooth_v24", "baseline_smooth_v24", "long_only_conc_p8"),
        Candidate("smooth_v24", "baseline_smooth_v24", "long_only_conc_p8.5"),
        Candidate("smooth_v24", "baseline_smooth_v24", "long_only_conc_p9"),
        Candidate("smooth_v24", "baseline_smooth_v24", "long_only_conc_p9.5"),
        Candidate("smooth_v24", "baseline_smooth_v24", "long_only_conc_p10"),
        # Smooth v35: triple smoothing (65/20/15) with momentum - PROVEN
        Candidate("smooth_v35", "baseline_smooth_v35", "long_only_conc_p8.5"),
        Candidate("smooth_v35", "baseline_smooth_v35", "long_only_conc_p9"),
        Candidate("smooth_v35", "baseline_smooth_v35", "long_only_conc_p9.25"),
        Candidate("smooth_v35", "baseline_smooth_v35", "long_only_conc_p9.5"),
        Candidate("smooth_v35", "baseline_smooth_v35", "long_only_conc_p9.75"),
        Candidate("smooth_v35", "baseline_smooth_v35", "long_only_conc_p10"),
        # Smooth v39: 67/18/15 smoothing + 15% momentum - CURRENT BEST
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p8.5"),
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p9"),
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p9.25"),
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p9.5"),
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p9.75"),
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p10"),
        Candidate("smooth_v39", "baseline_smooth_v39", "long_only_conc_p10.5"),
        # Smooth v46: v35 signal + re-z-scored
        Candidate("smooth_v46", "baseline_smooth_v46", "long_only_conc_p9"),
        Candidate("smooth_v46", "baseline_smooth_v46", "long_only_conc_p9.25"),
        Candidate("smooth_v46", "baseline_smooth_v46", "long_only_conc_p9.5"),
        Candidate("smooth_v46", "baseline_smooth_v46", "long_only_conc_p9.75"),
        Candidate("smooth_v46", "baseline_smooth_v46", "long_only_conc_p10"),
        # NEW: Smooth v50: v39 signal * inverse-vol scaling (vol-adjusted)
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p7"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p7.5"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p8"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p8.5"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p9"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p9.5"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p10"),
        # NEW: Smooth v51: v35 signal * inverse-vol scaling
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p7"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p7.5"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p8"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p8.5"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p9"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p9.5"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p10"),
        # NEW: Smooth v52: v39 signal with soft vol adjustment (blend 80% signal + 20% vol-adj)
        Candidate("smooth_v52", "baseline_smooth_v52", "long_only_conc_p8"),
        Candidate("smooth_v52", "baseline_smooth_v52", "long_only_conc_p8.5"),
        Candidate("smooth_v52", "baseline_smooth_v52", "long_only_conc_p9"),
        Candidate("smooth_v52", "baseline_smooth_v52", "long_only_conc_p9.5"),
        Candidate("smooth_v52", "baseline_smooth_v52", "long_only_conc_p10"),
        Candidate("smooth_v52", "baseline_smooth_v52", "long_only_conc_p10.5"),
        # NEW: Smooth v53: v39 with clipped signal (clip at +-2.5 before portfolio)
        Candidate("smooth_v53", "baseline_smooth_v53", "long_only_conc_p8.5"),
        Candidate("smooth_v53", "baseline_smooth_v53", "long_only_conc_p9"),
        Candidate("smooth_v53", "baseline_smooth_v53", "long_only_conc_p9.5"),
        Candidate("smooth_v53", "baseline_smooth_v53", "long_only_conc_p10"),
        Candidate("smooth_v53", "baseline_smooth_v53", "long_only_conc_p10.5"),
        # NEW: Smooth v54: v39 + re-z-scored (like v46 but based on v39)
        Candidate("smooth_v54", "baseline_smooth_v54", "long_only_conc_p8.5"),
        Candidate("smooth_v54", "baseline_smooth_v54", "long_only_conc_p9"),
        Candidate("smooth_v54", "baseline_smooth_v54", "long_only_conc_p9.25"),
        Candidate("smooth_v54", "baseline_smooth_v54", "long_only_conc_p9.5"),
        Candidate("smooth_v54", "baseline_smooth_v54", "long_only_conc_p9.75"),
        Candidate("smooth_v54", "baseline_smooth_v54", "long_only_conc_p10"),
        # NEW: Smooth v55: 68/17/15 smoothing + 14% momentum (between v35 and v39)
        Candidate("smooth_v55", "baseline_smooth_v55", "long_only_conc_p9"),
        Candidate("smooth_v55", "baseline_smooth_v55", "long_only_conc_p9.25"),
        Candidate("smooth_v55", "baseline_smooth_v55", "long_only_conc_p9.5"),
        Candidate("smooth_v55", "baseline_smooth_v55", "long_only_conc_p9.75"),
        Candidate("smooth_v55", "baseline_smooth_v55", "long_only_conc_p10"),
        # NEW: Smooth v56: 66/19/15 smoothing + 15% momentum (between v35 and v39)
        Candidate("smooth_v56", "baseline_smooth_v56", "long_only_conc_p9"),
        Candidate("smooth_v56", "baseline_smooth_v56", "long_only_conc_p9.25"),
        Candidate("smooth_v56", "baseline_smooth_v56", "long_only_conc_p9.5"),
        Candidate("smooth_v56", "baseline_smooth_v56", "long_only_conc_p9.75"),
        Candidate("smooth_v56", "baseline_smooth_v56", "long_only_conc_p10"),
        # NEW: Smooth v57: v39 signal + 10% 12m momentum overlay
        Candidate("smooth_v57", "baseline_smooth_v57", "long_only_conc_p9"),
        Candidate("smooth_v57", "baseline_smooth_v57", "long_only_conc_p9.25"),
        Candidate("smooth_v57", "baseline_smooth_v57", "long_only_conc_p9.5"),
        Candidate("smooth_v57", "baseline_smooth_v57", "long_only_conc_p9.75"),
        Candidate("smooth_v57", "baseline_smooth_v57", "long_only_conc_p10"),
        # v50/v51 at higher concentration powers
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p10.5"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p11"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p11.5"),
        Candidate("smooth_v50", "baseline_smooth_v50", "long_only_conc_p12"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p10.5"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p11"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p11.5"),
        Candidate("smooth_v51", "baseline_smooth_v51", "long_only_conc_p12"),
        # v60: 4-lag smoothing + vol-adjustment (deeper smoothing)
        Candidate("smooth_v60", "baseline_smooth_v60", "long_only_conc_p9"),
        Candidate("smooth_v60", "baseline_smooth_v60", "long_only_conc_p9.5"),
        Candidate("smooth_v60", "baseline_smooth_v60", "long_only_conc_p10"),
        Candidate("smooth_v60", "baseline_smooth_v60", "long_only_conc_p10.5"),
        Candidate("smooth_v60", "baseline_smooth_v60", "long_only_conc_p11"),
        # v61: v39 signal * inverse-vol-12m scaling (more stable vol)
        Candidate("smooth_v61", "baseline_smooth_v61", "long_only_conc_p9"),
        Candidate("smooth_v61", "baseline_smooth_v61", "long_only_conc_p9.5"),
        Candidate("smooth_v61", "baseline_smooth_v61", "long_only_conc_p10"),
        Candidate("smooth_v61", "baseline_smooth_v61", "long_only_conc_p10.5"),
        Candidate("smooth_v61", "baseline_smooth_v61", "long_only_conc_p11"),
        # v62: ensemble of v39+v35 base signals, then vol-adjust
        Candidate("smooth_v62", "baseline_smooth_v62", "long_only_conc_p9"),
        Candidate("smooth_v62", "baseline_smooth_v62", "long_only_conc_p9.5"),
        Candidate("smooth_v62", "baseline_smooth_v62", "long_only_conc_p10"),
        Candidate("smooth_v62", "baseline_smooth_v62", "long_only_conc_p10.5"),
        Candidate("smooth_v62", "baseline_smooth_v62", "long_only_conc_p11"),
        # v63: v39 signal * (inv_vol + momentum quality) interaction
        Candidate("smooth_v63", "baseline_smooth_v63", "long_only_conc_p9"),
        Candidate("smooth_v63", "baseline_smooth_v63", "long_only_conc_p9.5"),
        Candidate("smooth_v63", "baseline_smooth_v63", "long_only_conc_p10"),
        Candidate("smooth_v63", "baseline_smooth_v63", "long_only_conc_p10.5"),
        Candidate("smooth_v63", "baseline_smooth_v63", "long_only_conc_p11"),
        # v64: v50 with stronger vol tilt (0.4 instead of 0.3)
        Candidate("smooth_v64", "baseline_smooth_v64", "long_only_conc_p9"),
        Candidate("smooth_v64", "baseline_smooth_v64", "long_only_conc_p9.5"),
        Candidate("smooth_v64", "baseline_smooth_v64", "long_only_conc_p10"),
        Candidate("smooth_v64", "baseline_smooth_v64", "long_only_conc_p10.5"),
        Candidate("smooth_v64", "baseline_smooth_v64", "long_only_conc_p11"),
        # v65: v50 with weaker vol tilt (0.2 instead of 0.3)
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p9"),
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p9.5"),
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p10"),
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p10.5"),
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p11"),
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p11.5"),
        Candidate("smooth_v65", "baseline_smooth_v65", "long_only_conc_p12"),
        # v66: blended 6m+12m inverse vol with v39 smoothing + 0.2 tilt
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p9"),
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p9.5"),
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p10"),
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p10.5"),
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p11"),
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p11.5"),
        Candidate("smooth_v66", "baseline_smooth_v66", "long_only_conc_p12"),
        # v67: v65 with 0.15 vol tilt (even weaker)
        Candidate("smooth_v67", "baseline_smooth_v67", "long_only_conc_p10"),
        Candidate("smooth_v67", "baseline_smooth_v67", "long_only_conc_p10.5"),
        Candidate("smooth_v67", "baseline_smooth_v67", "long_only_conc_p11"),
        Candidate("smooth_v67", "baseline_smooth_v67", "long_only_conc_p11.5"),
        Candidate("smooth_v67", "baseline_smooth_v67", "long_only_conc_p12"),
        # v68: v65 with 0.25 vol tilt (between v65 and v50)
        Candidate("smooth_v68", "baseline_smooth_v68", "long_only_conc_p10"),
        Candidate("smooth_v68", "baseline_smooth_v68", "long_only_conc_p10.5"),
        Candidate("smooth_v68", "baseline_smooth_v68", "long_only_conc_p11"),
        Candidate("smooth_v68", "baseline_smooth_v68", "long_only_conc_p11.5"),
        Candidate("smooth_v68", "baseline_smooth_v68", "long_only_conc_p12"),
        # v69: 70/15/15 smoothing + 12% momentum + 0.2 vol tilt
        Candidate("smooth_v69", "baseline_smooth_v69", "long_only_conc_p10"),
        Candidate("smooth_v69", "baseline_smooth_v69", "long_only_conc_p10.5"),
        Candidate("smooth_v69", "baseline_smooth_v69", "long_only_conc_p11"),
        Candidate("smooth_v69", "baseline_smooth_v69", "long_only_conc_p11.5"),
        Candidate("smooth_v69", "baseline_smooth_v69", "long_only_conc_p12"),
        # v70: v39 smoothing + blended 3m+6m momentum (10%+5%) + 0.2 vol tilt
        Candidate("smooth_v70", "baseline_smooth_v70", "long_only_conc_p10"),
        Candidate("smooth_v70", "baseline_smooth_v70", "long_only_conc_p10.5"),
        Candidate("smooth_v70", "baseline_smooth_v70", "long_only_conc_p11"),
        Candidate("smooth_v70", "baseline_smooth_v70", "long_only_conc_p11.5"),
        Candidate("smooth_v70", "baseline_smooth_v70", "long_only_conc_p12"),
        # v71: v39 smoothing + 13% momentum + 0.2 vol tilt (slightly less momentum)
        Candidate("smooth_v71", "baseline_smooth_v71", "long_only_conc_p10"),
        Candidate("smooth_v71", "baseline_smooth_v71", "long_only_conc_p10.5"),
        Candidate("smooth_v71", "baseline_smooth_v71", "long_only_conc_p11"),
        Candidate("smooth_v71", "baseline_smooth_v71", "long_only_conc_p11.5"),
        Candidate("smooth_v71", "baseline_smooth_v71", "long_only_conc_p12"),
        # v72: v39 smoothing + 17% momentum + 0.2 vol tilt (slightly more momentum)
        Candidate("smooth_v72", "baseline_smooth_v72", "long_only_conc_p10"),
        Candidate("smooth_v72", "baseline_smooth_v72", "long_only_conc_p10.5"),
        Candidate("smooth_v72", "baseline_smooth_v72", "long_only_conc_p11"),
        Candidate("smooth_v72", "baseline_smooth_v72", "long_only_conc_p11.5"),
        Candidate("smooth_v72", "baseline_smooth_v72", "long_only_conc_p12"),
        # v85: VIX regime-aware signal (more t60, less momentum in high VIX)
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p9"),
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p9.5"),
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p10"),
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p10.5"),
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p11"),
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p11.5"),
        Candidate("smooth_v85", "baseline_smooth_v85", "long_only_conc_p12"),
        # v86: Hit-rate weighted signal (factors with better historical t60 accuracy get more weight)
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p9"),
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p9.5"),
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p10"),
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p10.5"),
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p11"),
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p11.5"),
        Candidate("smooth_v86", "baseline_smooth_v86", "long_only_conc_p12"),
        # v87: Momentum quality filter (only use momentum when 3m and 6m agree)
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p9"),
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p9.5"),
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p10"),
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p10.5"),
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p11"),
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p11.5"),
        Candidate("smooth_v87", "baseline_smooth_v87", "long_only_conc_p12"),
        # v88: Long-term momentum blend (add 24m and 36m momentum)
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p9"),
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p9.5"),
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p10"),
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p10.5"),
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p11"),
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p11.5"),
        Candidate("smooth_v88", "baseline_smooth_v88", "long_only_conc_p12"),
        # v89: Asymmetric signal - stronger tilt for high-conviction signals
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p9"),
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p9.5"),
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p10"),
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p10.5"),
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p11"),
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p11.5"),
        Candidate("smooth_v89", "baseline_smooth_v89", "long_only_conc_p12"),
        # Fitted models
        Candidate(
            "ridge",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(Ridge(alpha=2.0, random_state=seed)),
        ),
        Candidate(
            "ridge_lo",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(Ridge(alpha=1.0, random_state=seed)),
        ),
        Candidate(
            "ridge_hi",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(Ridge(alpha=4.0, random_state=seed)),
        ),
        Candidate(
            "elastic_net",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.004, l1_ratio=0.2, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v2",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.003, l1_ratio=0.15, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v3",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.25, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v4",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.003, l1_ratio=0.10, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v5",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        # v5 at other concentration powers
        Candidate(
            "elastic_net_v5",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v5",
            "fitted",
            "long_only_conc_p10.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v5",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v5",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        # v12: alpha=0.007, l1_ratio=0.3 (slightly more regularization than v5)
        Candidate(
            "elastic_net_v12",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v12",
            "fitted",
            "long_only_conc_p10.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v12",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v12",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v12",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        # v13: alpha=0.008, l1_ratio=0.3 (more regularization)
        Candidate(
            "elastic_net_v13",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v13",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v13",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v13",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        # v14: alpha=0.006, l1_ratio=0.35 (more L1 sparsity than v5)
        Candidate(
            "elastic_net_v14",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v14",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v14",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v14",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        # v15: alpha=0.007, l1_ratio=0.35 (more regularized + more sparse)
        Candidate(
            "elastic_net_v15",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v15",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v15",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v15",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.007, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        # v16: alpha=0.006, l1_ratio=0.40 (high sparsity)
        Candidate(
            "elastic_net_v16",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v16",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v16",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.006, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ),
        # v17: alpha=0.005, l1_ratio=0.30 (less regularized than v5)
        Candidate(
            "elastic_net_v17",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v17",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v17",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v17",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ),
        # v18: alpha=0.008, l1_ratio=0.35 (heavy regularization + sparsity)
        Candidate(
            "elastic_net_v18",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v18",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v18",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.008, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ),
        # v19: alpha=0.009, l1_ratio=0.3 (even heavier regularization)
        Candidate(
            "elastic_net_v19",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.009, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v19",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.009, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v19",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.009, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        # v20: alpha=0.010, l1_ratio=0.3 (strong regularization)
        Candidate(
            "elastic_net_v20",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v20",
            "fitted",
            "long_only_conc_p10.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v20",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v20",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v20",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ),
        # v21: alpha=0.011, l1_ratio=0.3 (slightly more regularized than v20)
        *[Candidate(
            "elastic_net_v21",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.011, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v22: alpha=0.012, l1_ratio=0.3
        *[Candidate(
            "elastic_net_v22",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.012, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v23: alpha=0.013, l1_ratio=0.3
        *[Candidate(
            "elastic_net_v23",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.013, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v24: alpha=0.015, l1_ratio=0.3
        *[Candidate(
            "elastic_net_v24",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.015, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v25: alpha=0.010, l1_ratio=0.25 (less sparse than v20)
        *[Candidate(
            "elastic_net_v25",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.25, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v26: alpha=0.010, l1_ratio=0.35 (more sparse than v20)
        *[Candidate(
            "elastic_net_v26",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v27: alpha=0.012, l1_ratio=0.35 (heavier + sparser)
        *[Candidate(
            "elastic_net_v27",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.012, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v28: alpha=0.015, l1_ratio=0.35
        *[Candidate(
            "elastic_net_v28",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.015, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v29: alpha=0.020, l1_ratio=0.3 (very heavy regularization)
        *[Candidate(
            "elastic_net_v29",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.020, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v30: alpha=0.010, l1_ratio=0.50 (high sparsity)
        *[Candidate(
            "elastic_net_v30",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.010, l1_ratio=0.50, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v31: alpha=0.014, l1_ratio=0.35 (just below v28)
        *[Candidate(
            "elastic_net_v31",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.014, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "11.5", "12", "13", "14"]],
        # v32: alpha=0.016, l1_ratio=0.35 (just above v28)
        *[Candidate(
            "elastic_net_v32",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.016, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "11.5", "12", "13", "14"]],
        # v33: alpha=0.018, l1_ratio=0.35
        *[Candidate(
            "elastic_net_v33",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.018, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "11.5", "12", "13", "14"]],
        # v34: alpha=0.015, l1_ratio=0.30 (v28 alpha with less sparsity)
        *[Candidate(
            "elastic_net_v34",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.015, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "11.5", "12", "13", "14"]],
        # v35_en: alpha=0.015, l1_ratio=0.40 (v28 alpha with more sparsity)
        *[Candidate(
            "elastic_net_v35",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.015, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "11.5", "12", "13", "14"]],
        # v36: alpha=0.015, l1_ratio=0.45 (high sparsity at v28 alpha)
        *[Candidate(
            "elastic_net_v36",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.015, l1_ratio=0.45, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v37: alpha=0.020, l1_ratio=0.35 (v29-level alpha with v28 sparsity)
        *[Candidate(
            "elastic_net_v37",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.020, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "11.5", "12", "13", "14"]],
        # v38: alpha=0.025, l1_ratio=0.35 (very heavy)
        *[Candidate(
            "elastic_net_v38",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.025, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v39_en: alpha=0.020, l1_ratio=0.40
        *[Candidate(
            "elastic_net_v39",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.020, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v40: alpha=0.025, l1_ratio=0.30
        *[Candidate(
            "elastic_net_v40",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.025, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v41: alpha=0.030, l1_ratio=0.35
        *[Candidate(
            "elastic_net_v41",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.030, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v42: alpha=0.012, l1_ratio=0.40
        *[Candidate(
            "elastic_net_v42",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.012, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v43: alpha=0.018, l1_ratio=0.30
        *[Candidate(
            "elastic_net_v43",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.018, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v44: alpha=0.028, l1_ratio=0.35 (slightly less regularized than v41)
        *[Candidate(
            "elastic_net_v44",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.028, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["11", "11.5", "12", "12.5", "13", "13.5", "14"]],
        # v45: alpha=0.032, l1_ratio=0.35 (slightly more than v41)
        *[Candidate(
            "elastic_net_v45",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.032, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["11", "11.5", "12", "12.5", "13", "13.5", "14"]],
        # v46_en: alpha=0.035, l1_ratio=0.35 (more regularized)
        *[Candidate(
            "elastic_net_v46",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.035, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["11", "11.5", "12", "12.5", "13", "13.5", "14"]],
        # v47: alpha=0.030, l1_ratio=0.30 (less sparse at v41 alpha)
        *[Candidate(
            "elastic_net_v47",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.030, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ) for p in ["11", "11.5", "12", "12.5", "13", "13.5", "14"]],
        # v48: alpha=0.030, l1_ratio=0.40 (more sparse at v41 alpha)
        *[Candidate(
            "elastic_net_v48",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.030, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["11", "11.5", "12", "12.5", "13", "13.5", "14"]],
        # v49: alpha=0.030, l1_ratio=0.45 (very sparse at v41 alpha)
        *[Candidate(
            "elastic_net_v49",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.030, l1_ratio=0.45, max_iter=5000, random_state=seed)),
        ) for p in ["11", "11.5", "12", "12.5", "13", "13.5", "14"]],
        # v41 extended to half-step powers
        *[Candidate(
            "elastic_net_v41",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.030, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["11.5", "12.5", "13.5"]],
        # v50_en: alpha=0.040, l1_ratio=0.35 (even heavier regularization)
        *[Candidate(
            "elastic_net_v50",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.040, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # v51_en: alpha=0.025, l1_ratio=0.40 
        *[Candidate(
            "elastic_net_v51",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.025, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # v52_en: alpha=0.035, l1_ratio=0.30
        *[Candidate(
            "elastic_net_v52",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.035, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # v53_en: alpha=0.035, l1_ratio=0.40
        *[Candidate(
            "elastic_net_v53",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.035, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # v54_en: alpha=0.038, l1_ratio=0.35 (between v41 and v50)
        *[Candidate(
            "elastic_net_v54",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.038, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v55_en: alpha=0.042, l1_ratio=0.35 (just above v50)
        *[Candidate(
            "elastic_net_v55",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.042, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v56_en: alpha=0.045, l1_ratio=0.35
        *[Candidate(
            "elastic_net_v56",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.045, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v57_en: alpha=0.040, l1_ratio=0.30 (v50 alpha, less sparse)
        *[Candidate(
            "elastic_net_v57",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.040, l1_ratio=0.30, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v58_en: alpha=0.040, l1_ratio=0.40 (v50 alpha, more sparse)
        *[Candidate(
            "elastic_net_v58",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.040, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v59_en: alpha=0.040, l1_ratio=0.45 (v50 alpha, very sparse)
        *[Candidate(
            "elastic_net_v59",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.040, l1_ratio=0.45, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v60_en: alpha=0.050, l1_ratio=0.35 (heavier regularization)
        *[Candidate(
            "elastic_net_v60",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.050, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v61_en: alpha=0.050, l1_ratio=0.40
        *[Candidate(
            "elastic_net_v61",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.050, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v62_en: alpha=0.060, l1_ratio=0.35
        *[Candidate(
            "elastic_net_v62",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.060, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # v63_en: alpha=0.035, l1_ratio=0.35 (between v52 and v50)
        *[Candidate(
            "elastic_net_v63",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.035, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Extend v50 and v53 to p10, p10.5
        *[Candidate(
            "elastic_net_v50",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.040, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11.5"]],
        *[Candidate(
            "elastic_net_v53",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.035, l1_ratio=0.40, max_iter=5000, random_state=seed)),
        ) for p in ["10", "10.5", "11.5"]],
        # Ridge v11: alpha=40.0
        *[Candidate(
            "ridge_v11",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=40.0, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # Ridge v12: alpha=75.0
        *[Candidate(
            "ridge_v12",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=75.0, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # Ridge v13: alpha=100.0
        *[Candidate(
            "ridge_v13",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=100.0, random_state=seed)),
        ) for p in ["11", "12", "13", "14"]],
        # Ridge v14: alpha=150.0
        *[Candidate(
            "ridge_v14",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=150.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v15: alpha=200.0
        *[Candidate(
            "ridge_v15",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=200.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v16: alpha=300.0
        *[Candidate(
            "ridge_v16",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=300.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v17: alpha=500.0
        *[Candidate(
            "ridge_v17",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=500.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v18: alpha=125.0
        *[Candidate(
            "ridge_v18",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=125.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Extend v28 and v29 to higher concentration powers
        *[Candidate(
            "elastic_net_v28",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.015, l1_ratio=0.35, max_iter=5000, random_state=seed)),
        ) for p in ["13", "14", "15", "16"]],
        *[Candidate(
            "elastic_net_v29",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.020, l1_ratio=0.3, max_iter=5000, random_state=seed)),
        ) for p in ["13", "14", "15", "16"]],
        # Ridge v4: alpha=5.0 (heavier than v3)
        *[Candidate(
            "ridge_v4",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=5.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v5: alpha=8.0
        *[Candidate(
            "ridge_v5",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=8.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v6: alpha=10.0
        *[Candidate(
            "ridge_v6",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=10.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v7: alpha=15.0 (very heavy)
        *[Candidate(
            "ridge_v7",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=15.0, random_state=seed)),
        ) for p in ["10", "10.5", "11", "11.5", "12"]],
        # Ridge v8: alpha=20.0
        *[Candidate(
            "ridge_v8",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=20.0, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # Ridge v9: alpha=30.0
        *[Candidate(
            "ridge_v9",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=30.0, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # Ridge v10: alpha=50.0
        *[Candidate(
            "ridge_v10",
            "fitted",
            f"long_only_conc_p{p}",
            lambda: make_linear_pipeline(Ridge(alpha=50.0, random_state=seed)),
        ) for p in ["10", "11", "12", "13", "14"]],
        # v3 at other concentration powers
        Candidate(
            "elastic_net_v3",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.25, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v3",
            "fitted",
            "long_only_conc_p10.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.25, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v3",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.25, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v3",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.005, l1_ratio=0.25, max_iter=5000, random_state=seed)),
        ),
        # Ridge at higher concentration powers
        Candidate(
            "ridge",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(Ridge(alpha=2.0, random_state=seed)),
        ),
        Candidate(
            "ridge",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(Ridge(alpha=2.0, random_state=seed)),
        ),
        Candidate(
            "ridge_v2",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(Ridge(alpha=1.5, random_state=seed)),
        ),
        Candidate(
            "ridge_v2",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(Ridge(alpha=1.5, random_state=seed)),
        ),
        Candidate(
            "ridge_v2",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(Ridge(alpha=1.5, random_state=seed)),
        ),
        Candidate(
            "ridge_v3",
            "fitted",
            "long_only_conc_p11",
            lambda: make_linear_pipeline(Ridge(alpha=3.0, random_state=seed)),
        ),
        Candidate(
            "ridge_v3",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(Ridge(alpha=3.0, random_state=seed)),
        ),
        Candidate(
            "ridge_v3",
            "fitted",
            "long_only_conc_p12",
            lambda: make_linear_pipeline(Ridge(alpha=3.0, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v6",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.0025, l1_ratio=0.15, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v7",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.0035, l1_ratio=0.15, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v8",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.003, l1_ratio=0.12, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v9",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.003, l1_ratio=0.18, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v10",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.002, l1_ratio=0.10, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "elastic_net_v11",
            "fitted",
            "long_only_conc_p11.5",
            lambda: make_linear_pipeline(ElasticNet(alpha=0.0025, l1_ratio=0.10, max_iter=5000, random_state=seed)),
        ),
        Candidate(
            "ridge_mid",
            "fitted",
            "long_only_conc_p10",
            lambda: make_linear_pipeline(Ridge(alpha=3.0, random_state=seed)),
        ),
    ]
    if profile == "full":
        candidates.extend(
            [
                Candidate(
                    "extra_trees",
                    "fitted",
                    "long_short",
                    lambda: make_tree_pipeline(
                        ExtraTreesRegressor(
                            n_estimators=240,
                            max_depth=8,
                            min_samples_leaf=6,
                            random_state=seed,
                            n_jobs=n_jobs,
                        )
                    ),
                ),
                Candidate(
                    "random_forest",
                    "fitted",
                    "long_short",
                    lambda: make_tree_pipeline(
                        RandomForestRegressor(
                            n_estimators=180,
                            max_depth=7,
                            min_samples_leaf=10,
                            random_state=seed,
                            n_jobs=n_jobs,
                        )
                    ),
                ),
                Candidate(
                    "gradient_boosting",
                    "fitted",
                    "long_short",
                    lambda: make_tree_pipeline(
                        GradientBoostingRegressor(
                            learning_rate=0.03,
                            n_estimators=180,
                            max_depth=2,
                            random_state=seed,
                        )
                    ),
                ),
            ]
        )
    if profile == "full" and XGBRegressor is not None:
        candidates.append(
            Candidate(
                "xgboost",
                "fitted",
                "long_short",
                lambda: make_tree_pipeline(
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=220,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=n_jobs,
                    )
                ),
            )
        )
    candidates = dedupe_candidates(candidates)
    if profile == "fast":
        fast_candidates = select_fast_candidates(candidates)
        return fast_candidates if fast_candidates else candidates
    return candidates


def baseline_prediction(frame: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "baseline_raw":
        return frame["t60_cs_z"].fillna(0.0)
    if kind == "baseline_raw_clipped":
        z = frame["t60_cs_z"].fillna(0.0)
        return z.clip(-2.0, 2.0)
    if kind == "baseline_combo":
        components = [
            0.55 * frame["t60_cs_z"].fillna(0.0),
            0.20 * frame["target_mean_3_cs_z"].fillna(0.0),
            0.15 * frame["target_mean_12_cs_z"].fillna(0.0),
            0.10 * frame["target_lag_1_cs_z"].fillna(0.0),
        ]
        return sum(components)
    if kind == "baseline_combo_v2":
        t60_z = frame["t60_cs_z"].fillna(0.0).clip(-2.5, 2.5)
        mom3_z = frame["target_mean_3_cs_z"].fillna(0.0).clip(-2.5, 2.5)
        mom12_z = frame["target_mean_12_cs_z"].fillna(0.0).clip(-2.5, 2.5)
        components = [
            0.70 * t60_z,
            0.15 * mom3_z,
            0.15 * mom12_z,
        ]
        return sum(components)
    if kind == "baseline_smooth_v24":
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.60 * t60_z + 0.25 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        return 0.85 * smooth + 0.15 * mom_z
    if kind == "baseline_smooth_v35":
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.65 * t60_z + 0.20 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        return 0.85 * smooth + 0.15 * mom_z
    if kind == "baseline_smooth_v39":
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        return 0.85 * smooth + 0.15 * mom_z
    if kind == "baseline_smooth_v46":
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.65 * t60_z + 0.20 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        signal = 0.85 * smooth + 0.15 * mom_z
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v50":
        # v39 signal * inverse-vol scaling: scale signal by 1/vol to favor low-vol factors
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        # Scale by inverse vol (use inv_vol_6_cs_z as cross-sectional z-scored inverse vol)
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        # Normalize inv_vol within the cross-section
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        # Blend: signal * (1 + 0.3 * inv_vol_norm) to tilt toward low-vol factors
        signal = base_signal * (1.0 + 0.3 * inv_vol_norm.clip(-2, 2))
        # Re-z-score
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v51":
        # v35 signal * inverse-vol scaling
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.65 * t60_z + 0.20 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.3 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v52":
        # v39 signal with soft vol adjustment: 80% pure signal + 20% vol-adjusted
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        vol_adj_signal = base_signal * (1.0 + 0.3 * inv_vol_norm.clip(-2, 2))
        # Blend pure and vol-adjusted
        signal = 0.80 * base_signal + 0.20 * vol_adj_signal
        # Re-z-score
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v53":
        # v39 with signal clipping at +-2.5
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        signal = 0.85 * smooth + 0.15 * mom_z
        return signal.clip(-2.5, 2.5)
    if kind == "baseline_smooth_v54":
        # v39 signal + re-z-scored (like v46 but based on v39)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        signal = 0.85 * smooth + 0.15 * mom_z
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v55":
        # 68/17/15 smoothing + 14% momentum (between v35 and v39)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.68 * t60_z + 0.17 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        return 0.86 * smooth + 0.14 * mom_z
    if kind == "baseline_smooth_v56":
        # 66/19/15 smoothing + 15% momentum (between v35 and v39)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.66 * t60_z + 0.19 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        return 0.85 * smooth + 0.15 * mom_z
    if kind == "baseline_smooth_v57":
        # v39 signal + 10% 12m momentum overlay (blend short and long momentum)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom3_z = frame["target_mean_3_cs_z"].fillna(0.0)
        mom12_z = frame["target_mean_12_cs_z"].fillna(0.0)
        return 0.85 * smooth + 0.10 * mom3_z + 0.05 * mom12_z
    if kind == "baseline_smooth_v60":
        # 4-lag smoothing (current+lag1+lag2+lag3) + vol-adjustment
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        lag3_z = frame["t60_lag_3_cs_z"].fillna(0.0)
        smooth = 0.55 * t60_z + 0.20 * lag1_z + 0.15 * lag2_z + 0.10 * lag3_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.3 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v61":
        # v39 signal * inverse-vol-12m scaling (more stable vol estimate)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_12"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.3 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v62":
        # Ensemble of v39 and v35 base signals, then vol-adjust
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth_v39 = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        smooth_v35 = 0.65 * t60_z + 0.20 * lag1_z + 0.15 * lag2_z
        smooth = 0.50 * smooth_v39 + 0.50 * smooth_v35  # = 0.66/0.19/0.15
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.3 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v63":
        # v39 signal * (inv_vol + momentum consistency) interaction
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        # Combine inv_vol with momentum sign agreement
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        # Momentum consistency: do 3m and 12m momentum agree in sign?
        mom12_z = frame["target_mean_12_cs_z"].fillna(0.0)
        mom_agree = (np.sign(mom_z) * np.sign(mom12_z)).clip(0, 1)  # 1 if agree, 0 if not
        quality = 0.7 * inv_vol_norm.clip(-2, 2) + 0.3 * mom_agree
        signal = base_signal * (1.0 + 0.25 * quality)
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v64":
        # v50 with stronger vol tilt (0.4 instead of 0.3)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.4 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v65":
        # v50 with weaker vol tilt (0.2 instead of 0.3)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v66":
        # Blended 6m+12m inverse vol with v39 smoothing + 0.2 tilt
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        # Blend 6m and 12m inverse vol for more stable estimate
        inv_vol_6 = frame["inv_vol_6"].fillna(1.0)
        inv_vol_12 = frame["inv_vol_12"].fillna(1.0)
        inv_vol = 0.5 * inv_vol_6 + 0.5 * inv_vol_12
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v67":
        # v65 with 0.15 vol tilt (even weaker)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.15 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v68":
        # v65 with 0.25 vol tilt (between v65 and v50)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.15 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.25 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v69":
        # 70/15/15 smoothing + 12% momentum + 0.2 vol tilt
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.70 * t60_z + 0.15 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.88 * smooth + 0.12 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v70":
        # v39 smoothing + blended 3m+6m momentum (10%+5%) + 0.2 vol tilt
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom3_z = frame["target_mean_3_cs_z"].fillna(0.0)
        mom6_z = frame["target_mean_6_cs_z"].fillna(0.0)
        base_signal = 0.85 * smooth + 0.10 * mom3_z + 0.05 * mom6_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v71":
        # v39 smoothing + 13% momentum + 0.2 vol tilt (slightly less momentum)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.87 * smooth + 0.13 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v72":
        # v39 smoothing + 17% momentum + 0.2 vol tilt (slightly more momentum)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.67 * t60_z + 0.18 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.83 * smooth + 0.17 * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v85":
        # VIX regime-aware: in high VIX, rely more on t60 (90/10); in low VIX, use more momentum (80/20)
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.70 * t60_z + 0.15 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        # Use VIX regime to adjust momentum weight
        high_vix = frame["high_vix"].fillna(0.0)
        low_vix = frame["low_vix"].fillna(0.0)
        # Base: 88% smooth + 12% momentum (like v69)
        # High VIX: reduce momentum to 5%, increase smooth to 95%
        # Low VIX: increase momentum to 18%, decrease smooth to 82%
        mom_weight = 0.12 - 0.07 * high_vix + 0.06 * low_vix
        smooth_weight = 1.0 - mom_weight
        base_signal = smooth_weight * smooth + mom_weight * mom_z
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v86":
        # Hit-rate weighted: factors with better historical t60 accuracy get amplified signal
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.70 * t60_z + 0.15 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.88 * smooth + 0.12 * mom_z
        # Scale by historical hit rate (cross-sectional z-scored)
        hit_z = frame["t60_hit_expanding_cs_z"].fillna(0.0).clip(-2, 2)
        # Tilt signal toward factors where t60 has historically been more accurate
        signal = base_signal * (1.0 + 0.15 * hit_z)
        # Also apply vol tilt
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v87":
        # Momentum quality filter: use momentum only when 3m and 6m agree
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.70 * t60_z + 0.15 * lag1_z + 0.15 * lag2_z
        mom3_z = frame["target_mean_3_cs_z"].fillna(0.0)
        mom6_z = frame["target_mean_6_cs_z"].fillna(0.0)
        # Only use momentum when 3m and 6m momentum agree in sign
        mom_agree = (np.sign(mom3_z) == np.sign(mom6_z)).astype(float)
        # When they agree, use 15% momentum; when they disagree, use 5%
        effective_mom = mom3_z * (0.05 + 0.10 * mom_agree)
        base_signal = smooth * (1.0 - 0.05 - 0.10 * mom_agree.mean()) + effective_mom
        # Simpler: just blend
        base_signal = 0.85 * smooth + 0.15 * mom3_z * mom_agree
        # For factors where momentum disagrees, just use smoothed signal
        base_signal = base_signal + 0.15 * smooth * (1.0 - mom_agree) - 0.15 * smooth * mom_agree + 0.15 * smooth * mom_agree
        # Simplify: base = smooth + momentum * agreement_filter
        base_signal = (1.0 - 0.15 * mom_agree) * smooth + 0.15 * mom_agree * mom3_z + (1.0 - mom_agree) * 0.0 * smooth
        # Actually let's keep it clean:
        base_signal = smooth + 0.15 * mom_agree * mom3_z
        # Re-z-score
        sigma = base_signal.std()
        if sigma > EPS:
            base_signal = (base_signal - base_signal.mean()) / sigma
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v88":
        # Long-term momentum blend: add 24m and 36m momentum for deeper mean-reversion/trend
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.70 * t60_z + 0.15 * lag1_z + 0.15 * lag2_z
        mom3_z = frame["target_mean_3_cs_z"].fillna(0.0)
        mom12_z = frame["target_mean_12_cs_z"].fillna(0.0)
        mom24_z = frame["target_mean_24_cs_z"].fillna(0.0)
        # Blend multiple momentum horizons
        mom_blend = 0.50 * mom3_z + 0.30 * mom12_z + 0.20 * mom24_z
        base_signal = 0.88 * smooth + 0.12 * mom_blend
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = base_signal * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    if kind == "baseline_smooth_v89":
        # Asymmetric conviction: amplify high-conviction signals (abs(z) > 1), dampen low conviction
        t60_z = frame["t60_cs_z"].fillna(0.0)
        lag1_z = frame["t60_lag_1_cs_z"].fillna(0.0)
        lag2_z = frame["t60_lag_2_cs_z"].fillna(0.0)
        smooth = 0.70 * t60_z + 0.15 * lag1_z + 0.15 * lag2_z
        mom_z = frame["target_mean_3_cs_z"].fillna(0.0)
        base_signal = 0.88 * smooth + 0.12 * mom_z
        # Asymmetric transform: amplify signals with |z| > 1
        abs_signal = base_signal.abs()
        # Apply a soft power transform that amplifies high |z| values
        # sign(x) * |x|^1.2 for |x| > 0.5, sign(x) * |x|^0.8 for |x| <= 0.5
        high_conv = (abs_signal > 0.5).astype(float)
        transformed = np.sign(base_signal) * (
            high_conv * (abs_signal ** 1.15) +
            (1.0 - high_conv) * (abs_signal ** 0.85)
        )
        # Re-z-score the transformed signal
        sigma = transformed.std()
        if sigma > EPS:
            transformed = (transformed - transformed.mean()) / sigma
        inv_vol = frame["inv_vol_6"].fillna(1.0)
        iv_mean = inv_vol.mean()
        iv_std = inv_vol.std()
        if iv_std > EPS:
            inv_vol_norm = (inv_vol - iv_mean) / iv_std
        else:
            inv_vol_norm = pd.Series(np.zeros(len(inv_vol)), index=inv_vol.index)
        signal = transformed * (1.0 + 0.2 * inv_vol_norm.clip(-2, 2))
        sigma = signal.std()
        if sigma > EPS:
            signal = (signal - signal.mean()) / sigma
        return signal
    raise ValueError(f"Unknown baseline kind: {kind}")


def portfolio_weights(predictions: pd.Series, mode: str, top_k: int) -> np.ndarray:
    values = predictions.to_numpy(dtype=float)
    if mode == "equal_weight":
        if len(values) == 0:
            return np.array([], dtype=float)
        return np.full(len(values), 1.0 / len(values))
    if mode == "long_only_ew":
        if len(values) == 0:
            return np.array([], dtype=float)
        order = np.argsort(values)[::-1]
        picks = order[: min(top_k, len(order))]
        weights = np.zeros(len(values), dtype=float)
        weights[picks] = 1.0 / len(picks)
        return weights
    if mode == "long_only_conc":
        if len(values) == 0:
            return np.array([], dtype=float)
        order = np.argsort(values)[::-1]
        picks = order[: min(top_k, len(order))]
        top_values = values[picks]
        top_values = top_values - np.nanmin(top_values)
        if np.all(np.abs(top_values) < EPS):
            top_values = np.ones_like(top_values)
        top_values = top_values ** 2
        weights = np.zeros(len(values), dtype=float)
        weights[picks] = top_values
        total = weights.sum()
        return weights / total if total > EPS else np.full(len(values), 1.0 / len(values))
    if mode.startswith("long_only_conc_p"):
        suffix = mode[len("long_only_conc_p"):]
        if "_k" in suffix:
            parts = suffix.split("_k")
            power = float(parts[0])
            effective_k = int(parts[1])
        else:
            power = float(suffix)
            effective_k = top_k
        if len(values) == 0:
            return np.array([], dtype=float)
        order = np.argsort(values)[::-1]
        picks = order[: min(effective_k, len(order))]
        top_values = values[picks]
        top_values = top_values - np.nanmin(top_values)
        if np.all(np.abs(top_values) < EPS):
            top_values = np.ones_like(top_values)
        
        # Scale to max=1 to perfectly prevent float underflow before raising to power
        tv_max = np.nanmax(top_values)
        if tv_max > EPS:
            top_values = top_values / tv_max
            
        top_values = top_values ** power
        weights = np.zeros(len(values), dtype=float)
        weights[picks] = top_values
        total = weights.sum()
        return weights / total if total > EPS else np.full(len(values), 1.0 / len(values))
    if mode.startswith("long_only_softmax"):
        suffix = mode[len("long_only_softmax"):]
        if suffix.startswith("_t"):
            temp = float(suffix[2:])
        else:
            temp = 2.0
        if len(values) == 0:
            return np.array([], dtype=float)
        order = np.argsort(values)[::-1]
        picks = order[: min(top_k, len(order))]
        top_values = values[picks]
        top_values = top_values - np.nanmax(top_values)
        exp_vals = np.exp(temp * top_values)
        weights = np.zeros(len(values), dtype=float)
        weights[picks] = exp_vals
        total = weights.sum()
        return weights / total if total > EPS else np.full(len(values), 1.0 / len(values))
    if mode == "long_only":
        if len(values) == 0:
            return np.array([], dtype=float)
        order = np.argsort(values)[::-1]
        picks = order[: min(top_k, len(order))]
        top_values = values[picks]
        top_values = top_values - np.nanmin(top_values)
        if np.all(np.abs(top_values) < EPS):
            top_values = np.ones_like(top_values)
        weights = np.zeros(len(values), dtype=float)
        weights[picks] = top_values
        total = weights.sum()
        return weights / total if total > EPS else np.full(len(values), 1.0 / len(values))
    if mode == "long_only_rank":
        if len(values) == 0:
            return np.array([], dtype=float)
        order = np.argsort(values)[::-1]
        k = min(top_k, len(order))
        picks = order[:k]
        weights = np.zeros(len(values), dtype=float)
        rank_weights = np.arange(k, 0, -1, dtype=float)
        weights[picks] = rank_weights
        total = weights.sum()
        return weights / total if total > EPS else np.full(len(values), 1.0 / len(values))
    if mode == "long_short":
        centered = values - np.nanmean(values)
        scale = np.nansum(np.abs(centered))
        if scale < EPS:
            return np.zeros(len(values), dtype=float)
        return centered / scale
    raise ValueError(f"Unknown portfolio mode: {mode}")


def fit_predict_candidate(
    candidate: Candidate, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> np.ndarray:
    model = candidate.builder()
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def run_backtest(
    panel: pd.DataFrame,
    candidates: list[Candidate],
    train_window: int,
    min_train_months: int,
    top_k: int,
    n_jobs: int,
) -> dict:
    dates = sorted(panel["Date"].unique())
    predictions_by_candidate: dict[str, list[pd.DataFrame]] = {candidate_key(c): [] for c in candidates}
    skipped_dates = 0

    for date_index, test_date in enumerate(dates):
        train_end = date_index
        train_start = 0 if train_window <= 0 else max(0, train_end - train_window)
        if train_end < min_train_months:
            skipped_dates += 1
            continue

        train_dates = dates[train_start:train_end]
        if train_dates and max(train_dates) >= test_date:
            raise RuntimeError(f"Lookahead leakage detected: train end {max(train_dates)} overlaps test date {test_date}.")
        train_frame = panel[panel["Date"].isin(train_dates)]
        test_frame = panel[panel["Date"] == test_date].copy()
        if train_frame.empty or test_frame.empty:
            skipped_dates += 1
            continue

        X_train = train_frame[FEATURE_COLUMNS + CATEGORICAL_FEATURES]
        y_train = train_frame["target"]
        X_test = test_frame[FEATURE_COLUMNS + CATEGORICAL_FEATURES]

        baseline_cache: dict[str, np.ndarray] = {}
        for candidate in candidates:
            if candidate.kind.startswith("baseline") and candidate.kind not in baseline_cache:
                baseline_cache[candidate.kind] = baseline_prediction(test_frame, candidate.kind).to_numpy(dtype=float)

        fitted_candidates = [candidate for candidate in candidates if candidate.kind == "fitted"]
        fitted_predictions: dict[str, np.ndarray] = {}
        if fitted_candidates:
            if n_jobs > 1 and len(fitted_candidates) > 1:
                max_workers = min(n_jobs, len(fitted_candidates))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(fit_predict_candidate, candidate, X_train, y_train, X_test): candidate_key(
                            candidate
                        )
                        for candidate in fitted_candidates
                    }
                    for future in as_completed(futures):
                        fitted_predictions[futures[future]] = future.result()
            else:
                for candidate in fitted_candidates:
                    fitted_predictions[candidate_key(candidate)] = fit_predict_candidate(candidate, X_train, y_train, X_test)

        for candidate in candidates:
            key = candidate_key(candidate)
            if candidate.kind == "equal_weight":
                preds = np.zeros(len(test_frame), dtype=float)
            elif candidate.kind.startswith("baseline"):
                preds = baseline_cache[candidate.kind]
            else:
                preds = fitted_predictions[key]

            test_output = test_frame[["Date", "factor", "target"]].copy()
            test_output["prediction"] = preds
            test_output["weight"] = portfolio_weights(test_output["prediction"], candidate.portfolio, top_k)
            test_output["portfolio_return"] = test_output["weight"] * test_output["target"]
            predictions_by_candidate[key].append(test_output)

    summaries = []
    monthly_returns: dict[str, pd.Series] = {}
    full_predictions: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        combined = pd.concat(predictions_by_candidate[key], ignore_index=True)
        full_predictions[key] = combined
        monthly = combined.groupby("Date")["portfolio_return"].sum().sort_index()
        monthly_returns[key] = monthly
        summaries.append(score_candidate(candidate, combined, monthly))

    leaderboard = pd.DataFrame(summaries)
    # Enforce return > 5%
    leaderboard["primary_score"] = np.where(
        leaderboard["annualized_return_pct"] >= 5.0,
        leaderboard["sharpe_ann"],
        leaderboard["sharpe_ann"] - 100.0,
    )
    leaderboard = leaderboard.sort_values("primary_score", ascending=False).reset_index(drop=True)
    best_key = leaderboard.loc[0, "candidate_key"]
    return {
        "leaderboard": leaderboard,
        "monthly_returns": monthly_returns,
        "predictions": full_predictions,
        "best_candidate_key": best_key,
        "skipped_dates": skipped_dates,
    }


def score_candidate(candidate: Candidate, combined: pd.DataFrame, monthly: pd.Series) -> dict:
    gross_returns = monthly.astype(float)
    mean_monthly = gross_returns.mean()
    vol_monthly = gross_returns.std(ddof=1)
    sharpe_ann = np.sqrt(12.0) * mean_monthly / vol_monthly if vol_monthly and abs(vol_monthly) > EPS else np.nan
    annualized_return = (
        ((1.0 + gross_returns.div(100.0)).prod() ** (12.0 / len(gross_returns)) - 1.0) * 100.0
        if len(gross_returns) > 0
        else np.nan
    )
    annualized_vol = vol_monthly * np.sqrt(12.0) if pd.notna(vol_monthly) else np.nan
    equity_curve = (1.0 + gross_returns / 100.0).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    turnover = (
        combined.pivot(index="Date", columns="factor", values="weight")
        .fillna(0.0)
        .sort_index()
        .diff()
        .abs()
        .sum(axis=1)
        .dropna()
        .mean()
    )
    rank_ic = combined.groupby("Date", sort=True)[["prediction", "target"]].apply(
        lambda frame: (
            np.nan
            if frame["prediction"].nunique() < 2 or frame["target"].nunique() < 2
            else frame["prediction"].corr(frame["target"], method="spearman")
        )
    )
    return {
        "candidate_key": candidate_key(candidate),
        "model_name": candidate.name,
        "portfolio": candidate.portfolio,
        "months": int(gross_returns.shape[0]),
        "mean_monthly_return_pct": float(mean_monthly),
        "vol_monthly_pct": float(vol_monthly),
        "annualized_return_pct": float(annualized_return),
        "annualized_vol_pct": float(annualized_vol),
        "sharpe_ann": float(sharpe_ann),
        "hit_rate": float((gross_returns > 0).mean()),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "avg_turnover": float(turnover),
        "mean_rank_ic": float(rank_ic.mean()),
        "median_rank_ic": float(rank_ic.median()),
    }


def candidate_key(candidate: Candidate) -> str:
    return f"{candidate.name}__{candidate.portfolio}"


def save_outputs(results: dict, config: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    leaderboard_path = OUTPUT_DIR / "leaderboard.csv"
    returns_path = OUTPUT_DIR / "monthly_returns.csv"
    summary_path = OUTPUT_DIR / "model_summary.json"
    predictions_path = OUTPUT_DIR / "best_candidate_predictions.csv"

    leaderboard = results["leaderboard"].copy()
    leaderboard.to_csv(leaderboard_path, index=False)

    returns_table = pd.DataFrame(results["monthly_returns"]).sort_index()
    returns_table.to_csv(returns_path, index=True)

    best_predictions = results["predictions"][results["best_candidate_key"]].copy()
    best_predictions.to_csv(predictions_path, index=False)
    
    # Also save as excel weights matrix automatically!
    try:
        weights_pivot = best_predictions.pivot(index="Date", columns="factor", values="weight")
        t2_path = OUTPUT_DIR.parent / "T2_Optimizer.xlsx"
        t2 = pd.read_excel(str(t2_path))
        factor_cols = [c for c in t2.columns if c != "Date"]
        weights_formatted = weights_pivot.reindex(columns=factor_cols).fillna(0.0).reset_index()
        weights_formatted.to_excel(str(OUTPUT_DIR / "Best_Model_Factor_Weights.xlsx"), index=False)
        print("Automatically exported Best_Model_Factor_Weights.xlsx!")
    except Exception as e:
        print(f"Could not export excel weights: {e}")

    payload = {
        "primary_score": float(leaderboard.loc[0, "primary_score"]),
        "best_candidate_key": results["best_candidate_key"],
        "config": config,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "artifacts": {
            "leaderboard_csv": str(leaderboard_path),
            "monthly_returns_csv": str(returns_path),
            "best_predictions_csv": str(predictions_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2))


def render_console(results: dict) -> None:
    leaderboard = results["leaderboard"]
    best = leaderboard.iloc[0]
    print(f"Primary score (annualized Sharpe with >=5% return constraint): {best['primary_score']:.4f}")
    print(f"Best candidate: {best['candidate_key']}")
    print(f"Months tested: {int(best['months'])}")
    print()
    print(
        leaderboard[
            [
                "candidate_key",
                "annualized_return_pct",
                "annualized_vol_pct",
                "sharpe_ann",
                "hit_rate",
                "mean_rank_ic",
                "avg_turnover",
                "max_drawdown_pct",
            ]
        ].round(4).to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    args.n_jobs = max(1, int(args.n_jobs))
    np.random.seed(args.seed)

    panel = load_inputs(args.target_shift_months, args.macro_lag_months)
    if panel.empty:
        raise RuntimeError("No rows remain after aligning the workbooks.")

    config = {
        "target_shift_months": args.target_shift_months,
        "macro_lag_months": args.macro_lag_months,
        "train_window": args.train_window,
        "min_train_months": args.min_train_months,
        "top_k": args.top_k,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "model_profile": args.model_profile,
        "primary_portfolio": PRIMARY_PORTFOLIO,
        "panel_rows": int(panel.shape[0]),
        "date_count": int(panel["Date"].nunique()),
        "factor_count": int(panel["factor"].nunique()),
        "date_start": panel["Date"].min().strftime("%Y-%m-%d"),
        "date_end": panel["Date"].max().strftime("%Y-%m-%d"),
    }

    global FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    FEATURE_COLUMNS = numeric_features(panel)
    NUMERIC_FEATURES = FEATURE_COLUMNS
    CATEGORICAL_FEATURES = categorical_features()

    candidates = build_candidates(args.seed, args.model_profile, args.n_jobs)
    results = run_backtest(panel, candidates, args.train_window, args.min_train_months, args.top_k, args.n_jobs)
    save_outputs(results, config)

    payload = {
        "primary_score": float(results["leaderboard"].loc[0, "primary_score"]),
        "best_candidate_key": results["best_candidate_key"],
        "leaderboard": results["leaderboard"].to_dict(orient="records"),
        "artifacts": {
            "leaderboard_csv": str(OUTPUT_DIR / "leaderboard.csv"),
            "monthly_returns_csv": str(OUTPUT_DIR / "monthly_returns.csv"),
            "best_predictions_csv": str(OUTPUT_DIR / "best_candidate_predictions.csv"),
            "summary_json": str(OUTPUT_DIR / "model_summary.json"),
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        render_console(results)


if __name__ == "__main__":
    main()
