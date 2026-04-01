#!/usr/bin/env python3
"""
Step Five idiot: Winner-only wrapper around train.py

PURPOSE:
Run the restored Step Five New winner configuration with a minimal wrapper script
that delegates all core functionality to train.py, ensuring exact behavioral consistency
with the original training pipeline.

FUNCTIONALITY:
- Load and engineer features via train.load_inputs()
- Configure the winning ElasticNet candidate (elastic_net_v62) only
- Run train.run_backtest() with identical parameters to Step Five New
- Export weights, statistics, visualizations, and model coefficients
- Maintain identical output format and file names

DEPENDENCIES:
- pandas: Data manipulation and Excel I/O
- numpy: Numerical computations and random seed
- sklearn: ElasticNet model
- matplotlib: Visualization generation
- train: Core feature engineering, backtesting, and portfolio construction
- pathlib: File path handling
- logging: Console output formatting

INPUTS:
- T60.xlsx: Factor signals dataset (via train.load_inputs())
- T2_Optimizer.xlsx: Target returns and factor universe template
- Extrernal Data.xlsx: Macro variables and market indicators (via train.load_inputs())

OUTPUTS:
- T2_rolling_window_weights.xlsx: Time-series factor weights (date x factor matrix)
- T2_strategy_statistics.xlsx: Performance metrics, monthly returns, and turnover statistics
- T2_factor_weight_heatmap.pdf: Heatmap visualization of top 20 factors over time
- T2_strategy_performance.pdf: Cumulative performance chart
- T2_latest_model_coefficients.xlsx: Model coefficients and metadata from latest training period
- Console output: Backtest results summary with performance metrics

AUTHOR: Wrapper for train.py winner configuration
DATE: 2026-03-11
VERSION: 1.0

NOTES:
- This script is intentionally thin: it keeps the exact train.py dependency so
  the outputs match the restored Step Five New behavior.
- All model fitting, feature engineering, and portfolio construction logic live
  in train.py.
- Produces identical results to Step Five New.py
- Random seed: 7 for reproducibility
"""
from __future__ import annotations

from pathlib import Path
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import train
from sklearn.linear_model import ElasticNet

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

ROOT = Path(__file__).resolve().parent
T2_PATH = ROOT / "T2_Optimizer.xlsx"


def configure_winner(seed: int) -> train.Candidate:
    """
    PURPOSE:
    Configure the winning ElasticNet model candidate.
    
    FUNCTIONALITY:
    Creates a Candidate object with the optimal hyperparameters discovered
    during the original training pipeline search.
    
    INPUTS:
    seed: Random seed for reproducibility
    
    OUTPUTS:
    train.Candidate object configured with:
    - name: "elastic_net_v62"
    - kind: "fitted" (ML model)
    - portfolio: "long_only_conc_p1" (concentrated long-only with power=1)
    - builder: Lambda returning ElasticNet pipeline with alpha=0.060, l1_ratio=0.35
    
    DEPENDENCIES:
    train: Candidate class and make_linear_pipeline function
    sklearn.linear_model: ElasticNet model
    
    NOTES:
    - Alpha=0.060: L1/L2 regularization strength
    - L1_ratio=0.35: 35% L1 (Lasso), 65% L2 (Ridge)
    - max_iter=5000: Maximum iterations for convergence
    - Portfolio power=1: Minimal concentration within the selected names
    """
    return train.Candidate(
        name="elastic_net_v62",
        kind="fitted",
        portfolio="long_only_conc_p1",
        builder=lambda: train.make_linear_pipeline(
            ElasticNet(alpha=0.060, l1_ratio=0.35, max_iter=5000, random_state=seed)
        ),
    )


def force_lagged_t60_hit(panel: pd.DataFrame) -> pd.DataFrame:
    """
    PURPOSE:
    Remove same-row target leakage from the direct t60 hit feature.

    FUNCTIONALITY:
    1. Shift t60_hit by one month within each factor
    2. Recompute the expanding hit-rate feature from the lagged series
    3. Recompute the cross-sectional z-score of the expanding hit-rate feature

    INPUTS:
    panel: Feature-engineered panel returned by train.load_inputs()

    OUTPUTS:
    DataFrame with t60_hit forced to use only prior-month information

    DEPENDENCIES:
    pandas: Grouped shifts and expanding calculations
    train: group_zscore helper
    """
    panel = panel.sort_values(["factor", "Date"]).copy()
    panel["t60_hit"] = panel.groupby("factor")["t60_hit"].shift(1)
    panel["t60_hit_expanding"] = panel.groupby("factor")["t60_hit"].transform(
        lambda series: series.expanding(min_periods=6).mean()
    )
    panel["t60_hit_expanding_cs_z"] = train.group_zscore(panel.groupby("Date")["t60_hit_expanding"])
    return panel


def build_weights_table(results: dict, winner: train.Candidate) -> pd.DataFrame:
    """
    PURPOSE:
    Transform backtest weights into T2_Optimizer-compatible format.
    
    FUNCTIONALITY:
    1. Extract predictions for winning candidate
    2. Pivot from long format (Date, factor, weight) to wide format (Date x factors)
    3. Align columns to match T2_Optimizer.xlsx template
    4. Fill missing values with 0.0
    5. Sort by date chronologically
    
    INPUTS:
    results: Results dictionary from train.run_backtest()
    winner: Candidate object that generated the weights
    
    OUTPUTS:
    DataFrame with:
    - Index: Date (datetime)
    - Columns: Factor names matching T2_Optimizer.xlsx
    - Values: Portfolio weights (sum to 1.0 per row)
    
    DEPENDENCIES:
    pandas: Pivot operations and Excel I/O
    train: candidate_key function
    
    NOTES:
    - Ensures exact column alignment with T2_Optimizer template
    - Missing weights filled with 0.0 (factor not selected that month)
    - Output ready for downstream consumption by Step Six
    """
    best_preds = results["predictions"][train.candidate_key(winner)].copy()
    pivoted = best_preds.pivot(index="Date", columns="factor", values="weight").fillna(0.0)

    t2_template = pd.read_excel(T2_PATH)
    output_cols = [column for column in t2_template.columns if column != "Date"]
    weights_df = pivoted.reindex(columns=output_cols, fill_value=0.0)
    weights_df.index = pd.to_datetime(weights_df.index)
    return weights_df.sort_index()


def build_scores_table(results: dict, winner: train.Candidate) -> pd.DataFrame:
    """
    PURPOSE:
    Transform raw model scores into a date x factor matrix.

    FUNCTIONALITY:
    1. Extract raw predictions for the winning candidate
    2. Pivot from long format (Date, factor, prediction) to wide format (Date x factors)
    3. Align columns to match T2_Optimizer.xlsx template
    4. Fill missing values with 0.0
    5. Sort by date chronologically

    INPUTS:
    results: Results dictionary from train.run_backtest()
    winner: Candidate object that generated the scores

    OUTPUTS:
    DataFrame with:
    - Index: Date (datetime)
    - Columns: Factor names matching T2_Optimizer.xlsx
    - Values: Raw model prediction scores prior to portfolio weighting

    DEPENDENCIES:
    pandas: Pivot operations and Excel I/O
    train: candidate_key function

    NOTES:
    - These scores are the factor alphas used to rank factors before weight construction
    - Output uses the same factor ordering as the weights table
    """
    best_preds = results["predictions"][train.candidate_key(winner)].copy()
    pivoted = best_preds.pivot(index="Date", columns="factor", values="prediction").fillna(0.0)

    t2_template = pd.read_excel(T2_PATH)
    output_cols = [column for column in t2_template.columns if column != "Date"]
    scores_df = pivoted.reindex(columns=output_cols, fill_value=0.0)
    scores_df.index = pd.to_datetime(scores_df.index)
    return scores_df.sort_index()


def create_factor_weight_heatmap(weights_df: pd.DataFrame, top_n: int = 20) -> Path:
    """
    PURPOSE:
    Generate a visual heatmap of factor weights over time.
    
    FUNCTIONALITY:
    Creates a heatmap showing the top N factors by average weight:
    1. Calculate average weight per factor across all dates
    2. Select top N factors by average weight
    3. Transpose to show factors on Y-axis, dates on X-axis
    4. Render as color-coded heatmap
    5. Save as PDF file
    
    INPUTS:
    weights_df: DataFrame with dates as index, factors as columns
    top_n: Number of top factors to display (default: 20)
    
    OUTPUTS:
    Path object pointing to saved PDF file (T2_factor_weight_heatmap.pdf)
    
    DEPENDENCIES:
    matplotlib.pyplot: Plotting and visualization
    numpy: Array operations for tick positioning
    pathlib: Path object creation
    
    NOTES:
    - Shows only top factors to prevent overcrowding
    - Color intensity indicates weight magnitude
    - X-axis shows up to 12 evenly-spaced date labels
    - 300 DPI for publication quality
    - Figure size: 16x9 inches
    """
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_factors = avg_weights.head(top_n).index.tolist()
    plot_df = weights_df[top_factors].T

    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(plot_df.values, aspect="auto")
    ax.set_yticks(range(len(top_factors)))
    ax.set_yticklabels(top_factors, fontsize=8)
    tick_idx = np.linspace(0, len(plot_df.columns) - 1, min(12, len(plot_df.columns)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([plot_df.columns[i].strftime("%Y-%m") for i in tick_idx], rotation=45, ha="right")
    ax.set_title("T2 Factor Weight Heatmap")
    fig.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()

    output_path = ROOT / "T2_factor_weight_heatmap.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_strategy_performance_plot(monthly_returns: pd.Series) -> Path:
    """
    PURPOSE:
    Generate a cumulative performance chart for the strategy.
    
    FUNCTIONALITY:
    Creates a line plot showing cumulative returns over time:
    1. Convert monthly percentage returns to growth factors
    2. Calculate cumulative product (equity curve)
    3. Plot equity curve with baseline at 1.0
    4. Save as PDF file
    
    INPUTS:
    monthly_returns: Series of monthly returns in percentage points
    
    OUTPUTS:
    Path object pointing to saved PDF file (T2_strategy_performance.pdf)
    
    DEPENDENCIES:
    matplotlib.pyplot: Plotting and visualization
    pathlib: Path object creation
    
    NOTES:
    - Y-axis starts at 1.0 (initial capital)
    - Dashed horizontal line at 1.0 shows breakeven
    - Returns must be in percentage points (not decimals)
    - 300 DPI for publication quality
    - Figure size: 12x7 inches
    """
    cumulative = (1.0 + monthly_returns / 100.0).cumprod()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(cumulative.index, cumulative.values, linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_title("T2 Strategy Cumulative Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path = ROOT / "T2_strategy_performance.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_fast_outputs(results: dict, winner: train.Candidate) -> dict[str, Path]:
    """
    PURPOSE:
    Export all backtest results to Excel and PDF files.
    
    FUNCTIONALITY:
    Generates comprehensive output package:
    1. Weights workbook: Time-series factor weights plus raw model scores (Excel)
    2. Statistics workbook: Performance metrics, returns, turnover (Excel)
    3. Weight heatmap: Visual factor allocation over time (PDF)
    4. Performance chart: Cumulative returns plot (PDF)
    
    INPUTS:
    results: Results dictionary from train.run_backtest()
    winner: Candidate object that generated the results
    
    OUTPUTS:
    Dictionary mapping file types to Path objects:
    - weights_xlsx: T2_rolling_window_weights.xlsx
    - stats_xlsx: T2_strategy_statistics.xlsx
    - heatmap_pdf: T2_factor_weight_heatmap.pdf
    - performance_pdf: T2_strategy_performance.pdf
    
    DEPENDENCIES:
    pandas: Excel writing with xlsxwriter engine
    matplotlib: Chart generation (via helper functions)
    pathlib: File path handling
    train: candidate_key function
    
    NOTES:
    - Excel files use xlsxwriter engine for formatting control
    - Date columns formatted as 'dd-mmm-yyyy'
    - Statistics workbook has three sheets: Summary Statistics, Monthly Returns, Monthly Turnover
    - Weights workbook includes the original weights sheet plus a Scores sheet
    - All files saved to ROOT directory
    - Turnover calculated as half of sum of absolute weight changes
    - Includes skewness and kurtosis in summary statistics
    """
    key = train.candidate_key(winner)
    weights_df = build_weights_table(results, winner)
    scores_df = build_scores_table(results, winner)
    monthly_returns = results["monthly_returns"][key].copy().sort_index()
    leaderboard_row = results["leaderboard"].iloc[0]

    weight_changes = weights_df.diff().abs()
    monthly_turnover = weight_changes.sum(axis=1) / 2.0

    stats = {
        "Annualized Return (%)": float(leaderboard_row["annualized_return_pct"]),
        "Annualized Volatility (%)": float(leaderboard_row["annualized_vol_pct"]),
        "Sharpe Ratio": float(leaderboard_row["sharpe_ann"]),
        "Maximum Drawdown (%)": float(leaderboard_row["max_drawdown_pct"]),
        "Average Monthly Turnover (%)": float(leaderboard_row["avg_turnover"] * 100.0),
        "Positive Months (%)": float(leaderboard_row["hit_rate"] * 100.0),
        "Skewness": float(monthly_returns.skew()),
        "Kurtosis": float(monthly_returns.kurtosis()),
    }

    weights_path = ROOT / "T2_rolling_window_weights.xlsx"
    with pd.ExcelWriter(weights_path, engine="xlsxwriter") as writer:
        weights_df.to_excel(writer)
        scores_df.to_excel(writer, sheet_name="Scores")

        workbook = writer.book
        date_format = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        for sheet_name in ["Sheet1", "Scores"]:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 12, date_format)

    stats_path = ROOT / "T2_strategy_statistics.xlsx"
    with pd.ExcelWriter(stats_path, engine="xlsxwriter") as writer:
        stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
        stats_df.to_excel(writer, sheet_name="Summary Statistics", index=False)

        returns_df = pd.DataFrame({"Hybrid Strategy Returns": monthly_returns})
        returns_df.to_excel(writer, sheet_name="Monthly Returns")

        turnover_df = pd.DataFrame({"Monthly Turnover": monthly_turnover})
        turnover_df.to_excel(writer, sheet_name="Monthly Turnover")

        workbook = writer.book
        date_format = workbook.add_format({"num_format": "dd-mmm-yyyy"})
        for sheet_name in ["Monthly Returns", "Monthly Turnover"]:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 12, date_format)

    heatmap_path = create_factor_weight_heatmap(weights_df)
    perf_plot_path = create_strategy_performance_plot(monthly_returns)

    return {
        "weights_xlsx": weights_path,
        "stats_xlsx": stats_path,
        "heatmap_pdf": heatmap_path,
        "performance_pdf": perf_plot_path,
    }


def extract_latest_model_coefficients(
    panel: pd.DataFrame,
    winner: train.Candidate,
    train_window: int = 0,
    min_train_months: int = 60,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    PURPOSE:
    Extract and rank model coefficients from the most recent training period.
    
    FUNCTIONALITY:
    Trains the model on the latest available data and extracts:
    1. Feature names from preprocessor (after one-hot encoding)
    2. Model coefficients (weights) for each feature
    3. Absolute values for ranking by importance
    4. Model intercept term
    
    INPUTS:
    panel: Full panel DataFrame with all dates and features
    winner: Candidate object to extract coefficients from
    train_window: Number of months for training (0 = expanding)
    min_train_months: Minimum months required before extraction (default: 60)
    
    OUTPUTS:
    Tuple containing:
    - DataFrame with columns [feature, coefficient, abs_coefficient]
    - Timestamp of the latest test date used
    
    DEPENDENCIES:
    pandas: DataFrame operations and timestamp handling
    numpy: Array operations and type conversion
    train: FEATURE_COLUMNS and CATEGORICAL_FEATURES globals
    sklearn: Pipeline access and coefficient extraction
    
    NOTES:
    - Coefficients sorted by absolute value (descending)
    - Intercept included as first row
    - Feature names include one-hot encoded categorical variables
    - Uses same training logic as backtest for consistency
    - Raises RuntimeError if no eligible dates found
    """
    dates = sorted(panel["Date"].unique())
    eligible_indices = [idx for idx in range(len(dates)) if idx >= min_train_months]
    if not eligible_indices:
        raise RuntimeError("No eligible test date found for coefficient extraction.")

    latest_index = eligible_indices[-1]
    latest_date = pd.to_datetime(dates[latest_index])
    train_start = 0 if train_window <= 0 else max(0, latest_index - train_window)
    train_dates = dates[train_start:latest_index]
    train_frame = panel[panel["Date"].isin(train_dates)]
    if train_frame.empty:
        raise RuntimeError("Latest training frame is empty; cannot extract coefficients.")

    X_train = train_frame[train.FEATURE_COLUMNS + train.CATEGORICAL_FEATURES]
    y_train = train_frame["target"]

    model = winner.builder()
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(regressor.coef_, dtype=float)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False, ignore_index=True)

    intercept_row = pd.DataFrame(
        [
            {
                "feature": "intercept",
                "coefficient": float(regressor.intercept_),
                "abs_coefficient": abs(float(regressor.intercept_)),
            }
        ]
    )
    return pd.concat([intercept_row, coef_df], ignore_index=True), latest_date


def main() -> None:
    """
    PURPOSE:
    Main execution function for Step Five idiot.
    
    FUNCTIONALITY:
    Orchestrates the complete workflow:
    1. Sets random seed for reproducibility
    2. Loads and engineers features from input files (via train.py)
    3. Configures the winning ElasticNet model
    4. Runs expanding window backtest (via train.py)
    5. Displays performance metrics
    6. Exports weights, statistics, visualizations, and coefficients
    
    INPUTS:
    None: Uses hardcoded file paths and parameters
    
    OUTPUTS:
    - Five output files (weights, statistics, heatmap, performance, coefficients)
    - Console output: Performance metrics and backtest statistics
    
    DEPENDENCIES:
    All imported modules (pandas, numpy, sklearn, matplotlib, train, etc.)
    
    NOTES:
    The winning strategy uses:
    - Model: ElasticNet with alpha=0.060, l1_ratio=0.35
    - Portfolio: long_only_conc_p1 (concentrated long-only with power=1)
    - Training: Expanding window with 60-month minimum
    - Universe: Top 12 factors by prediction
    - Random seed: 7 for reproducibility
    - Identical to Step Five New.py output
    """
    seed = 7
    np.random.seed(seed)
    train.PRIMARY_PORTFOLIO = "long_only_conc_p1"

    logging.info("=" * 80)
    logging.info("T2 FACTOR TIMING - STEP FIVE NEW: ML-BASED FACTOR TIMING")
    logging.info("=" * 80)
    logging.info("Model: ElasticNet (alpha=0.060, l1_ratio=0.35)")
    logging.info("Portfolio: long_only_conc_p1 (concentrated long-only with power=1)")
    logging.info("Training: Expanding window with 60-month minimum")
    logging.info("=" * 80)
    logging.info("Loading datasets and engineering features...")
    panel = train.load_inputs(target_shift_months=0, macro_lag_months=1)
    panel = force_lagged_t60_hit(panel)
    logging.info("Applied one-month lag to t60_hit to remove same-row target leakage.")

    train.FEATURE_COLUMNS = train.numeric_features(panel)
    train.NUMERIC_FEATURES = train.FEATURE_COLUMNS
    train.CATEGORICAL_FEATURES = train.categorical_features()

    winner = configure_winner(seed)

    logging.info("Running Expanding Window Walk-Forward Backtest (this may take a minute) ...")
    results = train.run_backtest(
        panel=panel,
        candidates=[winner],
        train_window=0,
        min_train_months=60,
        top_k=12,
        n_jobs=-1,
    )

    leaderboard = results["leaderboard"]
    row = leaderboard.iloc[0]

    logging.info("=" * 80)
    logging.info("BACKTEST RESULTS SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Annualized Return (%)         : {row['annualized_return_pct']:8.2f}")
    logging.info(f"Annualized Volatility (%)     : {row['annualized_vol_pct']:8.2f}")
    logging.info(f"Sharpe Ratio                  : {row['sharpe_ann']:8.2f}")
    logging.info(f"Maximum Drawdown (%)          : {row['max_drawdown_pct']:8.2f}")
    logging.info(f"Average Monthly Turnover (%)  : {row['avg_turnover'] * 100:8.2f}")
    logging.info(f"Hit Rate (%)                  : {row['hit_rate'] * 100:8.2f}")
    logging.info(f"Mean Rank IC                  : {row['mean_rank_ic']:8.2f}")
    logging.info(f"Median Rank IC                : {row['median_rank_ic']:8.2f}")

    logging.info("Generating outputs (weights, statistics, visualizations)...")
    output_files = save_fast_outputs(results, winner)

    logging.info("Saving results to Excel files...")
    for file_type, file_path in output_files.items():
        if file_type == "weights_xlsx":
            logging.info(f"Weights saved to {file_path.name}")
        elif file_type == "stats_xlsx":
            logging.info(f"Strategy statistics saved to {file_path.name}")
        elif file_type == "heatmap_pdf":
            logging.info(f"Factor weight heatmap saved to {file_path.name}")
        elif file_type == "performance_pdf":
            logging.info(f"Strategy performance plot saved to {file_path.name}")

    coefficients_df, latest_date = extract_latest_model_coefficients(
        panel=panel,
        winner=winner,
        train_window=0,
        min_train_months=60,
    )
    coefficients_path = ROOT / "T2_latest_model_coefficients.xlsx"
    with pd.ExcelWriter(coefficients_path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        coefficients_df.to_excel(writer, sheet_name="Coefficients", index=False)
        metadata_df = pd.DataFrame(
            {
                "field": ["latest_test_date", "model_name", "portfolio", "alpha", "l1_ratio"],
                "value": [latest_date, winner.name, winner.portfolio, 0.060, 0.35],
            }
        )
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Metadata"]
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        worksheet.set_column(1, 1, 15, date_format)

    logging.info(f"Model coefficients saved to {coefficients_path.name}")
    logging.info("=" * 80)
    logging.info("STEP FIVE NEW COMPLETED SUCCESSFULLY")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
