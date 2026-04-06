# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

### Running Individual Steps
```bash
python "Step Zero Create P2P Scores.py"
python "Step One Create T2Master.py"
python "Step Two Create Normalized Tidy.py"
python "Step Two Point Five Create Benchmark Rets.py"
python "Step Three Top20 Portfolios.py"
python "Step Four Create Monthly Top20 Returns.py"
python "Step Five 60 Month Optimal Portfolios.py"
python "Step Seven Visualize Factor Weights.py"
python "Step Eight Write Country Weights.py"
python "Step Nine Calculate Portfolio Returns.py"
python "Step Ten Create Final Report.py"
python "Step Fourteen Target Optimization.py"
python "Step Fifteen Market Regime Analysis.py"
```

### Archived Steps
The following steps have been moved to Archive/ and are not part of the main flow:
- Step Six T2 Factor Timing Top3.py
- Step Eleven Compare Strategies.py
- Step Twelve MultiPeriod Forecast.py
- Step Thirteen Create Country alphas from Factor Alphas.py
- Step Run All.py

### Testing and Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks for interactive analysis
jupyter lab
```

## Architecture

### Pipeline Overview
The T2 Factor Timing system is a sequential pipeline for momentum-based country selection and portfolio optimization:

1. **Data Preparation (Steps 0-2.5)**: 
   - Ingests Bloomberg financial data and P2P scores
   - Creates normalized data with quality enhancements (outlier detection, winsorization)
   - Generates benchmark returns for performance comparison

2. **Portfolio Construction (Steps 3-5)**:
   - Identifies top 20 countries by momentum across multiple lookback periods
   - Calculates monthly returns for each momentum portfolio
   - Performs 60-month rolling window optimization to find optimal factor weights

3. **Portfolio Implementation (Steps 7-9)**:
   - Visualizes factor weights over time
   - Translates factor weights to country-level allocations
   - Calculates final portfolio returns with rebalancing

4. **Reporting and Optimization (Steps 10, 14-15)**:
   - Creates comprehensive performance reports with visualizations
   - Optimizes country weights using CVXPY with turnover constraints
   - Analyzes strategy performance across market regime conditions

### Key Data Flow
```
Bloomberg Data → T2 Master → Normalized Data → Top20 Portfolios → Factor Returns → 
Factor Weights → Country Weights → Portfolio Returns → Performance Reports → 
Target Optimization → Market Regime Analysis
```

### Critical Implementation Details

**Data Quality Pipeline**: Step One implements sophisticated data cleaning with forward-filling, local outlier detection (±3σ rolling windows), and global winsorization. Market cap data receives special handling to preserve large-cap influence.

**Portfolio Optimization**: The strategy uses rolling window optimization to find optimal factor weights that maximize risk-adjusted returns. Factor weights are constrained by maximum allocation limits defined in Step Factor Categories.xlsx.

**Target Optimization**: Step Fourteen implements CVXPY-based optimization that balances three objectives: maximizing portfolio alpha, minimizing drift from rolled-forward weights, and minimizing transaction costs from turnover.

**Market Regime Analysis**: Step Fifteen analyzes strategy performance across different market conditions (Bull/Bear, High/Low Volatility, Economic Expansion/Contraction) and identifies which factors drive performance in each regime.

### File Dependencies
- Input data must be in Excel format with specific sheet names
- All intermediate outputs are Excel files for compatibility
- Visualizations saved as PDFs in outputs/visualizations/
- Date columns must be datetime-indexed throughout the pipeline

### Development and Debugging Tools

**Archived Analysis Files** (in Archive/ directory):
- `Step Six T2 Factor Timing Top3.py` - Factor timing with adaptive rotation
- `Step Six Grid Search.ipynb` - Jupyter notebook for parameter optimization
- `Step Run All.py` - Sequential pipeline execution script
- Various experimental and comparison analysis files

**Logging**: All scripts generate detailed logs to console and `T2_processing.log`

### Data Quality and Error Handling
- Missing country data is filled with the mean of available countries
- Forward-filling handles temporal gaps in data
- Outlier detection uses ±3σ rolling windows with winsorization
- Market cap data receives special handling to preserve large-cap influence
- All processing steps include comprehensive error handling and validation

## Learned User Preferences

- Align GDELT month-end dates to the same reporting convention as other T2 files (e.g. a calendar month-end such as 2/28/2026 is stored as the matching first-of-following-month style such as 3/1/2026 used elsewhere in the pipeline).
- Keep the full pipeline restricted to the GDELT-aligned sample window once dates are aligned (do not silently extend analysis beyond that period).
- For the GDELT factor-category workbook, include all listed factors, including z-scored variants.
- Sign-flip “risk” category variables so their direction matches the intended economic interpretation alongside other factor groups.
- Classic T2 `Step Three Top20 Portfolios Fast.py` regression support (`step_three_regression_utils.py`) uses **simple monthly cross-sectional regression** (one OLS per factor per month), **not** Fama-MacBeth; regress **country excess returns** (1M return minus the equal-weight benchmark) on factor scores, and treat the monthly slope series as the factor return in charts and summaries—do **not** subtract the benchmark again from those slopes.
- LASSO / ElasticNet overlay was removed from Step Three (2026-04-06); only sort-based and univariate OLS regression remain.

## Learned Workspace Facts

- A parallel GDELT branch exists alongside the classic T2 Master path: factors come from `GDELT.xlsx` (and derived tidy outputs); country returns and other return fields still come from T2 Master / existing market inputs, not from GDELT.
- GDELT factors are maintained in three variants per variable: raw, cross-sectional (CS), and time-series (TS).
- The user-aligned analysis start for GDELT-wide work is 2015-09-01 (end date follows GDELT coverage after date harmonization).
- `Step Factor Categories GDELT.xlsx` is the GDELT counterpart to `Step Factor Categories.xlsx` for caps, groupings, and factor lists.
- `GDELT.xlsx` may include documentation-only sheets **`README`** and **`README_VARIABLES`**; **`Step Two GDELT Create Tidy.py`** skips them and only processes wide monthly factor panels (dates in column A).
- When raw Top20 or portfolio outputs disagree between the archived `Archive/Step Three GDELT.py` and `Step Three GDELT Top20 Portfolios Fast.py`, treat Step Two GDELT tidy/normalization as a primary place to reconcile definitions before changing Step Three.
- T2+GDELT **combined Top 3** branch: equal-weight the top three countries per factor per month using `Step Three T2 GDELT Combined Top3 Portfolios Fast.py`, `Step Four T2 GDELT Combined Top3 Returns FAST.py`, and `Step Five T2 GDELT Combined Top3 FAST.py`, producing `T2_GDELT_Combined_Top3_*` outputs parallel to the Top20 combined filenames.
- `T2_GDELT_Combined_Optimizer.xlsx` (Step Four combined output) can have many more columns than true factor signals (for example ~196 columns vs ~168 factors): extras include raw cumulative return fields such as `3MRet`, `6MRet`, `9MRet`, `12MRet` and other non-factor variables. `Step Five T2 GDELT Combined FAST.py` must restrict optimizer columns to those listed in `Step Factor Categories T2 GDELT Combined.xlsx`; any column not in that workbook falls back to a default max weight of 1.0 and can massively inflate reported backtest returns.
- `Step Five T2 GDELT Combined FAST.py` defines `USE_COVARIANCE` (default False). When it is False or `LAMBDA` is 0, no sample covariance matrix is built and the objective is scaled expected return minus HHI penalty only (no `cp.quad_form` risk term); set `USE_COVARIANCE = True` and `LAMBDA > 0` to restore the mean–variance covariance term.
- **`GDELT Top20.xlsx`** (Step Three GDELT) uses sheets **`Full_Sample`**, **`Trailing_1Y`**, **`Trailing_3Y`**, **`Trailing_5Y`**; use **`Full_Sample`** where code assumed a single full-history sheet. Step Four GDELT also writes **`GDELT_RSQ.xlsx`** (`Monthly_RSQ`): R² of OLS on **12-month** trailing cumulative **filled** net returns (aligned with the optimizer); the **first 11** monthly rows have blank RSQ cells. **`Step Five GDELT FAST.py`** drops **all-NaN** months after loading **`GDELT_Optimizer.xlsx`** (e.g. stub incomplete last row) before trailing-window covariance, **symmetrizes** Σ before **`cp.quad_form`**, and logs dropped dates.
- For **factor redundancy**, pairwise correlation on **`GDELT_Optimizer.xlsx`** **`Monthly_Net_Returns`** measures co-movement of **backtest return** series; redundancy in the **signal panel** should use **`GDELT_Factors_MasterCSV.csv`** via **mean monthly cross-sectional** correlation between factors—the two views can differ materially (e.g. defensive vs risk structure in the panel vs returns).