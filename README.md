## T2 Factor Timing Long–Short — Full Repository Documentation

### Last updated: 2025-08-31  
Version: 3.0 (Long–Short)

This README documents the entire codebase, including the main flow and Archive utilities. It explains what each program does, all inputs and outputs, the workflow, missing-data policy, Excel formatting practices, and how to run everything end-to-end.


## Project overview

The T2 Factor Timing system builds country portfolios by timing a broad set of macro, valuation, quality, momentum, technical, and commodity factors. It:
- Normalizes and tidies historical factor data across countries
- Forms factor portfolios (Top 20% or soft 15–25% band)
- Computes factor net returns vs an equal-weight benchmark
- Optimizes multi-factor allocations through time (now long–short capable)
- Translates factor weights into country weights (net; shorts allowed)
- Measures final portfolio performance
- Analyzes performance by market regimes and asset classes
- Produces professional Excel workbooks and high-quality PDF reports

### Long–Short Mode (what’s new in 3.0)
- Step 5: Long–short factor optimizer with net/gross constraints (default 200/100 → net=1.0, gross=3.0).
- Step 7: Visuals show net factor weights; latest chart is diverging by sign, small multiples centered at zero.
- Step 8: Vectorized country mapping; negative factor weights select bottom countries; produces net country weights.
- Step 9: Portfolio returns computed as Σ w×r (no renormalization); turnover unchanged (0.5×Σ|Δw|).
- Step 14: Country-level long–short optimization with net/gross constraints and long/short sheets.
- Steps 15–20: Docs and file paths updated; factor exposure analyses remain compatible.


## End-to-end workflow

```mermaid
flowchart LR
  Z0[Step 0: Create P2P scores<br/>Step Zero Create P2P Scores.py] --> A1
  A1[Step 1: Create T2 Master<br/>Step One Create T2Master.py] --> A2
  A2[Step 2: Normalize & tidy<br/>Step Two Create Normalized Tidy.py] --> A25
  A25[Step 2.5: Benchmarks<br/>Step Two Point Five Create Benchmark Rets.py] --> B3
  B3[Step 3: Top 20 portfolios<br/>Step Three Top20 Portfolios.py] --> B4
  B4[Step 4: Monthly net returns + T60<br/>Step Four Create Monthly Top20 Returns.py] --> C5
  C5[Step 5: Optimizer (FAST)<br/>Step Five FAST.py] --> D6
  D6[Step 6: Country alphas<br/>Step Six Create Country alphas from Factor alphas.py] --> E7
  C5 --> E7
  C5 --> G7
  E7[Step 7: Visualize factor weights<br/>Step Seven Visualize Factor Weights.py] --> H8
  G7[Step 8: Write country weights<br/>Step Eight Write Country Weights.py] --> H8
  H8[Step 9: Portfolio returns<br/>Step Nine Calculate Portfolio Returns.py] --> I10
  I10[Step 10: Final report<br/>Step Ten Create Final Report.py] --> J14
  H8 --> J14
  D6 --> J14
  J14[Step 14: Target optimization<br/>Step Fourteen Target Optimization.py] --> K15
  K15[Step 15: Market regime (market conditions)<br/>Step Fifteen Market Regime Analysis.py] --> L16
  D6 --> L16
  L16[Step 16: Factor performance by conditions<br/>Step Sixteen Market Regime Analysis.py] --> M17
  H8 --> M17
  M17[Step 17: GMM regime analysis<br/>Step Seventeen Market Regime Analysis.py] --> N18
  H8 --> N18
  N18[Step 18: Asset class charts<br/>Step Eighteen Asset Class Charts.py] --> O20
  H8 --> O20
  O20[Step 20: PORCH factor exposures<br/>Step Twenty PORCH.py] --> PFF
  D6 --> PFF
  H8 --> PFF
  PFF[FINALFINAL: Latest alphas + weights<br/>Step FINALFINAL.py]
```


## Quick start

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Core files expected in repo root (or same working folder):
  - `T2 Master.xlsx` (created by Step 1)
  - `Normalized_T2_Master.xlsx`, `Normalized_T2_MasterCSV.csv` (created by Step 2)
  - `Portfolio_Data.xlsx` (created by Step 2.5)
  - `T2_Optimizer.xlsx` + `T60.xlsx` (created by Step 4)
  - `T2_rolling_window_weights.xlsx`, `T2_strategy_statistics.xlsx` (created by Step 5)

If starting from raw Bloomberg data, begin at Step 0 → Step 1. If you already have `T2 Master.xlsx`, you can start at Step 2.


## Detailed step-by-step documentation

Each step lists purpose, inputs, outputs, and important notes.

### Step 0 — Create P2P scores
- Program: `Step Zero Create P2P Scores.py`
- Purpose: Build Price-to-Peak (P2P) momentum scores for country ETFs; produces historical scores for optional use.
- Inputs:
  - `AssetList.xlsx` (tickers list)
- Outputs:
  - `P2P_Country_Historical_Scores.xlsx` (historical P2P scores)
- How it works (high level):
  - Downloads monthly prices (SPY + country tickers) via Yahoo Finance
  - Computes per-ticker 12-month price-to-peak and trend (slope×R²)
  - P2P score = latest price/rolling-12M-peak × R² × sign(slope)
  - Forms Top-20% baskets next month to evaluate strategy vs equal-weight and SPY
  - Writes a clean, wide Excel of historical P2P scores (dates + tickers)
- Edge cases: Skips tickers with insufficient history; ignores NaNs in month windows

### Step 1 — Create T2 Master
- Program: `Step One Create T2Master.py`
- Purpose: Ingest raw Bloomberg multi-sheet data → clean, align, engineer features; produce `T2 Master.xlsx`.
- Key operations:
  - Validates required sheets (returns, prices, 120MA, valuation, FX, bonds, EPS, commodities)
  - Standardizes dates, forward-fills within country, local outlier flags, winsorization (5–95th)
  - Computes forward returns (1/3/6/9/12M), trailing returns (1/3/12M), spreads (e.g., 12M-1M)
  - Writes multiple sheets with consistent types and formatting
- Logs: `T2_processing.log` captures all steps and warnings

### Step 2 — Create normalized tidy
- Program: `Step Two Create Normalized Tidy.py`
- Purpose: Normalize each variable cross-sectionally (CS) and in time-series (TS) to make features comparable.
- Methodology:
  - For each sheet/variable in `T2 Master.xlsx`:
    - CS: z-score by date across countries
    - TS: expanding z-score within country (min_periods=12)
  - Returns (1M/3M/6M/9M/12M) are retained in original units
  - Invert selected “lower is better” variables (multiply by -1)
- Outputs: wide (`Normalized_T2_Master.xlsx`) + long (`Normalized_T2_MasterCSV.csv`)

### Step 2.5 — Create benchmark returns
- Program: `Step Two Point Five Create Benchmark Rets.py`
- Builds:
  - Equal-weight benchmark from `1MRet`
  - Market-cap-weighted benchmark from `Mcap Weights`
  - US/SPX reference (if present)
- Data alignment:
  - Aligns dates/columns; pads/truncates `Mcap Weights` to returns index; uses common countries only
- Output workbook `Portfolio_Data.xlsx`:
  - `Returns`: country monthly returns (decimal)
  - `Weights`: market cap weights aligned to returns index
  - `Benchmarks`: columns include `equal_weight` (and others if present)

### Step 3 — Top 20 portfolios (hard cut)
- Program: `Step Three Top20 Portfolios.py`
- Methodology:
  - For each factor and month: rank countries by factor value; select Top 20%; equal-weight them
  - Compute factor monthly return and excess (vs equal-weight)
  - Track holdings (date×country=1 if included)
- Outputs:
  - `T2 Top20.xlsx` performance table (IR, vol, drawdown, hit rate, etc.)
  - `T2 Top20.pdf` charts (cumulative excess, rolling metrics)
  - `T2_Top_20_Exposure.csv` (binary exposure matrix: Date, Country, factor columns)

### Step 4 — Monthly Top20 net returns (fuzzy soft-band)
- Program: `Step Four Create Monthly Top20 Returns.py`
- Methodology (fuzzy logic):
  - Instead of hard Top 20%, uses a soft band: full weight if rank < 15%; linear taper from 15–25%; 0 below 25%
  - For each factor and date: compute portfolio return minus equal-weight benchmark → net returns
  - Builds `T60.xlsx`: 60-month trailing averages of factor net returns (shifted one month)
- Outputs:
  - `T2_Optimizer.xlsx` sheet `Monthly_Net_Returns`
  - `T60.xlsx` sheet `T60`

### Step 5 — Optimizer (FAST)
- Program: `Step Five FAST.py`
- Objective (convex, long–short): maximize w′μ − λ·w′Σw − γ·(||w_long||²+||w_short||²) with w=w_long−w_short; net/gross constraints (default net=1.0, gross=3.0), 0≤legs≤100%.
- Design:
  - CVXPY + OSQP; warm-start; hybrid window (first 60 months expanding, then exact 60 rolling)
  - Optional sign gating to steer longs to positive μ and shorts to negative μ
- Outputs:
  - `T2_rolling_window_weights.xlsx` sheets: Net_Weights, Long_Weights, Short_Weights
  - `T2_strategy_statistics.xlsx` (Summary Statistics, Monthly Returns)
  - Visuals: `T2_factor_weight_heatmap.pdf`, `T2_strategy_performance.pdf`

### Step 6 — Country alphas from factor alphas
- Program: `Step Six Create Country alphas from Factor alphas.py`
- Methodology:
  - For each month: country alpha = sum_over_factors(exposure_country,factor × factor_alpha)
  - Exposures from `T2_Top_20_Exposure.csv` (per date); alphas from `T60.xlsx`
  - Missing exposure values filled by factor-date average; logs imputations
- Output workbook `T2_Country_Alphas.xlsx`:
  - `Country_Scores`: dates×countries country alpha
  - `Data_Quality`: completeness by country
  - `Factor_Counts`: valid factors used per country/date
  - `Missing_Data_Log`: imputation details

### Step 7 — Visualize factor weights
- Program: `Step Seven Visualize Factor Weights.py`
- Methodology:
  - Reads `T2_rolling_window_weights.xlsx` (Net_Weights by default)
  - Selects factors by maximum absolute weight (≥5% default)
  - Exports latest diverging bar chart (by sign) + small multiples centered at zero

### Step 8 — Write country weights (fuzzy mapping)
- Program: `Step Eight Write Country Weights.py`
- Methodology:
  - For each date’s factor weights (net): rank countries; positive weights use top ranks, negative use bottom ranks
  - Apply fuzzy 15–25% soft-band per factor; column-normalize; multiply by factor weights (with sign)
  - Aggregate across factors to net country weights (sum ≈ 1.0; negatives allowed)
- Outputs:
  - `T2_Final_Country_Weights.xlsx` (All Periods [net], Summary Statistics, Latest Weights)
  - `T2_Country_Final.xlsx` (ordered `Country Weights` sheet with per-factor totals row)

### Step 9 — Calculate portfolio returns
- Program: `Step Nine Calculate Portfolio Returns.py`
- Methodology:
  - Align monthly dates between `T2_Final_Country_Weights.xlsx` and `Portfolio_Data.xlsx`
  - For each month: portfolio return = Σ country_weight × country_return (no renormalization; supports shorts)
  - Turnover: 0.5 × Σ|w_t − w_{t-1}|; plus rolling and cumulative turnover stats
- Outputs (Excel):
  - `Monthly Returns`: Portfolio, Equal Weight, Net Return
  - `Cumulative Returns`: growth-of-1 series
  - `Statistics`: annualized returns/vol, Sharpe, drawdown, hit rate, skew, kurtosis (+ turnover stats for portfolio)
  - `Net Returns`, `Turnover Analysis`
- PDF: `T2_Final_Portfolio_Returns.pdf` with 3-panel charts

### Step 10 — Final strategy report
- Program: `Step Ten Create Final Report.py`
- Methodology:
  - Loads all key Excel outputs
  - Builds comprehensive PDF with tables and charts: Top 20, factor returns, current factor allocation, country allocations, period returns, and enhanced analyses
- Output: `T2_Strategy_Report_Comprehensive_YYYY-MM-DD.pdf`

### Step 14 — Target optimization (country-level)
- Program: `Step Fourteen Target Optimization.py`
- Objective (long–short): maximize alpha − drift_penalty − turnover_cost using net weights with net/gross constraints (default 200/100)
- Data alignment:
  - Intersects dates and countries across target weights, returns, and alphas
  - Uses previous month’s weights rolled forward with returns to compute turnover relative to proposed weights
- Outputs:
  - `T2_Optimized_Country_Weights.xlsx` with:
    - `Optimized_Weights` (net), `Long_Optimized_Weights`, `Short_Optimized_Weights`, `Latest_Weights`
    - `Monthly_Returns`, `Cumulative_Returns`, `Performance_Statistics`, `Turnover_Analysis`
    - `Optimization_Metrics`, `Summary_Statistics`, `Country_Alphas`, `Configuration`
  - Visuals: `T2_Optimized_Strategy_Analysis.pdf`, `T2_Turnover_Analysis.pdf`, `T2_Weighted_Average_Factor_Exposure.pdf`, `T2_Weighted_Average_Factor_Rolling_Analysis.pdf`

### Step 15 — Market regime analysis (market conditions approach)
- Program: `Step Fifteen Market Regime Analysis.py`
- Regimes:
  - Market trend (Bull/Correction/Bear) via drawdown from peaks of equal-weight benchmark
  - Volatility regime (Low/Normal/High) via rolling vol percentiles
  - Economic proxy via rolling 12M returns
- Outputs: Excel and PDF summarizing strategy performance per regime and factor drivers

### Step 16 — Factor performance by market conditions
- Program: `Step Sixteen Market Regime Analysis.py`
- Conditions:
  - Volatility (Low/Med/High), Momentum (Weak/Neutral/Strong), Dispersion (Low/Med/High), Economic Cycle (Contraction/Normal/Expansion)
- Analytics:
  - Compute factor returns and excess returns using `T2_Top_20_Exposure.csv` and 1M returns
  - Performance tables by condition, conditional correlations, quilt charts
- Outputs: `T2_Factor_Market_Condition_Analysis.xlsx` + PDFs

### Step 17 — GMM regime analysis (data-driven)
- Program: `Step Seventeen Market Regime Analysis.py`
- Methodology:
  - Build features from benchmark (vol short/long, skew, kurt)
  - Fit Gaussian Mixture Models (n=2..6) and choose by BIC
  - Label regimes (e.g., High-Vol Bearish / Low-Vol Bullish); analyze strategy/factors by regime
- Outputs: Excel + two PDFs (performance and factor analyses)

### Step 18 — Asset class charts
- Program: `Step Eighteen Asset Class Charts.py`
- Methodology:
  - Map countries → asset classes via `Step Factor Categories.xlsx` sheet `Asset Class`
  - Aggregate country weights to asset-class time series for Final & Optimized
  - Plot combined timeline and per-class pages; also produce per-country multi-page PDF
- Outputs: `T2_Asset_Class.xlsx`, `T2_Asset_Class_Weights.pdf`, `T2_Asset_Class_Split.pdf`, `T2_Country_Weights.xlsx`, `T2_Country_Weights_Split.pdf`, log file

### Step 20 — PORCH factor exposure comparison
- Program: `Step Twenty PORCH.py`
- Methodology:
  - Get latest country weights for Final and Optimized; build equal-weight benchmark
  - Pivot latest factor exposures from `Normalized_T2_MasterCSV.csv` for the latest date
  - Compute portfolio-level factor exposures = Σ w_country × exposure_country,factor
  - Create multi-page PDF tables + Excel with top differences and summary stats

### FINALFINAL — Latest alphas and weights snapshot
- Program: `Step FINALFINAL.py`
- Methodology: Merge latest country alphas (`T2_Country_Alphas.xlsx` last row) with `T2_Optimized_Country_Weights.xlsx` `Latest_Weights`
- Output: `T2_FINAL_T60.xlsx` `Latest_Country_Alpha_Weights`


## Main outputs (what they contain)

- `T2 Master.xlsx`: Clean master dataset with many sheets (forward/trailing returns, valuation, technicals, macro, commodities).
- `Normalized_T2_Master.xlsx`: Wide, normalized data by variable; `_CS` (cross-section), `_TS` (time-series) suffixes.
- `Normalized_T2_MasterCSV.csv`: Long, tidy dataset used across downstream steps.
- `Portfolio_Data.xlsx`: Standardized returns, weights, and benchmark series.
- `T2 Top20.xlsx` / `T2 Top20.pdf`: Factor portfolio performance (Top 20% hard-cut version) and charts.
- `T2_Top_20_Exposure.csv`: Binary exposure matrix (date, country, factors as columns).
- `T2_Optimizer.xlsx`: Monthly factor net returns vs equal-weight benchmark.
- `T60.xlsx`: 60-month trailing averages of factor net returns (shifted by one month).
- `T2_rolling_window_weights.xlsx`: Optimized factor weights through time (hybrid window).
- `T2_strategy_statistics.xlsx`: Side-by-side stats for different optimization windows; includes Monthly Returns sheet.
- `T2_factor_weight_heatmap.pdf` / `T2_strategy_performance.pdf`: Visualization of factor allocation and cumulative performance.
- `T2_Final_Country_Weights.xlsx`: Country weights All Periods, summary stats, and Latest Weights snapshot.
- `T2_Country_Final.xlsx`: Latest ordered country weights with factor contribution totals.
- `T2_Final_Portfolio_Returns.xlsx` / `.pdf`: Portfolio, benchmark, net, cumulative, and turnover analyses.
- `T2_Optimized_Country_Weights.xlsx`: Optimized country weights plus returns, stats, exposures, and configuration.
- `T2_Optimized_Strategy_Analysis.pdf`, `T2_Turnover_Analysis.pdf`, `T2_Weighted_Average_Factor_Exposure.pdf`, `T2_Weighted_Average_Factor_Rolling_Analysis.pdf`: Optimization visuals and factor exposure change analysis.
- `T2_Factor_Market_Condition_Analysis.xlsx` + PDFs `T2_Factor_Performance_Heatmap.pdf`, `T2_Factor_Quilt_Chart.pdf`, `T2_Factor_Correlation_Analysis.pdf`: Factor performance under market conditions.
- `T2_GMM_Regime_Analysis.xlsx`, `T2_GMM_Regime_Performance.pdf`, `T2_GMM_Factor_Analysis.pdf`: GMM-based regime results.
- `T2_Asset_Class.xlsx`, `T2_Asset_Class_Weights.pdf`, `T2_Asset_Class_Split.pdf`, `T2_Country_Weights.xlsx`, `T2_Country_Weights_Split.pdf`, `T2_Asset_Class_MissingDataLog.txt`: Asset class and country-level visualizations and logs.
- `T2_Portfolio_Factor_Exposures.pdf` / `.xlsx`: Factor exposure tables comparing Final vs Optimized vs Equal-Weight.
- `T2_FINAL_T60.xlsx`: Latest month country alphas and country weights merged.
- Logs: `T2_processing.log`, `optimization_log.txt` for auditability.


## Missing data policy (repository-wide)

- Country-level missing data are handled consistently:
  - Step 1: forward-fill where appropriate, with outlier control
  - Step 2: forward-fill then 0 for residuals (documented in step code)
  - Step 4: cross-sectional fill to avoid gaps for T60 calculations
  - Step 6: missing country-factor exposures filled with factor-date means, with imputation log in output
  - Step 9: restricts analysis to dates common to weights and returns; last month excluded if future returns not available
  - Step 18: missing asset-class mappings filled by timepoint mean and logged in `T2_Asset_Class_MissingDataLog.txt`

Maintain a log of replacements and data completeness outputs (see `T2_Country_Alphas.xlsx` sheets and Step 18 log) to audit data quality.


## Excel date formatting guidelines (as implemented)

Across Excel writers, date columns are formatted for readability, typically `dd-mmm-yyyy` or `yyyy-mm-dd`, and date columns’ widths are set (≈ 12–15). Where relevant:
- Dates are first converted to proper `datetime`
- Index dates are promoted to a visible Date column when needed
- Trailing/rolling calculations are shifted correctly to avoid look-ahead bias


## How to run the pipeline

You can run the steps manually in order (recommended to keep control of outputs):

```bash
python "Step Zero Create P2P Scores.py"               # optional
python "Step One Create T2Master.py"
python "Step Two Create Normalized Tidy.py"
python "Step Two Point Five Create Benchmark Rets.py"
python "Step Three Top20 Portfolios.py"
python "Step Four Create Monthly Top20 Returns.py"
python "Step Five FAST.py"
python "Step Six Create Country alphas from Factor alphas.py"
python "Step Seven Visualize Factor Weights.py"
python "Step Eight Write Country Weights.py"
python "Step Nine Calculate Portfolio Returns.py"
python "Step Ten Create Final Report.py"
python "Step Fourteen Target Optimization.py"
python "Step Fifteen Market Regime Analysis.py"
python "Step Sixteen Market Regime Analysis.py"
python "Step Seventeen Market Regime Analysis.py"
python "Step Eighteen Asset Class Charts.py"
python "Step Twenty PORCH.py"
python "Step FINALFINAL.py"
```

Archive also contains a pipeline runner (`Archive/Step Run All.py`) that executes a legacy sequence.


## Archive utilities (not main flow)

These are reference or experimental utilities. They do not write core outputs unless noted.
- `Archive/Step Five 60 Month Optimal Portfolios.py`: Earlier optimizer with additional visuals and exports (superseded by FAST).
- `Archive/Step Six Decile Analysis.py`: Performance by decile (1–10) and lookback (24–90m); writes a PDF chart.
- `Archive/Step Five Grid Search *NOT PART OF FLOW*.py`: Factor count vs lookback grid search (console outputs).
- `Archive/Step Six Grid Search *NOT PART OF FLOW*.py` and `Archive/Step Six Grid Search.ipynb`: Similar grid search notebooks/scripts (console outputs).
- `Archive/Step_Six_T2_Factor_timing_top3.py`, `Archive/Step Six T2 Factor Timing Top3.py`, `Archive/Step Six T2 Factor timing top3.ipynb`: Top-3 factor timing variants; write exploratory Excel files.
- `Archive/Step Four Create Monthly Top20 Returns OLD.py`, `Archive/Step Three Top20 Portfolios OLD.py`: Historical versions of Steps 3–4.
- `Archive/Step Thirteen Create Country alphas from Factor Alphas.py`: Earlier country alpha computation using `Normalized_T2_MasterCSV.csv`.
- `Archive/Step Eleven Compare Strategies.py`: Multi-portfolio comparison that reads a folder of portfolio files and produces consolidated charts/Excel.


## Troubleshooting

- File not found: verify that prerequisite outputs exist before running downstream steps (e.g., run Step 4 before Step 5).
- Misaligned dates: ensure monthly indices are standardized to month-end or first-of-month consistently; most scripts convert to period-based timestamps.
- Optimization solver issues: for CVXPY/OSQP, ensure package versions match those in `requirements.txt` and consider reducing problem size or adjusting penalties.
- Missing countries or factors: check the normalization and exposure outputs; look at data quality sheets and the Step 18 missing-data log.


## System requirements and performance

- Designed for modern Macs (M-series) and high RAM; vectorized pandas/NumPy and CVXPY accelerate heavy steps.
- Some steps (Step 5 FAST, Step 14) exploit warm-start and hybrid windows for speed.


## Change log

- 2.1 (2025-08-08): Added deeper per-program methodology details for every step.  
- 2.0 (2025-08-08): Full rewrite of README to match actual scripts and outputs (includes Archive).  
- 1.x (2025-06 to 2025-07): Multiple pipeline improvements (fuzzy soft-band, FAST optimizer, regime analyses).


## Contact

For questions, open an issue or consult the docstrings at the top of each script (they list detailed INPUT FILES / OUTPUT FILES, version, and last updated).
