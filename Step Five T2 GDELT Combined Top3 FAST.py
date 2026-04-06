"""
=============================================================================
SCRIPT NAME: Step Five T2 GDELT Combined Top3 FAST.py - T2 + GDELT combined optimization (Top 3)
=============================================================================

INPUT FILES:
- T2_GDELT_Combined_Top3_Optimizer.xlsx: Monthly factor returns from Step Four Combined Top3
- Step Factor Categories T2 GDELT Combined.xlsx: Merged T2 + GDELT caps

OUTPUT FILES:
- T2_GDELT_Combined_Top3_rolling_window_weights.xlsx
- T2_GDELT_Combined_Top3_strategy_statistics.xlsx
- T2_GDELT_Combined_Top3_factor_weight_heatmap.pdf
- T2_GDELT_Combined_Top3_strategy_performance.pdf

VERSION: 1.0
LAST UPDATED: 2026-04-03

DESCRIPTION:
Identical optimization logic to Step Five T2 GDELT Combined FAST.py but reads
the Top-3 factor return files produced by Step Four Combined Top3 and writes
outputs with the Top3 suffix.

Key optimizations:
1. CVXPY/OSQP convex optimization engine
2. Warm-start from previous solutions
3. Batch data preparation and pre-computed windows
4. Vectorized constraint processing
5. Sparse matrix operations where applicable

DEPENDENCIES:
- pandas, numpy, cvxpy, matplotlib, openpyxl

USAGE:
python "Step Five T2 GDELT Combined Top3 FAST.py"

NOTES:
- Uses CVXPY with OSQP solver for fast convex optimization
- ``USE_COVARIANCE`` (default False): when False or ``LAMBDA`` is 0, no sample Σ is built
  and the objective is scaled expected return minus HHI penalty only (no ``quad_form``).
- Set ``USE_COVARIANCE = True`` and ``LAMBDA > 0`` to re-enable mean–variance risk term
=============================================================================
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime
import logging
import time
import warnings

from step_five_multiwindow_stats import log_step_five_multiwindow_table

warnings.filterwarnings('ignore')
plt.style.use('default')

# =============================================================================
# FILE PATHS — Top-3 variants
# =============================================================================
INPUT_OPTIMIZER   = "T2_GDELT_Combined_Top3_Optimizer.xlsx"
INPUT_FACTOR_CATS = "Step Factor Categories T2 GDELT Combined.xlsx"
OUTPUT_WEIGHTS    = "T2_GDELT_Combined_Top3_rolling_window_weights.xlsx"
OUTPUT_STATS      = "T2_GDELT_Combined_Top3_strategy_statistics.xlsx"
OUTPUT_HEATMAP    = "T2_GDELT_Combined_Top3_factor_weight_heatmap.pdf"
OUTPUT_PERF       = "T2_GDELT_Combined_Top3_strategy_performance.pdf"

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================
LAMBDA = 0.0
HHI_PENALTY = 0.005
WINDOW_SIZE_T2    = 60
WINDOW_SIZE_GDELT = 12
USE_COVARIANCE = False
GDELT_PREFIX = "GDELT_"

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('T2_processing.log', mode='a'),
            logging.StreamHandler()
        ]
    )

class FastPortfolioOptimizer:
    """
    High-performance portfolio optimizer using CVXPY with warm-start capabilities.

    Maximize: w'μ - λ*w'Σw - γ*||w||²
    Subject to: sum(w) = 1, 0 ≤ w ≤ max_weights
    """

    def __init__(self, n_assets, factor_names, lambda_param=1.0, hhi_penalty=0.01, max_weights=None):
        self.n_assets = n_assets
        self.factor_names = factor_names
        self.lambda_param = lambda_param
        self.hhi_penalty = hhi_penalty

        if max_weights is None:
            self.max_weights_array = np.ones(n_assets)
        else:
            self.max_weights_array = np.array([max_weights.get(name, 1.0) for name in factor_names])

        self.weights_var = cp.Variable(n_assets)
        self.constraints = [
            self.weights_var >= 0,
            self.weights_var <= self.max_weights_array,
            cp.sum(self.weights_var) == 1
        ]
        self.prev_weights = None

    def optimize_weights(self, expected_returns, covariance_matrix=None):
        portfolio_return = self.weights_var.T @ expected_returns
        concentration_penalty = self.hhi_penalty * cp.sum_squares(self.weights_var)

        use_risk = (covariance_matrix is not None and self.lambda_param > 0)
        if use_risk:
            risk_penalty = self.lambda_param * cp.quad_form(self.weights_var, covariance_matrix)
            objective = cp.Maximize(portfolio_return - risk_penalty - concentration_penalty)
        else:
            objective = cp.Maximize(portfolio_return - concentration_penalty)

        problem = cp.Problem(objective, self.constraints)

        if self.prev_weights is not None:
            self.weights_var.value = self.prev_weights

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if problem.status == cp.OPTIMAL:
                optimal_weights = self.weights_var.value
                self.prev_weights = optimal_weights.copy()
                return optimal_weights
            else:
                logging.warning(f"Optimization failed with status: {problem.status}")
                return np.ones(self.n_assets) / self.n_assets
        except Exception as e:
            logging.error(f"Optimization error: {e}")
            return np.ones(self.n_assets) / self.n_assets

def load_and_prepare_data():
    logging.info("Loading input data...")

    returns = pd.read_excel(INPUT_OPTIMIZER, index_col=0)
    returns.index = pd.to_datetime(returns.index)

    factor_categories = pd.read_excel(INPUT_FACTOR_CATS)
    max_weights = dict(zip(factor_categories['Factor Name'], factor_categories['Max']))

    returns = returns.apply(pd.to_numeric, errors="coerce") / 100.0

    if 'Monthly Return_CS' in returns.columns:
        returns = returns.drop(columns=['Monthly Return_CS'])

    allowed_factors = set(max_weights.keys())
    extra_cols = [c for c in returns.columns if c not in allowed_factors]
    if extra_cols:
        logging.warning(
            "Dropping %d columns not in factor categories file: %s",
            len(extra_cols), extra_cols,
        )
        returns = returns[[c for c in returns.columns if c in allowed_factors]]

    logging.info(f"Loaded returns data: {returns.shape[0]} periods, {returns.shape[1]} factors")
    logging.info(f"Loaded max weight constraints for {len(max_weights)} factors")

    gdelt_cols = [c for c in returns.columns if c.startswith(GDELT_PREFIX)]
    t2_cols    = [c for c in returns.columns if not c.startswith(GDELT_PREFIX)]
    logging.info(f"Factor split: {len(t2_cols)} T2 (window {WINDOW_SIZE_T2}m), "
                 f"{len(gdelt_cols)} GDELT (window {WINDOW_SIZE_GDELT}m)")

    if USE_COVARIANCE and LAMBDA > 0:
        n_f, n_t = returns.shape[1], returns.shape[0]
        min_win = min(WINDOW_SIZE_T2, WINDOW_SIZE_GDELT)
        if n_f > n_t:
            logging.warning(
                "More factors (%d) than time periods (%d): sample covariance is rank-deficient.", n_f, n_t,
            )
        if n_f > min_win:
            logging.warning(
                "Factor count (%d) > smallest WINDOW_SIZE (%d): rolling covariance is under-identified.", n_f, min_win,
            )

    return returns, max_weights

def calculate_rolling_statistics(returns_df):
    logging.info("Pre-computing rolling statistics...")
    logging.info("Window sizes — T2: %d months, GDELT: %d months", WINDOW_SIZE_T2, WINDOW_SIZE_GDELT)
    if not USE_COVARIANCE or LAMBDA == 0:
        logging.info(
            "USE_COVARIANCE=%s, LAMBDA=%s — objective uses expected returns + HHI only (no Σ).",
            USE_COVARIANCE, LAMBDA,
        )

    t2_cols    = [c for c in returns_df.columns if not c.startswith(GDELT_PREFIX)]
    gdelt_cols = [c for c in returns_df.columns if c.startswith(GDELT_PREFIX)]
    all_cols   = list(returns_df.columns)

    dates = returns_df.index
    expected_returns_dict = {}
    covariance_dict = {}
    dates_list = []

    for i, date in enumerate(dates[1:], 1):
        if i <= WINDOW_SIZE_T2:
            t2_window = returns_df[t2_cols].iloc[:i]
        else:
            t2_window = returns_df[t2_cols].iloc[i - WINDOW_SIZE_T2:i]

        if i <= WINDOW_SIZE_GDELT:
            gdelt_window = returns_df[gdelt_cols].iloc[:i]
        else:
            gdelt_window = returns_df[gdelt_cols].iloc[i - WINDOW_SIZE_GDELT:i]

        if len(t2_window) < 1 or len(gdelt_window) < 1:
            continue

        t2_means    = t2_window.mean(axis=0)
        gdelt_means = gdelt_window.mean(axis=0)

        combined_means = pd.concat([t2_means, gdelt_means]).reindex(all_cols)
        expected_returns = 8 * combined_means.values

        if USE_COVARIANCE and LAMBDA > 0:
            max_win = max(WINDOW_SIZE_T2, WINDOW_SIZE_GDELT)
            if i <= max_win:
                full_window = returns_df.iloc[:i]
            else:
                full_window = returns_df.iloc[i - max_win:i]
            cov_matrix = np.cov(full_window.values.T, ddof=0) * 12
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-6)
                cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            except np.linalg.LinAlgError:
                avg_var = np.diag(cov_matrix).mean()
                cov_matrix = np.eye(len(all_cols)) * avg_var
            covariance_dict[date] = cov_matrix
        else:
            covariance_dict[date] = None

        expected_returns_dict[date] = expected_returns
        dates_list.append(date)

    logging.info(f"Pre-computed statistics for {len(dates_list)} optimization periods")
    return expected_returns_dict, covariance_dict, dates_list

def run_fast_optimization():
    start_time = time.time()

    returns_df, max_weights = load_and_prepare_data()

    next_month_date = returns_df.index[-1] + pd.DateOffset(months=1)
    logging.info(f"Will calculate extra month optimization for: {next_month_date.strftime('%Y-%m')}")

    expected_returns_dict, covariance_dict, optimization_dates = calculate_rolling_statistics(returns_df)

    n_assets = len(returns_df.columns)
    factor_names = list(returns_df.columns)

    optimizer = FastPortfolioOptimizer(
        n_assets=n_assets,
        factor_names=factor_names,
        lambda_param=LAMBDA,
        hhi_penalty=HHI_PENALTY,
        max_weights=max_weights
    )

    extended_dates = optimization_dates + [next_month_date]
    weights_df = pd.DataFrame(index=extended_dates, columns=factor_names)

    logging.info("Running batch portfolio optimization...")

    for i, date in enumerate(optimization_dates):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Optimizing {i+1}/{len(optimization_dates)} periods... ({elapsed:.1f}s elapsed)")

        expected_returns = expected_returns_dict[date]
        covariance_matrix = covariance_dict[date]

        optimal_weights = optimizer.optimize_weights(expected_returns, covariance_matrix)

        optimal_weights = np.maximum(optimal_weights, 0)

        weight_sum = optimal_weights.sum()
        if abs(weight_sum - 1.0) > 1e-6:
            scaled_weights = optimal_weights / weight_sum
            max_weights_array = np.array([max_weights.get(name, 1.0) for name in factor_names])
            if np.any(scaled_weights > max_weights_array + 1e-8):
                logging.warning(f"Renormalization would violate max weight constraints. Sum: {weight_sum:.6f}")
            else:
                optimal_weights = scaled_weights

        weights_df.loc[date] = optimal_weights

    logging.info(f"Calculating extra month optimization for {next_month_date.strftime('%Y-%m')}")

    t2_cols    = [c for c in factor_names if not c.startswith(GDELT_PREFIX)]
    gdelt_cols = [c for c in factor_names if c.startswith(GDELT_PREFIX)]

    t2_start    = max(0, len(returns_df.index) - WINDOW_SIZE_T2)
    gdelt_start = max(0, len(returns_df.index) - WINDOW_SIZE_GDELT)

    t2_means    = returns_df[t2_cols].iloc[t2_start:].mean(axis=0)
    gdelt_means = returns_df[gdelt_cols].iloc[gdelt_start:].mean(axis=0)
    combined_means = pd.concat([t2_means, gdelt_means]).reindex(factor_names)
    expected_returns_extra = 8 * combined_means.values

    if USE_COVARIANCE and LAMBDA > 0:
        max_win = max(WINDOW_SIZE_T2, WINDOW_SIZE_GDELT)
        cov_start = max(0, len(returns_df.index) - max_win)
        extra_month_data = returns_df.iloc[cov_start:]
        cov_matrix_extra = np.cov(extra_month_data.values.T, ddof=0) * 12
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix_extra)
            eigenvals = np.maximum(eigenvals, 1e-6)
            cov_matrix_extra = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            avg_var = np.diag(cov_matrix_extra).mean()
            cov_matrix_extra = np.eye(len(factor_names)) * avg_var
    else:
        cov_matrix_extra = None

    optimal_weights_extra = optimizer.optimize_weights(expected_returns_extra, cov_matrix_extra)

    optimal_weights_extra = np.maximum(optimal_weights_extra, 0)
    weight_sum = optimal_weights_extra.sum()
    if abs(weight_sum - 1.0) > 1e-6:
        scaled_weights = optimal_weights_extra / weight_sum
        max_weights_array = np.array([max_weights.get(name, 1.0) for name in factor_names])
        if not np.any(scaled_weights > max_weights_array + 1e-8):
            optimal_weights_extra = scaled_weights

    weights_df.loc[next_month_date] = optimal_weights_extra

    total_time = time.time() - start_time
    logging.info(f"Optimization completed in {total_time:.1f} seconds")
    logging.info(f"Average time per optimization: {total_time/len(extended_dates):.3f} seconds")
    logging.info(f"Extra month ({next_month_date.strftime('%Y-%m')}) optimization completed")

    return weights_df, returns_df

def calculate_strategy_performance(weights_df, returns_df):
    logging.info("Calculating strategy performance...")

    portfolio_returns = []
    aligned_dates = []

    for date in weights_df.index:
        if date in returns_df.index:
            next_month_idx = returns_df.index.get_loc(date) + 1
            if next_month_idx < len(returns_df.index):
                next_month_date = returns_df.index[next_month_idx]
                weights = weights_df.loc[date].values
                returns = returns_df.loc[next_month_date].values
                portfolio_return = np.sum(weights * returns)
                portfolio_returns.append(portfolio_return)
                aligned_dates.append(next_month_date)

    portfolio_returns_series = pd.Series(portfolio_returns, index=aligned_dates)

    ann_return = (1 + portfolio_returns_series.mean())**12 - 1
    ann_vol = portfolio_returns_series.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0

    cum_returns = (1 + portfolio_returns_series).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    weight_changes = weights_df.diff().abs()
    monthly_turnover = weight_changes.sum(axis=1) / 2
    avg_turnover = monthly_turnover.mean()

    stats = {
        'Annualized Return (%)': ann_return * 100,
        'Annualized Volatility (%)': ann_vol * 100,
        'Sharpe Ratio': sharpe,
        'Maximum Drawdown (%)': max_drawdown * 100,
        'Average Monthly Turnover (%)': avg_turnover * 100,
        'Positive Months (%)': (portfolio_returns_series > 0).mean() * 100,
        'Skewness': portfolio_returns_series.skew(),
        'Kurtosis': portfolio_returns_series.kurtosis()
    }

    results = {
        'statistics': stats,
        'monthly_returns': portfolio_returns_series,
        'monthly_turnover': monthly_turnover
    }

    return results

def create_factor_weight_heatmap(weights_df, top_n=20):
    logging.info("Creating factor weight heatmap...")

    weights_df = weights_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    categories = {
        'Value': ['Best PE', 'Best PBK', 'Best Price Sales', 'Best Div Yield', 'Trailing PE', 'Positive PE',
                 'Shiller PE', 'Earnings Yield', 'Best Cash Flow', 'EV to EBITDA'],
        'Momentum': ['1MTR', '3MTR', '12-1MTR', 'RSI14', '120MA', 'Advance Decline', 'Signal', 'P2P'],
        'Economic': ['Inflation', 'REER', '10Yr Bond', 'GDP', 'Debt to GDP', 'Budget Def', 'Current Account'],
        'Quality': ['Best ROE', 'BEST EPS', 'Operating Margin', 'Trailing EPS', 'Debt to EV', 'LT Growth', 'Bloom Country Risk', '20 day vol'],
        'Commodity': ['Oil', 'Gold', 'Copper', 'Agriculture', 'Currency']
    }

    category_colors = {
        'Value': '#2d5a3d',
        'Momentum': '#1e3a5f',
        'Economic': '#5d2e5a',
        'Quality': '#5a3d2d',
        'Commodity': '#5a4b2d'
    }

    category_backgrounds = {
        'Value': '#f8fdfb',
        'Momentum': '#f8fafd',
        'Economic': '#fdf8fd',
        'Quality': '#fdfbf8',
        'Commodity': '#fdfcf8'
    }

    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_factors = avg_weights.head(top_n).index.tolist()

    filtered_weights_df = weights_df[top_factors]

    categorized_factors = []
    category_map = {}
    category_groups = {}

    for category, keywords in categories.items():
        category_factors = []
        for factor in top_factors:
            for keyword in keywords:
                if keyword in factor and factor not in categorized_factors:
                    category_factors.append(factor)
                    category_map[factor] = category
                    break
        if category_factors:
            category_factors.sort()
            categorized_factors.extend(category_factors)

    for factor in top_factors:
        if factor not in categorized_factors:
            categorized_factors.append(factor)
            category_map[factor] = "Other"

    ordered_factors = categorized_factors[:top_n] if len(categorized_factors) > 0 else top_factors

    current_pos = 0
    for category in ['Value', 'Momentum', 'Economic', 'Quality', 'Commodity']:
        cat_factors = [f for f in ordered_factors if category_map.get(f) == category]
        if cat_factors:
            category_groups[category] = {
                'start': current_pos,
                'end': current_pos + len(cat_factors) - 1,
                'factors': cat_factors
            }
            current_pos += len(cat_factors)

    sample_weights = filtered_weights_df[ordered_factors].iloc[::4].copy()
    if len(sample_weights) == 0:
        logging.warning("No weight data available for heatmap")
        return

    sample_weights = sample_weights.astype(float)

    clean_factor_names = []
    for factor in ordered_factors:
        clean_name = factor.replace('_', ' ')
        clean_name = clean_name.replace('1MTR', '1 Month Return')
        clean_factor_names.append(clean_name)

    colors_cmap = ['#f8fffe', '#e8f5f3', '#d1ebe6', '#a8dadc', '#79c2d0', '#5aa9c4', '#457b9d']
    sophisticated_cmap = mcolors.LinearSegmentedColormap.from_list("sophisticated", colors_cmap, N=256)
    sophisticated_cmap.set_under('white')

    fig, ax = plt.subplots(figsize=(20, 12))
    plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    for category, group_info in category_groups.items():
        start_pos = group_info['start'] - 0.5
        height = group_info['end'] - group_info['start'] + 1
        background_rect = Rectangle(
            (-0.5, start_pos), len(sample_weights.index), height,
            facecolor=category_backgrounds[category],
            edgecolor='none', alpha=0.3, zorder=0
        )
        ax.add_patch(background_rect)

    max_value = sample_weights.values.max()
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 0.5
    else:
        max_value = min(max_value, 0.5)

    im = ax.imshow(sample_weights.T.values, cmap=sophisticated_cmap, aspect='auto',
                   vmin=0.001, vmax=max_value)

    tick_positions = range(0, len(sample_weights.index), 3)
    ax.set_xticks(tick_positions)
    date_labels = [sample_weights.index[i].strftime('%Y') for i in tick_positions]
    ax.set_xticklabels(date_labels, fontsize=12, fontweight='500', color='#333333')

    ax.set_yticks(range(len(clean_factor_names)))
    ax.set_yticklabels(clean_factor_names, fontsize=11, fontweight='400', color='#333333')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label('Portfolio Weight (%)', rotation=270, labelpad=25,
                   fontsize=13, fontweight='600', color='#333333')
    cbar.ax.tick_params(labelsize=11, colors='#333333')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0%}'))
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#cccccc')

    ax.set_xticks(np.arange(-0.5, len(sample_weights.index), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clean_factor_names), 1), minor=True)
    ax.grid(which="minor", color="#f0f0f0", linestyle='-', linewidth=0.5, alpha=0.8)

    for category, group_info in category_groups.items():
        end_pos = group_info['end']
        if end_pos < len(ordered_factors) - 1:
            ax.axhline(y=end_pos + 0.5, color='#999999', linewidth=1.0, alpha=0.8, zorder=4)

    middle_x = len(sample_weights.index) / 2
    for category, group_info in category_groups.items():
        start_pos = group_info['start']
        end_pos = group_info['end']
        center_pos = (start_pos + end_pos) / 2

        ax.text(middle_x, center_pos, category,
                fontsize=16, fontweight='700', color=category_colors[category],
                ha='center', va='center', rotation=0,
                bbox=dict(boxstyle="round,pad=0.5",
                         facecolor='white',
                         edgecolor=category_colors[category],
                         linewidth=2.0, alpha=0.95),
                zorder=5)

    ax.set_title('Portfolio Weight Allocation Over Time (Top 3)\nTop 20 Factors by Average Weight',
                 fontsize=22, fontweight='700', color='#1a1a1a', pad=30,
                 fontfamily='serif')

    ax.set_xlabel('Year', fontsize=16, fontweight='600', color='#333333', labelpad=15)
    ax.set_ylabel('Investment Factor', fontsize=16, fontweight='600', color='#333333', labelpad=15)

    ax.tick_params(axis='both', which='both', length=0, pad=8)
    ax.set_facecolor('#fcfcfc')
    fig.patch.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#dddddd')

    plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)
    plt.tight_layout()

    plt.savefig(OUTPUT_HEATMAP, bbox_inches='tight', dpi=300, facecolor='white',
                edgecolor='none', format='pdf')
    plt.close()

    logging.info(f"Factor weight heatmap saved to {OUTPUT_HEATMAP}")

def create_strategy_performance_plot(performance_results):
    logging.info("Creating strategy performance plot...")

    monthly_returns = performance_results['monthly_returns']
    cum_returns = (1 + monthly_returns).cumprod()

    plt.figure(figsize=(15, 8))
    plt.plot(cum_returns.index, cum_returns.values, label='Top-3 Strategy', linewidth=2, color='#457b9d')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')

    plt.title('T2 + GDELT Combined Top-3 Strategy Cumulative Performance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1%}'.format(y-1)))

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')

    stats = performance_results['statistics']
    ann_return = stats['Annualized Return (%)']
    sharpe_ratio = stats['Sharpe Ratio']
    max_dd = stats['Maximum Drawdown (%)']

    plt.figtext(0.02, 0.02,
                f'Annual Return: {ann_return:.1f}%\nSharpe Ratio: {sharpe_ratio:.2f}\nMax Drawdown: {max_dd:.1f}%',
                fontsize=11,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=10))

    plt.tight_layout()
    plt.savefig(OUTPUT_PERF, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Strategy performance plot saved to {OUTPUT_PERF}")

def save_results(weights_df, performance_results):
    logging.info("Saving results to Excel files...")

    weights_df.to_excel(OUTPUT_WEIGHTS)
    logging.info(f"Weights saved to {OUTPUT_WEIGHTS}")

    with pd.ExcelWriter(OUTPUT_STATS, engine='xlsxwriter') as writer:
        stats_df = pd.DataFrame(list(performance_results['statistics'].items()),
                               columns=['Metric', 'Value'])
        stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

        returns_df = pd.DataFrame({
            'Top3 Strategy Returns': performance_results['monthly_returns']
        })
        returns_df.to_excel(writer, sheet_name='Monthly Returns')

        turnover_df = pd.DataFrame({
            'Monthly Turnover': performance_results['monthly_turnover']
        })
        turnover_df.to_excel(writer, sheet_name='Monthly Turnover')

        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        for sheet_name in ['Monthly Returns', 'Monthly Turnover']:
            if sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column(0, 0, 12, date_format)

    logging.info(f"Strategy statistics saved to {OUTPUT_STATS}")

    create_factor_weight_heatmap(weights_df)
    create_strategy_performance_plot(performance_results)

def main():
    setup_logging()

    logging.info("="*80)
    logging.info("COMBINED T2+GDELT - STEP FIVE TOP-3 FAST: OPTIMIZATION")
    logging.info("="*80)
    logging.info("Optimization Engine: CVXPY with OSQP solver")
    logging.info("Features: Warm-start, batch processing, vectorized operations")
    logging.info("="*80)

    try:
        weights_df, returns_df = run_fast_optimization()
        performance_results = calculate_strategy_performance(weights_df, returns_df)
        save_results(weights_df, performance_results)

        logging.info("="*80)
        logging.info("OPTIMIZATION RESULTS SUMMARY (multi-window)")
        logging.info("="*80)
        log_step_five_multiwindow_table(
            performance_results["monthly_returns"],
            performance_results["monthly_turnover"],
            logging.info,
        )
        logging.info("="*80)
        logging.info("STEP FIVE T2 GDELT COMBINED TOP-3 FAST COMPLETED SUCCESSFULLY")
        logging.info("="*80)

    except Exception as e:
        logging.error(f"Error in Step Five T2 GDELT Combined Top3 FAST: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
