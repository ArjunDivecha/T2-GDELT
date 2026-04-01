"""
=============================================================================
SCRIPT NAME: Step Five FAST.py — Long–Short Factor Optimization (Hybrid)
=============================================================================

INPUT FILES:
- T2_Optimizer.xlsx: Monthly factor net returns (from Step Four)

OUTPUT FILES:
- T2_rolling_window_weights.xlsx:
  - Sheet Net_Weights: optimized net factor weights (can be negative)
  - Sheet Long_Weights: optimized long leg (>= 0)
  - Sheet Short_Weights: optimized short leg (>= 0)
- T2_strategy_statistics.xlsx: Strategy performance statistics and monthly returns
- T2_factor_weight_heatmap.pdf: Heatmap of net factor weights through time
- T2_strategy_performance.pdf: Cumulative performance visualization

VERSION: 3.0 (Long–Short)
LAST UPDATED: 2025-08-31
AUTHOR: Quant Team

DESCRIPTION:
High-performance long–short factor optimization using CVXPY + OSQP. Implements a hybrid
window: first 60 months expanding, then rolling 60 months. The optimizer uses separate
long and short vectors (w_long, w_short) with net weights w = w_long − w_short and enforces
target net and gross exposure via constraints. A small HHI-like penalty controls
concentration; optional sign gating steers longs to positive-μ factors and shorts to
negative-μ factors.

Key features:
1) Long–short variables with net/gross constraints (e.g., 200/100 ⇒ net=1.0, gross=3.0)
2) Warm-started CVXPY/OSQP for speed
3) Batch pre-computation of μ and Σ per window
4) Optional sign gating with feasibility safeguards

DEPENDENCIES:
- pandas, numpy, cvxpy, osqp, matplotlib, openpyxl, seaborn

USAGE:
python "Step Five FAST.py"

NOTES:
- Defaults: NET_TARGET=1.0, GROSS_TARGET=3.0, per-factor caps 100% on each leg
- Heatmap/plots use net weights; downstream scripts read Net_Weights by default
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

warnings.filterwarnings('ignore')
plt.style.use('default')

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Portfolio optimization parameters
LAMBDA = 0.5                # Risk aversion parameter (higher = more risk-averse)
HHI_PENALTY = 0.005         # Concentration penalty (higher = more diversified)
WINDOW_SIZE = 60            # Rolling window size in months
# EMA_DECAY removed - using simple arithmetic mean like original Step Five

# Long–Short configuration (200/100 with 100% per-factor caps)
LONG_SHORT = False           # Enable long–short optimization
NET_TARGET = 1.0            # 200% long - 100% short = +1.0 net exposure
GROSS_TARGET = 3.0          # 2.0 long + 1.0 short gross exposure
MAX_LONG_DEFAULT = 1.0      # Max long weight per factor (100%)
MAX_SHORT_DEFAULT = 1.0     # Max short weight per factor (100%)
USE_SIGN_GATING = True      # Only long factors with positive μ and short negative μ
SIGN_THRESHOLD = 0.0        # μ threshold for gating (0 = any sign)

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging():
    """Setup logging configuration"""
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
    High-performance long–short portfolio optimizer using CVXPY with warm-start.

    Net weights: w = w_long - w_short
    Objective:   maximize w'μ - λ·w'Σw - γ·(||w_long||² + ||w_short||²)
    Constraints: sum(w_long) - sum(w_short) = NET_TARGET
                 sum(w_long) + sum(w_short) = GROSS_TARGET
                 0 ≤ w_long ≤ max_long, 0 ≤ w_short ≤ max_short
    """

    def __init__(self, n_assets, factor_names, lambda_param=1.0, hhi_penalty=0.01,
                 max_long=None, max_short=None):
        self.n_assets = n_assets
        self.factor_names = factor_names
        self.lambda_param = lambda_param
        self.hhi_penalty = hhi_penalty

        # Per-factor caps (defaults to 100%)
        self.max_long_array = np.array([ (max_long or {}).get(name, MAX_LONG_DEFAULT) for name in factor_names ])
        self.max_short_array = np.array([ (max_short or {}).get(name, MAX_SHORT_DEFAULT) for name in factor_names ])

        # Variables (reused across periods)
        self.w_long = cp.Variable(n_assets)
        self.w_short = cp.Variable(n_assets)

        # Parameters (per-period)
        self.mu_param = cp.Parameter(n_assets)
        self.cov_param = cp.Parameter((n_assets, n_assets), PSD=True)
        self.max_long_param = cp.Parameter(n_assets, nonneg=True)
        self.max_short_param = cp.Parameter(n_assets, nonneg=True)

        w_net = self.w_long - self.w_short

        # Objective
        ret = self.mu_param @ w_net
        risk = self.lambda_param * cp.quad_form(w_net, self.cov_param)
        conc = self.hhi_penalty * (cp.sum_squares(self.w_long) + cp.sum_squares(self.w_short))
        objective = cp.Maximize(ret - risk - conc)

        # Constraints
        constraints = [
            self.w_long >= 0, self.w_short >= 0,
            cp.sum(self.w_long) - cp.sum(self.w_short) == NET_TARGET,
            cp.sum(self.w_long) + cp.sum(self.w_short) == GROSS_TARGET,
            self.w_long <= self.max_long_param,
            self.w_short <= self.max_short_param,
        ]

        self.problem = cp.Problem(objective, constraints)
        self.prev_w_long = None
        self.prev_w_short = None

    def optimize_weights(self, expected_returns, covariance_matrix):
        # Ensure numpy arrays with proper shape
        mu = np.asarray(expected_returns, dtype=float).reshape(-1)
        cov = np.asarray(covariance_matrix, dtype=float)

        # Optional sign gating: only allow long where μ>τ and short where μ<−τ
        if USE_SIGN_GATING:
            allow_long = (mu > SIGN_THRESHOLD).astype(float)
            allow_short = (mu < -SIGN_THRESHOLD).astype(float)
            # Feasibility check: relax gating if not enough capacity for gross targets
            if allow_long.sum() == 0 or self.max_long_array[allow_long > 0].sum() < GROSS_TARGET / 2:
                allow_long = np.ones_like(mu)
            if allow_short.sum() == 0 or self.max_short_array[allow_short > 0].sum() < GROSS_TARGET / 2:
                allow_short = np.ones_like(mu)
        else:
            allow_long = np.ones_like(mu)
            allow_short = np.ones_like(mu)

        self.mu_param.value = mu
        self.cov_param.value = cov
        self.max_long_param.value = self.max_long_array * allow_long
        self.max_short_param.value = self.max_short_array * allow_short

        # Warm start
        if self.prev_w_long is not None:
            self.w_long.value = self.prev_w_long
            self.w_short.value = self.prev_w_short

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                logging.warning(f"Optimization status: {self.problem.status}")
            w_long = np.clip(self.w_long.value, 0, None)
            w_short = np.clip(self.w_short.value, 0, None)
            self.prev_w_long, self.prev_w_short = w_long, w_short
            return w_long - w_short, w_long, w_short
        except Exception as e:
            logging.error(f"Optimization error: {e}")
            # Fallback: evenly split gross across factors, match net target
            n = self.n_assets
            long_share = GROSS_TARGET + NET_TARGET
            short_share = GROSS_TARGET - NET_TARGET
            w_long = np.full(n, long_share / (2*GROSS_TARGET) * (GROSS_TARGET / n))
            w_short = np.full(n, short_share / (2*GROSS_TARGET) * (GROSS_TARGET / n))
            return w_long - w_short, w_long, w_short

def load_and_prepare_data():
    """
    Load and prepare all data with optimized pandas operations.
    
    Returns:
        tuple: (returns_df, max_long_dict, max_short_dict)
    """
    logging.info("Loading input data...")
    
    # Load returns data
    returns = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    returns.index = pd.to_datetime(returns.index)
    
    # Convert to decimals if needed
    if returns.abs().mean().mean() > 1:
        returns = returns / 100
    
    # Remove Monthly Return_CS if present
    if 'Monthly Return_CS' in returns.columns:
        returns = returns.drop(columns=['Monthly Return_CS'])
    
    logging.info(f"Loaded returns data: {returns.shape[0]} periods, {returns.shape[1]} factors")

    # Per-factor caps: 100% on both long and short legs
    cols = list(returns.columns)
    max_long = {name: MAX_LONG_DEFAULT for name in cols}
    max_short = {name: MAX_SHORT_DEFAULT for name in cols}

    return returns, max_long, max_short

def calculate_rolling_statistics(returns_df, window_size):
    """
    Pre-compute all rolling statistics for batch optimization.
    
    Args:
        returns_df: DataFrame of factor returns
        window_size: Rolling window size (60 months)
        # decay_factor parameter removed - using simple arithmetic mean
        
    Returns:
        tuple: (expected_returns_dict, covariance_dict, dates_list)
    """
    logging.info("Pre-computing rolling statistics...")
    
    dates = returns_df.index
    expected_returns_dict = {}
    covariance_dict = {}
    dates_list = []
    
    for i, date in enumerate(dates[1:], 1):  # Start from month 2
        # Determine window bounds
        if i <= window_size:
            # Expanding window (first 60 months)
            window_data = returns_df.iloc[:i]
        else:
            # Rolling window (60 months)
            window_data = returns_df.iloc[i-window_size:i]
        
        # Skip if insufficient data (start from month 2 like original)
        if len(window_data) < 1:  # Need at least 1 month (match original behavior)
            continue
            
        # Calculate simple arithmetic mean expected returns (matching original Step Five line 433)
        # Note: Original uses raw returns for optimization, scaling only for debug output
        factor_means = window_data.mean(axis=0)
        # Apply 8x scaling factor to match original utility calculation (line 148 in original)
        expected_returns = 8 * factor_means
        
        # Covariance matrix (annualized)
        # Use ddof=0 to match original np.std(portfolio_returns) calculation (line 145)
        cov_matrix = np.cov(window_data.values.T, ddof=0) * 12
        
        # Ensure positive semi-definite with more robust method
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Higher floor for numerical stability
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except np.linalg.LinAlgError:
            # Fall back to identity matrix scaled by average variance
            avg_var = np.diag(cov_matrix).mean()
            cov_matrix = np.eye(len(window_data.columns)) * avg_var
        
        expected_returns_dict[date] = expected_returns
        covariance_dict[date] = cov_matrix
        dates_list.append(date)
    
    logging.info(f"Pre-computed statistics for {len(dates_list)} optimization periods")
    return expected_returns_dict, covariance_dict, dates_list

def run_fast_optimization():
    """
    Main optimization pipeline using CVXPY with all performance optimizations.
    """
    start_time = time.time()
    
    # Load data
    returns_df, max_long, max_short = load_and_prepare_data()
    
    # Calculate next month date for extra optimization (matching 60 Month program)
    next_month_date = returns_df.index[-1] + pd.DateOffset(months=1)
    logging.info(f"Will calculate extra month optimization for: {next_month_date.strftime('%Y-%m')}")
    
    # Pre-compute all rolling statistics
    expected_returns_dict, covariance_dict, optimization_dates = calculate_rolling_statistics(
        returns_df, WINDOW_SIZE
    )
    
    # Initialize fast optimizer
    n_assets = len(returns_df.columns)
    factor_names = list(returns_df.columns)
    
    optimizer = FastPortfolioOptimizer(
        n_assets=n_assets,
        factor_names=factor_names,
        lambda_param=LAMBDA,
        hhi_penalty=HHI_PENALTY,
        max_long=max_long,
        max_short=max_short
    )
    
    # Initialize results storage - include the extra month in the index
    extended_dates = optimization_dates + [next_month_date]
    net_df = pd.DataFrame(index=extended_dates, columns=factor_names, dtype=float)
    long_df = pd.DataFrame(index=extended_dates, columns=factor_names, dtype=float)
    short_df = pd.DataFrame(index=extended_dates, columns=factor_names, dtype=float)
    
    # Run batch optimization for existing dates
    logging.info("Running batch portfolio optimization...")
    
    for i, date in enumerate(optimization_dates):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Optimizing {i+1}/{len(optimization_dates)} periods... ({elapsed:.1f}s elapsed)")
        
        # Get pre-computed statistics
        expected_returns = expected_returns_dict[date]
        covariance_matrix = covariance_dict[date]
        
        # Optimize weights (net, long, short)
        w_net, w_long, w_short = optimizer.optimize_weights(expected_returns, covariance_matrix)

        # Store results
        net_df.loc[date] = w_net
        long_df.loc[date] = w_long
        short_df.loc[date] = w_short
    
    # Calculate the extra month optimization (matching 60 Month program logic)
    logging.info(f"Calculating extra month optimization for {next_month_date.strftime('%Y-%m')}")
    
    # Use the last 60 months of data for the next month (same as 60 Month program)
    start_idx = max(0, len(returns_df.index) - WINDOW_SIZE)
    extra_month_data = returns_df.iloc[start_idx:]
    
    # Calculate expected returns and covariance for extra month
    factor_means = extra_month_data.mean(axis=0)
    expected_returns_extra = 8 * factor_means  # Apply 8x scaling factor
    
    # Covariance matrix (annualized)
    cov_matrix_extra = np.cov(extra_month_data.values.T, ddof=0) * 12
    
    # Ensure positive semi-definite
    try:
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix_extra)
        eigenvals = np.maximum(eigenvals, 1e-6)
        cov_matrix_extra = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    except np.linalg.LinAlgError:
        avg_var = np.diag(cov_matrix_extra).mean()
        cov_matrix_extra = np.eye(len(extra_month_data.columns)) * avg_var
    
    # Optimize weights for extra month
    w_net_extra, w_long_extra, w_short_extra = optimizer.optimize_weights(expected_returns_extra, cov_matrix_extra)

    # Store extra month results
    net_df.loc[next_month_date] = w_net_extra
    long_df.loc[next_month_date] = w_long_extra
    short_df.loc[next_month_date] = w_short_extra
    
    total_time = time.time() - start_time
    logging.info(f"Optimization completed in {total_time:.1f} seconds")
    logging.info(f"Average time per optimization: {total_time/len(extended_dates):.3f} seconds")
    logging.info(f"Extra month ({next_month_date.strftime('%Y-%m')}) optimization completed")
    
    return net_df, long_df, short_df, returns_df

def calculate_strategy_performance(weights_df, returns_df):
    """
    Calculate strategy performance metrics and returns.
    
    Args:
        weights_df: DataFrame of optimal weights
        returns_df: DataFrame of factor returns
        
    Returns:
        dict: Performance statistics and time series
    """
    logging.info("Calculating strategy performance...")
    
    # Calculate portfolio returns
    portfolio_returns = []
    aligned_dates = []
    
    for date in weights_df.index:
        if date in returns_df.index:
            # Get next month's returns (forward-looking)
            next_month_idx = returns_df.index.get_loc(date) + 1
            if next_month_idx < len(returns_df.index):
                next_month_date = returns_df.index[next_month_idx]
                weights = weights_df.loc[date].values
                returns = returns_df.loc[next_month_date].values
                
                # Calculate portfolio return
                portfolio_return = np.sum(weights * returns)
                portfolio_returns.append(portfolio_return)
                aligned_dates.append(next_month_date)
    
    # Create portfolio returns series
    portfolio_returns_series = pd.Series(portfolio_returns, index=aligned_dates)
    
    # Calculate performance statistics
    ann_return = (1 + portfolio_returns_series.mean())**12 - 1
    ann_vol = portfolio_returns_series.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + portfolio_returns_series).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Turnover analysis
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
    """Create an award-winning, sophisticated factor weight heatmap with executive dashboard styling"""
    logging.info("Creating award-winning factor weight heatmap...")
    
    # Ensure all data is numeric by converting any non-numeric values to float
    weights_df = weights_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Define factor categories and their prefixes/keywords
    categories = {
        'Value': ['Best PE', 'Best PBK', 'Best Price Sales', 'Best Div Yield', 'Trailing PE', 'Positive PE', 
                 'Shiller PE', 'Earnings Yield', 'Best Cash Flow', 'EV to EBITDA'],
        'Momentum': ['1MTR', '3MTR', '12-1MTR', 'RSI14', '120MA', 'Advance Decline', 'Signal', 'P2P'],
        'Economic': ['Inflation', 'REER', '10Yr Bond', 'GDP', 'Debt to GDP', 'Budget Def', 'Current Account'],
        'Quality': ['Best ROE', 'BEST EPS', 'Operating Margin', 'Trailing EPS', 'Debt to EV', 'LT Growth', 'Bloom Country Risk', '20 day vol'],
        'Commodity': ['Oil', 'Gold', 'Copper', 'Agriculture', 'Currency']
    }

    # Sophisticated category styling
    category_colors = {
        'Value': '#2d5a3d',      # Deep forest green
        'Momentum': '#1e3a5f',   # Deep navy blue  
        'Economic': '#5d2e5a',   # Deep purple
        'Quality': '#5a3d2d',    # Deep brown
        'Commodity': '#5a4b2d'   # Deep golden brown
    }
    
    category_backgrounds = {
        'Value': '#f8fdfb',      # Very light green tint
        'Momentum': '#f8fafd',   # Very light blue tint
        'Economic': '#fdf8fd',   # Very light purple tint
        'Quality': '#fdfbf8',    # Very light brown tint
        'Commodity': '#fdfcf8'   # Very light golden tint
    }

    # Get top factors by average weight
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_factors = avg_weights.head(top_n).index.tolist()
    
    # Ensure we're working with the subset of weights that are in top_factors
    filtered_weights_df = weights_df[top_factors]

    # Organize factors by category
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

    # Calculate category positions
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

    # Sample data every 4 months for better visualization
    sample_weights = filtered_weights_df[ordered_factors].iloc[::4].copy()
    if len(sample_weights) == 0:
        logging.warning("No weight data available for heatmap")
        return
        
    # Ensure data is numeric and handle any remaining issues
    sample_weights = sample_weights.astype(float)

    # Use full factor names (no truncation for professional look) and rename 1MTR
    clean_factor_names = []
    for factor in ordered_factors:
        clean_name = factor.replace('_', ' ')
        clean_name = clean_name.replace('1MTR', '1 Month Return')
        clean_factor_names.append(clean_name)

    # Create figure with optimal proportions
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Set sophisticated styling
    plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Create category background rectangles
    for category, group_info in category_groups.items():
        start_pos = group_info['start'] - 0.5
        height = group_info['end'] - group_info['start'] + 1
        
        # Add subtle background tinting
        background_rect = Rectangle(
            (-0.5, start_pos), len(sample_weights.index), height,
            facecolor=category_backgrounds[category], 
            edgecolor='none', alpha=0.3, zorder=0
        )
        ax.add_patch(background_rect)

    # Create the main heatmap with diverging colormap centered at zero (long–short net weights)
    values = sample_weights.T.values
    max_abs = np.nanmax(np.abs(values)) if np.isfinite(np.nanmax(np.abs(values))) else 0.5
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 0.5
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.cm.RdBu_r  # diverging colormap
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')

    # Sophisticated tick formatting
    tick_positions = range(0, len(sample_weights.index), 3)  # Every 3rd year for cleaner look
    ax.set_xticks(tick_positions)
    date_labels = [sample_weights.index[i].strftime('%Y') for i in tick_positions]
    ax.set_xticklabels(date_labels, fontsize=12, fontweight='500', color='#333333')
    
    ax.set_yticks(range(len(clean_factor_names)))
    ax.set_yticklabels(clean_factor_names, fontsize=11, fontweight='400', color='#333333')

    # Professional colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label('Net Weight (%)', rotation=270, labelpad=25, 
                   fontsize=13, fontweight='600', color='#333333')
    cbar.ax.tick_params(labelsize=11, colors='#333333')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0%}'))
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor('#cccccc')

    # Elegant grid
    ax.set_xticks(np.arange(-0.5, len(sample_weights.index), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(clean_factor_names), 1), minor=True)
    ax.grid(which="minor", color="#f0f0f0", linestyle='-', linewidth=0.5, alpha=0.8)

    # Add thin horizontal lines between categories for clear boundaries
    for category, group_info in category_groups.items():
        end_pos = group_info['end']
        # Draw thin line below each category (except the last one)
        if end_pos < len(ordered_factors) - 1:
            ax.axhline(y=end_pos + 0.5, color='#999999', linewidth=1.0, alpha=0.8, zorder=4)

    # Add horizontal category headers in the middle of the heatmap
    middle_x = len(sample_weights.index) / 2
    for category, group_info in category_groups.items():
        start_pos = group_info['start']
        end_pos = group_info['end']
        center_pos = (start_pos + end_pos) / 2
        
        # Add elegant horizontal category header (no rotation)
        ax.text(middle_x, center_pos, category,
                fontsize=16, fontweight='700', color=category_colors[category],
                ha='center', va='center', rotation=0,
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor='white', 
                         edgecolor=category_colors[category],
                         linewidth=2.0, alpha=0.95),
                zorder=5)

    # Professional title with elegant typography
    ax.set_title('Portfolio Weight Allocation Over Time\nTop 20 Factors by Average Weight', 
                 fontsize=22, fontweight='700', color='#1a1a1a', pad=30,
                 fontfamily='serif')

    # Sophisticated axis labels
    ax.set_xlabel('Year', fontsize=16, fontweight='600', color='#333333', labelpad=15)
    ax.set_ylabel('Investment Factor', fontsize=16, fontweight='600', color='#333333', labelpad=15)

    # Remove tick marks for cleaner appearance
    ax.tick_params(axis='both', which='both', length=0, pad=8)

    # Set elegant background
    ax.set_facecolor('#fcfcfc')
    fig.patch.set_facecolor('white')

    # Add subtle frame
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#dddddd')

    # Optimal layout with generous margins
    plt.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)
    plt.tight_layout()

    # Save with high quality
    plt.savefig('T2_factor_weight_heatmap.pdf',
                bbox_inches='tight', dpi=300, facecolor='white', 
                edgecolor='none', format='pdf')
    plt.close()
    
    logging.info("✓ Award-winning factor weight heatmap created with sophisticated design")

def create_strategy_performance_plot(performance_results):
    """
    Create strategy performance visualization showing cumulative returns.
    
    Args:
        performance_results: Dict containing monthly returns and statistics
    """
    logging.info("Creating strategy performance plot...")
    
    # Get monthly returns
    monthly_returns = performance_results['monthly_returns']
    
    # Calculate cumulative returns
    cum_returns = (1 + monthly_returns).cumprod()
    
    # Create figure
    plt.figure(figsize=(15, 8))
    plt.plot(cum_returns.index, cum_returns.values, label='Hybrid Strategy', linewidth=2, color='#457b9d')
    
    # Add baseline
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    plt.title('T2 Strategy Cumulative Performance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.1%}'.format(y-1)))
    
    # Enhance grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    # Add performance annotations
    stats = performance_results['statistics']
    ann_return = stats['Annualized Return (%)']
    sharpe_ratio = stats['Sharpe Ratio']
    max_dd = stats['Maximum Drawdown (%)']
    
    plt.figtext(0.02, 0.02, 
                f'Annual Return: {ann_return:.1f}%\nSharpe Ratio: {sharpe_ratio:.2f}\nMax Drawdown: {max_dd:.1f}%', 
                fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=10))
    
    plt.tight_layout()
    plt.savefig('T2_strategy_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Strategy performance plot saved to T2_strategy_performance.pdf")

def save_results(weights_df, performance_results, long_df=None, short_df=None):
    """
    Save results to Excel files and create visualizations.
    
    Args:
        weights_df: DataFrame of optimal weights
        performance_results: Dict of performance metrics
    """
    logging.info("Saving results to Excel files...")
    
    # Save weights (write Net as default Sheet1 for downstream compatibility; also add LS sheets)
    weights_output_file = 'T2_rolling_window_weights.xlsx'
    with pd.ExcelWriter(weights_output_file, engine='xlsxwriter') as writer:
        # Backward-compatible default sheet
        weights_df.to_excel(writer, sheet_name='Sheet1')
        # Explicitly named sheets
        weights_df.to_excel(writer, sheet_name='Net_Weights')
        if long_df is not None:
            long_df.to_excel(writer, sheet_name='Long_Weights')
        if short_df is not None:
            short_df.to_excel(writer, sheet_name='Short_Weights')
    logging.info(f"Weights saved to {weights_output_file} (Net/Long/Short)")
    
    # Save strategy statistics
    stats_output_file = 'T2_strategy_statistics.xlsx'
    
    with pd.ExcelWriter(stats_output_file, engine='xlsxwriter') as writer:
        # Summary statistics
        stats_df = pd.DataFrame(list(performance_results['statistics'].items()),
                               columns=['Metric', 'Value'])
        stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Monthly returns
        returns_df = pd.DataFrame({
            'Hybrid Strategy Returns': performance_results['monthly_returns']
        })
        returns_df.to_excel(writer, sheet_name='Monthly Returns')
        
        # Monthly turnover
        turnover_df = pd.DataFrame({
            'Monthly Turnover': performance_results['monthly_turnover']
        })
        turnover_df.to_excel(writer, sheet_name='Monthly Turnover')
        
        # Apply proper date formatting to sheets with date indices
        workbook = writer.book
        date_format = workbook.add_format({'num_format': 'dd-mmm-yyyy'})
        
        # Format date columns in sheets with date indices
        for sheet_name in ['Monthly Returns', 'Monthly Turnover']:
            if sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column(0, 0, 12, date_format)  # Format index column (dates)
    
    logging.info(f"Strategy statistics saved to {stats_output_file}")
    
    # Create visualizations
    create_factor_weight_heatmap(weights_df)
    create_strategy_performance_plot(performance_results)

def main():
    """Main execution function"""
    setup_logging()
    
    logging.info("="*80)
    logging.info("T2 FACTOR TIMING - STEP FIVE FAST: HIGH-PERFORMANCE OPTIMIZATION")
    logging.info("="*80)
    logging.info("Optimization Engine: CVXPY with OSQP solver")
    logging.info("Features: Warm-start, batch processing, vectorized operations")
    logging.info("="*80)
    
    try:
        # Run optimization (long–short)
        weights_df, long_df, short_df, returns_df = run_fast_optimization()
        
        # Calculate performance
        performance_results = calculate_strategy_performance(weights_df, returns_df)
        
        # Save results (include long/short breakdown)
        save_results(weights_df, performance_results, long_df=long_df, short_df=short_df)
        
        # Display summary
        logging.info("="*80)
        logging.info("OPTIMIZATION RESULTS SUMMARY")
        logging.info("="*80)
        
        for metric, value in performance_results['statistics'].items():
            logging.info(f"{metric:30s}: {value:8.2f}")
        
        logging.info("="*80)
        logging.info("STEP FIVE FAST COMPLETED SUCCESSFULLY")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"Error in Step Five FAST: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
