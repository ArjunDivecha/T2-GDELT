"""
=============================================================================
SCRIPT NAME: Step Five Top Three.py — Top-2 forecast factors, return-proportional weights
=============================================================================

INPUT FILES:
- T2_Optimizer.xlsx: Monthly factor returns from Step Four
- Step Factor Categories.xlsx: Factor categories (optional reference; not used for weight caps in this variant)

OUTPUT FILES:
- T2_rolling_window_weights.xlsx: Factor weights (only top 2 non-zero each month)
- T2_strategy_statistics.xlsx: Strategy performance statistics and monthly returns
- T2_factor_weight_heatmap.pdf: Heatmap visualization of factor weights over time
- T2_strategy_performance.pdf: Cumulative performance visualization

VERSION: 3.1
LAST UPDATED: 2026-03-22
AUTHOR: Claude Code

DESCRIPTION:
Each rebalance month, forecast return per factor is the same rolling statistic as before:
simple mean of factor returns in the hybrid window (expanding then 60-month rolling),
times 8 (same scaling as the prior Step Five utility).

Instead of CVXPY optimization, this script:
1. Picks the TWO factors with the highest forecast return.
2. Sets weights proportional to those forecast values. If that would imply a negative
   weight (mixed positive/negative forecasts among the two), weights use the positive
   part of each forecast only, then renormalize to sum to 1. If both forecasts are
   negative, weights are still proportional to those values (ratios are positive and sum to 1).

Hybrid window: first 60 months expanding, then rolling 60 months.

DEPENDENCIES:
- pandas
- numpy
- matplotlib
- openpyxl

USAGE:
python "Step Five Top Three.py"

NOTES:
- No mean-variance optimizer; only ranking + proportional weighting on forecasts.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from datetime import datetime
import logging
import time
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')

# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

WINDOW_SIZE = 60           # Rolling window size in months (after burn-in, uses last 60 months)
TOP_N = 2                  # How many factors to hold each month (highest forecast return)

# =============================================================================
# TOP-K PROPORTIONAL WEIGHTING (replaces CVXPY optimizer)
# =============================================================================

def top_k_proportional_weights(expected_returns: np.ndarray, n_select: int = TOP_N) -> np.ndarray:
    """
    Choose the factors with the highest forecast returns and weight proportional to forecast.

    Args:
        expected_returns: 1D array of per-factor forecast returns (same order as columns).
        n_select: How many factors to hold (default TOP_N).

    Returns:
        Full-length weight vector: non-zero only on the selected factors; sum to 1; all >= 0.
    """
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    n_assets = len(mu)
    w = np.zeros(n_assets)
    k = min(n_select, n_assets)
    if k < 1:
        return w

    # Indices of top-k forecasts (descending)
    order = np.argsort(-mu)
    top_idx = order[:k]
    m = mu[top_idx]

    s = float(np.sum(m))
    if abs(s) < 1e-15:
        w[top_idx] = 1.0 / k
        return w

    part = m / s
    # Mixed signs can make some weights negative; use positive part only, then normalize
    if np.any(part < -1e-12):
        pos = np.maximum(m, 0.0)
        sp = float(np.sum(pos))
        if sp > 1e-15:
            part = pos / sp
        else:
            part = np.ones(k) / k

    w[top_idx] = part
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total > 0:
        w = w / total
    return w


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

def load_and_prepare_data():
    """
    Load and prepare all data with optimized pandas operations.
    
    Returns:
        tuple: (returns_df, max_weights_dict) — max_weights kept for parity with other Step Five scripts
    """
    logging.info("Loading input data...")
    
    # Load returns data
    returns = pd.read_excel('T2_Optimizer.xlsx', index_col=0)
    returns.index = pd.to_datetime(returns.index)
    
    # Load factor constraints 
    factor_categories = pd.read_excel('Step Factor Categories.xlsx')
    max_weights = dict(zip(factor_categories['Factor Name'], factor_categories['Max']))
    
    # Convert to decimals if needed
    if returns.abs().mean().mean() > 1:
        returns = returns / 100
    
    # Remove Monthly Return_CS if present
    if 'Monthly Return_CS' in returns.columns:
        returns = returns.drop(columns=['Monthly Return_CS'])
    
    logging.info(f"Loaded returns data: {returns.shape[0]} periods, {returns.shape[1]} factors")
    logging.info(f"Loaded max weight constraints for {len(max_weights)} factors")
    
    return returns, max_weights

def calculate_rolling_statistics(returns_df, window_size):
    """
    Pre-compute per-date forecast return vectors (rolling mean of factor returns × 8).

    Args:
        returns_df: DataFrame of factor returns
        window_size: Rolling window size (60 months) after expanding-window phase

    Returns:
        tuple: (expected_returns_dict mapping date -> 1d ndarray of forecasts, dates_list)
    """
    logging.info("Pre-computing rolling forecast returns (mean × 8)...")
    
    dates = returns_df.index
    expected_returns_dict = {}
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
            
        # Forecast = simple mean of past window, same 8x scaling as prior Step Five
        factor_means = window_data.mean(axis=0)
        expected_returns = 8 * factor_means
        
        expected_returns_dict[date] = expected_returns.values
        dates_list.append(date)
    
    logging.info(f"Pre-computed forecasts for {len(dates_list)} rebalance periods")
    return expected_returns_dict, dates_list

def run_fast_optimization():
    """
    Build monthly weights: the TOP_N factors with highest forecast return, weighted proportional to forecast.
    """
    start_time = time.time()
    
    returns_df, _max_weights = load_and_prepare_data()
    factor_names = list(returns_df.columns)
    
    next_month_date = returns_df.index[-1] + pd.DateOffset(months=1)
    logging.info(f"Extra month row (forward-looking weights): {next_month_date.strftime('%Y-%m')}")
    
    expected_returns_dict, optimization_dates = calculate_rolling_statistics(
        returns_df, WINDOW_SIZE
    )
    
    extended_dates = optimization_dates + [next_month_date]
    weights_df = pd.DataFrame(index=extended_dates, columns=factor_names, dtype=float)
    
    logging.info(f"Assigning top-{TOP_N} proportional weights each period...")
    
    for i, date in enumerate(optimization_dates):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            logging.info(f"Period {i+1}/{len(optimization_dates)}... ({elapsed:.1f}s elapsed)")
        
        mu = expected_returns_dict[date]
        optimal_weights = top_k_proportional_weights(mu, n_select=TOP_N)
        weights_df.loc[date] = optimal_weights
    
    # Extra month: same rule using last WINDOW_SIZE months of returns
    logging.info(f"Top-{TOP_N} weights for extra month {next_month_date.strftime('%Y-%m')}")
    start_idx = max(0, len(returns_df.index) - WINDOW_SIZE)
    extra_month_data = returns_df.iloc[start_idx:]
    factor_means = extra_month_data.mean(axis=0)
    expected_returns_extra = (8 * factor_means).values
    optimal_weights_extra = top_k_proportional_weights(expected_returns_extra, n_select=TOP_N)
    weights_df.loc[next_month_date] = optimal_weights_extra
    
    total_time = time.time() - start_time
    logging.info(f"Completed in {total_time:.1f} seconds ({total_time/max(len(extended_dates),1):.3f}s per row)")
    
    return weights_df, returns_df

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

    # Create sophisticated custom colormap: Light sage → Soft teal → Deep teal
    colors = ['#f8fffe', '#e8f5f3', '#d1ebe6', '#a8dadc', '#79c2d0', '#5aa9c4', '#457b9d']
    sophisticated_cmap = mcolors.LinearSegmentedColormap.from_list("sophisticated", colors, N=256)
    sophisticated_cmap.set_under('white')

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

    # Create the main heatmap
    # Ensure we have valid min/max values
    max_value = sample_weights.values.max()
    if not np.isfinite(max_value) or max_value <= 0:
        max_value = 0.5
    else:
        max_value = min(max_value, 0.5)
        
    im = ax.imshow(sample_weights.T.values, cmap=sophisticated_cmap, aspect='auto',
                   vmin=0.001, vmax=max_value)

    # Sophisticated tick formatting
    tick_positions = range(0, len(sample_weights.index), 3)  # Every 3rd year for cleaner look
    ax.set_xticks(tick_positions)
    date_labels = [sample_weights.index[i].strftime('%Y') for i in tick_positions]
    ax.set_xticklabels(date_labels, fontsize=12, fontweight='500', color='#333333')
    
    ax.set_yticks(range(len(clean_factor_names)))
    ax.set_yticklabels(clean_factor_names, fontsize=11, fontweight='400', color='#333333')

    # Professional colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cbar.set_label('Portfolio Weight (%)', rotation=270, labelpad=25, 
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

def save_results(weights_df, performance_results):
    """
    Save results to Excel files and create visualizations.
    
    Args:
        weights_df: DataFrame of optimal weights
        performance_results: Dict of performance metrics
    """
    logging.info("Saving results to Excel files...")
    
    # Save weights
    weights_output_file = 'T2_rolling_window_weights.xlsx'
    weights_df.to_excel(weights_output_file)
    logging.info(f"Weights saved to {weights_output_file}")
    
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
    logging.info("T2 FACTOR TIMING - STEP FIVE TOP THREE (TOP-2 FORECAST-RANK + PROPORTIONAL WEIGHTS)")
    logging.info("="*80)
    logging.info("Rule: each month, hold the 2 factors with highest rolling forecast return;")
    logging.info("      weight them in proportion to those forecasts (long-only adjustment if needed).")
    logging.info("="*80)
    
    try:
        weights_df, returns_df = run_fast_optimization()
        
        # Calculate performance
        performance_results = calculate_strategy_performance(weights_df, returns_df)
        
        # Save results
        save_results(weights_df, performance_results)
        
        # Display summary
        logging.info("="*80)
        logging.info("RESULTS SUMMARY")
        logging.info("="*80)
        
        for metric, value in performance_results['statistics'].items():
            logging.info(f"{metric:30s}: {value:8.2f}")
        
        logging.info("="*80)
        logging.info("STEP FIVE TOP THREE COMPLETED SUCCESSFULLY")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"Error in Step Five Top Three: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()