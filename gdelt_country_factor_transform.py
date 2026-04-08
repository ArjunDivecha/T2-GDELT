"""
=============================================================================
UTILITY: gdelt_country_factor_transform.py
=============================================================================

Centralized utility for country↔factor transformation with fuzzy logic.
Used by Step Four (factor returns) and Step Eight (country weights).

This ensures consistent country selection logic across the pipeline,
eliminating performance discrepancies between factor and country portfolios.

FUZZY LOGIC:
- Top 15% of countries: full weight
- 15-25% band: linear taper
- Bottom 75%: zero weight

VERSION: 1.0
LAST UPDATED: 2026-04-08
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

# Fuzzy logic parameters
SOFT_BAND_TOP = 0.15
SOFT_BAND_CUTOFF = 0.25


def select_countries_by_factor(
    factor_exposures: pd.Series,
    method: str = 'fuzzy'
) -> pd.Series:
    """
    Given factor exposures for countries, return country weights.
    Uses fuzzy 15-25% logic by default.

    Parameters
    ----------
    factor_exposures : pd.Series
        Country → factor value (higher = better for positive exposure)
    method : str
        Selection method ('fuzzy' or 'top20')

    Returns
    -------
    pd.Series
        Country → weight (sums to 1.0 for selected countries)
    """
    # Rank countries by factor value (descending)
    ranked = factor_exposures.sort_values(ascending=False)
    n = len(ranked)
    
    if n == 0:
        return pd.Series(dtype=float)
    
    # Compute rank percentiles
    rank_pct = (np.arange(n) + 1) / n
    
    # Apply fuzzy logic
    weights = np.zeros(n)
    
    # Full weight for top band (< 15%)
    top_mask = rank_pct < SOFT_BAND_TOP
    weights[top_mask] = 1.0
    
    # Linearly decreasing weight in the grey band (15% - 25%)
    in_band = (rank_pct >= SOFT_BAND_TOP) & (rank_pct <= SOFT_BAND_CUTOFF)
    weights[in_band] = 1.0 - (rank_pct[in_band] - SOFT_BAND_TOP) / (SOFT_BAND_CUTOFF - SOFT_BAND_TOP)
    
    # Filter to non-zero weights
    nonzero_mask = weights > 0
    if not nonzero_mask.any():
        return pd.Series(dtype=float)
    
    weights = weights[nonzero_mask]
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    return pd.Series(weights, index=ranked.index[nonzero_mask])


def calculate_country_weights_from_factors(
    factor_weights: pd.Series,
    factor_exposures_df: pd.DataFrame,
    date: Optional[pd.Timestamp] = None
) -> pd.Series:
    """
    Given factor weights and factor exposures, calculate country weights.
    Used by Step Eight to translate factor portfolio → country portfolio.

    Parameters
    ----------
    factor_weights : pd.Series
        Factor → weight (can be negative for short)
    factor_exposures_df : pd.DataFrame
        Countries × Factors matrix (index=countries, columns=factors)
    date : pd.Timestamp, optional
        Date for logging/debugging

    Returns
    -------
    pd.Series
        Country → weight (net weight, can be negative)
    """
    country_weights = {}
    
    for factor, fw in factor_weights.items():
        if fw == 0:
            continue
            
        if factor not in factor_exposures_df.columns:
            continue
            
        # Get factor exposures for this factor
        factor_exposures = factor_exposures_df[factor].dropna()
        
        if factor_exposures.empty:
            continue
        
        # Select countries based on factor value
        if fw > 0:
            # Positive weight: select top countries
            country_w = select_countries_by_factor(factor_exposures)
        else:
            # Negative weight: select bottom countries (invert selection)
            country_w = select_countries_by_factor(-factor_exposures)
        
        # Apply factor weight to country weights
        for country, cw in country_w.items():
            if country not in country_weights:
                country_weights[country] = 0.0
            country_weights[country] += fw * cw
    
    return pd.Series(country_weights)


def calculate_factor_return_from_countries(
    factor_exposures: pd.Series,
    country_returns: pd.Series,
    benchmark_return: float
) -> float:
    """
    Given factor exposures and country returns, calculate factor return.
    Used by Step Four to calculate factor portfolio returns.

    Parameters
    ----------
    factor_exposures : pd.Series
        Country → factor value
    country_returns : pd.Series
        Country → return
    benchmark_return : float
        Equal-weight benchmark return for the period

    Returns
    -------
    float
        Factor net return (portfolio return - benchmark)
    """
    # Select countries based on factor value
    country_weights = select_countries_by_factor(factor_exposures)
    
    if country_weights.empty:
        return 0.0
    
    # Calculate portfolio return
    aligned_returns = country_returns.reindex(country_weights.index).fillna(0)
    portfolio_return = (country_weights * aligned_returns).sum()
    
    # Net return
    net_return = portfolio_return - benchmark_return
    
    return net_return


def calculate_all_factor_country_weights(
    factor_weights_df: pd.DataFrame,
    factor_exposures_df: pd.DataFrame
) -> Dict[pd.Timestamp, pd.Series]:
    """
    Calculate country weights for all dates from factor weights.
    Convenience function for Step Eight.

    Parameters
    ----------
    factor_weights_df : pd.DataFrame
        Dates × Factors matrix (index=dates, columns=factors)
    factor_exposures_df : pd.DataFrame
        Countries × Factors matrix (index=countries, columns=factors)

    Returns
    -------
    dict
        {date: pd.Series of country weights}
    """
    all_country_weights = {}
    
    for date in factor_weights_df.index:
        w = factor_weights_df.loc[date]
        country_w = calculate_country_weights_from_factors(w, factor_exposures_df, date)
        if not country_w.empty:
            all_country_weights[date] = country_w
    
    return all_country_weights
