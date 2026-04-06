"""
=============================================================================
MODULE: gdelt_track_config.py
=============================================================================

Defines file-name bundles for the **GDELT-only** and **T2+GDELT combined**
post–Step-Five pipeline (Steps 6–14 entry scripts).

The classic **T2** track keeps using the original filenames (T60.xlsx,
T2_Final_Country_Weights.xlsx, etc.) in the root Step Six / Seven / … scripts.

VERSION: 1.0
LAST UPDATED: 2026-04-02
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostStep5TrackConfig:
    """
    All disk paths for one parallel track after Step Five optimization.

    ``track_label`` is used in PDF titles and log messages.
    """

    track_label: str
    report_brand: str
    # Step 6 / 14 shared inputs
    t60_xlsx: str
    exposure_csv: str
    country_alphas_out: str
    # Step 7
    rolling_weights_xlsx: str
    latest_factor_alloc_pdf: str
    factor_grid_pdf: str
    # Step 8
    factors_master_csv: str
    final_country_weights_xlsx: str
    country_final_xlsx: str
    # Step 9
    final_portfolio_xlsx: str
    final_portfolio_pdf: str
    # Step 10
    top20_xlsx: str
    optimizer_xlsx: str
    strategy_statistics_xlsx: str
    report_pdf_prefix: str
    # Step 11 (compare — which weight files to load, in order)
    compare_weight_files: tuple[str, ...]
    compare_strategies_pdf: str
    compare_strategies_xlsx: str
    # Step 12 (multi-period summary from optimizer)
    multiperiod_summary_xlsx: str
    # Step 14
    optimized_country_weights_xlsx: str
    optimized_strategy_pdf: str
    turnover_pdf: str
    weighted_factor_exp_pdf: str
    weighted_factor_roll_pdf: str


GDELT_POST5 = PostStep5TrackConfig(
    track_label="GDELT",
    report_brand="GDELT Strategy",
    t60_xlsx="GDELT_T60.xlsx",
    exposure_csv="GDELT_Top_20_Exposure.csv",
    country_alphas_out="GDELT_Country_Alphas.xlsx",
    rolling_weights_xlsx="GDELT_rolling_window_weights.xlsx",
    latest_factor_alloc_pdf="GDELT_latest_factor_allocation.pdf",
    factor_grid_pdf="GDELT_factor_allocation_grid.pdf",
    factors_master_csv="GDELT_Factors_MasterCSV.csv",
    final_country_weights_xlsx="GDELT_Final_Country_Weights.xlsx",
    country_final_xlsx="GDELT_Country_Final.xlsx",
    final_portfolio_xlsx="GDELT_Final_Portfolio_Returns.xlsx",
    final_portfolio_pdf="GDELT_Final_Portfolio_Returns.pdf",
    top20_xlsx="GDELT Top20.xlsx",
    optimizer_xlsx="GDELT_Optimizer.xlsx",
    strategy_statistics_xlsx="GDELT_strategy_statistics.xlsx",
    report_pdf_prefix="GDELT_Strategy_Report_Comprehensive",
    compare_weight_files=(
        "GDELT_Final_Country_Weights.xlsx",
        "T2_Final_Country_Weights.xlsx",
    ),
    compare_strategies_pdf="GDELT_Compare_Strategies.pdf",
    compare_strategies_xlsx="GDELT_Compare_Return_Results.xlsx",
    multiperiod_summary_xlsx="GDELT_MultiPeriod_Factor_Summary.xlsx",
    optimized_country_weights_xlsx="GDELT_Optimized_Country_Weights.xlsx",
    optimized_strategy_pdf="GDELT_Optimized_Strategy_Analysis.pdf",
    turnover_pdf="GDELT_Turnover_Analysis.pdf",
    weighted_factor_exp_pdf="GDELT_Weighted_Average_Factor_Exposure.pdf",
    weighted_factor_roll_pdf="GDELT_Weighted_Average_Factor_Rolling_Analysis.pdf",
)

COMBINED_POST5 = PostStep5TrackConfig(
    track_label="T2_GDELT_COMBINED",
    report_brand="T2 + GDELT Combined Strategy",
    t60_xlsx="T2_GDELT_Combined_T60.xlsx",
    exposure_csv="T2_GDELT_Combined_Top_20_Exposure.csv",
    country_alphas_out="T2_GDELT_Combined_Country_Alphas.xlsx",
    rolling_weights_xlsx="T2_GDELT_Combined_rolling_window_weights.xlsx",
    latest_factor_alloc_pdf="T2_GDELT_Combined_latest_factor_allocation.pdf",
    factor_grid_pdf="T2_GDELT_Combined_factor_allocation_grid.pdf",
    factors_master_csv="Combined_T2_GDELT_Factors_MasterCSV.csv",
    final_country_weights_xlsx="T2_GDELT_Combined_Final_Country_Weights.xlsx",
    country_final_xlsx="T2_GDELT_Combined_Country_Final.xlsx",
    final_portfolio_xlsx="T2_GDELT_Combined_Final_Portfolio_Returns.xlsx",
    final_portfolio_pdf="T2_GDELT_Combined_Final_Portfolio_Returns.pdf",
    top20_xlsx="T2_GDELT_Combined_Top20.xlsx",
    optimizer_xlsx="T2_GDELT_Combined_Optimizer.xlsx",
    strategy_statistics_xlsx="T2_GDELT_Combined_strategy_statistics.xlsx",
    report_pdf_prefix="T2_GDELT_Combined_Strategy_Report_Comprehensive",
    compare_weight_files=(
        "T2_GDELT_Combined_Final_Country_Weights.xlsx",
        "T2_Final_Country_Weights.xlsx",
        "GDELT_Final_Country_Weights.xlsx",
    ),
    compare_strategies_pdf="T2_GDELT_Combined_Compare_Strategies.pdf",
    compare_strategies_xlsx="T2_GDELT_Combined_Compare_Return_Results.xlsx",
    multiperiod_summary_xlsx="T2_GDELT_Combined_MultiPeriod_Factor_Summary.xlsx",
    optimized_country_weights_xlsx="T2_GDELT_Combined_Optimized_Country_Weights.xlsx",
    optimized_strategy_pdf="T2_GDELT_Combined_Optimized_Strategy_Analysis.pdf",
    turnover_pdf="T2_GDELT_Combined_Turnover_Analysis.pdf",
    weighted_factor_exp_pdf="T2_GDELT_Combined_Weighted_Average_Factor_Exposure.pdf",
    weighted_factor_roll_pdf="T2_GDELT_Combined_Weighted_Average_Factor_Rolling_Analysis.pdf",
)

T2_MASTER_FILE = "T2 Master.xlsx"
