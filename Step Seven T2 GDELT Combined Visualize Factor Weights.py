"""
=============================================================================
SCRIPT: Step Seven T2 GDELT Combined Visualize Factor Weights.py
=============================================================================

INPUT:  T2_GDELT_Combined_rolling_window_weights.xlsx
OUTPUT: T2_GDELT_Combined_latest_factor_allocation.pdf,
        T2_GDELT_Combined_factor_allocation_grid.pdf

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Seven T2 GDELT Combined Visualize Factor Weights.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step07_visualize import run_visualize_factor_weights

if __name__ == "__main__":
    run_visualize_factor_weights(COMBINED_POST5)
