"""
=============================================================================
SCRIPT: Step Seven GDELT Visualize Factor Weights.py
=============================================================================

INPUT:  GDELT_rolling_window_weights.xlsx (from Step Five GDELT FAST)
OUTPUT: GDELT_latest_factor_allocation.pdf, GDELT_factor_allocation_grid.pdf

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Seven GDELT Visualize Factor Weights.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step07_visualize import run_visualize_factor_weights

if __name__ == "__main__":
    run_visualize_factor_weights(GDELT_POST5)
