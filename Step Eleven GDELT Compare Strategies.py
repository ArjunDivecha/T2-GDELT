"""
=============================================================================
SCRIPT: Step Eleven GDELT Compare Strategies.py
=============================================================================

Compares country-weight portfolios (from Step Eight outputs) against the equal-weight
benchmark. Missing optional files (e.g. T2_Final_Country_Weights.xlsx) are skipped.

INPUT:
- Portfolio_Data.xlsx
- GDELT_Final_Country_Weights.xlsx (required)
- T2_Final_Country_Weights.xlsx (optional, for side-by-side)

OUTPUT:
- GDELT_Compare_Strategies.pdf
- GDELT_Compare_Return_Results.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Eleven GDELT Compare Strategies.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step11_compare import run_compare_strategies

if __name__ == "__main__":
    run_compare_strategies(GDELT_POST5)
