"""
=============================================================================
SCRIPT: Step Eleven T2 GDELT Combined Compare Strategies.py
=============================================================================

INPUT: Portfolio_Data.xlsx + any of:
- T2_GDELT_Combined_Final_Country_Weights.xlsx
- T2_Final_Country_Weights.xlsx
- GDELT_Final_Country_Weights.xlsx
(missing files skipped)

OUTPUT:
- T2_GDELT_Combined_Compare_Strategies.pdf
- T2_GDELT_Combined_Compare_Return_Results.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Eleven T2 GDELT Combined Compare Strategies.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step11_compare import run_compare_strategies

if __name__ == "__main__":
    run_compare_strategies(COMBINED_POST5)
