"""
=============================================================================
SCRIPT: Step Nine T2 GDELT Combined Calculate Portfolio Returns.py
=============================================================================

INPUT:
- T2_GDELT_Combined_Final_Country_Weights.xlsx
- Portfolio_Data.xlsx

OUTPUT:
- T2_GDELT_Combined_Final_Portfolio_Returns.xlsx
- T2_GDELT_Combined_Final_Portfolio_Returns.pdf

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Nine T2 GDELT Combined Calculate Portfolio Returns.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step09_portfolio import run_portfolio_returns

if __name__ == "__main__":
    run_portfolio_returns(COMBINED_POST5)
