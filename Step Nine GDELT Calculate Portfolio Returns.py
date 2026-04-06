"""
=============================================================================
SCRIPT: Step Nine GDELT Calculate Portfolio Returns.py
=============================================================================

INPUT:
- GDELT_Final_Country_Weights.xlsx (Step Eight)
- Portfolio_Data.xlsx (Returns + Benchmarks)

OUTPUT:
- GDELT_Final_Portfolio_Returns.xlsx
- GDELT_Final_Portfolio_Returns.pdf

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Nine GDELT Calculate Portfolio Returns.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step09_portfolio import run_portfolio_returns

if __name__ == "__main__":
    run_portfolio_returns(GDELT_POST5)
