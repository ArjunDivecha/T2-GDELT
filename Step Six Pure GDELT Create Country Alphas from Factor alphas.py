"""
=============================================================================
SCRIPT: Step Six Pure GDELT Create Country Alphas from Factor alphas.py
=============================================================================

Pure GDELT track only (not T2+GDELT combined). Same logic as
``Step Six GDELT Create Country Alphas from Factor alphas.py``; this filename
matches the Combined Step Six script for easier pairing.

INPUT FILES:
- GDELT_T60.xlsx
- GDELT_Top_20_Exposure.csv
- T2 Master.xlsx

OUTPUT:
- GDELT_Country_Alphas.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-03
USAGE: python "Step Six Pure GDELT Create Country Alphas from Factor alphas.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step06_country_alphas import run_country_alphas

if __name__ == "__main__":
    run_country_alphas(GDELT_POST5)
