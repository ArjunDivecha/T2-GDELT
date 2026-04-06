"""
=============================================================================
SCRIPT: Step Six T2 GDELT Combined Create Country Alphas from Factor alphas.py
=============================================================================

INPUT FILES:
- T2_GDELT_Combined_T60.xlsx
- T2_GDELT_Combined_Top_20_Exposure.csv
- T2 Master.xlsx

OUTPUT:
- T2_GDELT_Combined_Country_Alphas.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Six T2 GDELT Combined Create Country Alphas from Factor alphas.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step06_country_alphas import run_country_alphas

if __name__ == "__main__":
    run_country_alphas(COMBINED_POST5)
