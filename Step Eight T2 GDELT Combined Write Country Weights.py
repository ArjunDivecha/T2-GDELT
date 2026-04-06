"""
=============================================================================
SCRIPT: Step Eight T2 GDELT Combined Write Country Weights.py
=============================================================================

INPUT:
- T2_GDELT_Combined_rolling_window_weights.xlsx
- Combined_T2_GDELT_Factors_MasterCSV.csv
- T2 Master.xlsx

OUTPUT:
- T2_GDELT_Combined_Final_Country_Weights.xlsx
- T2_GDELT_Combined_Country_Final.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Eight T2 GDELT Combined Write Country Weights.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step08_country_weights import run_country_weights

if __name__ == "__main__":
    run_country_weights(COMBINED_POST5)
