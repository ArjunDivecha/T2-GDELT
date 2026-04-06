"""
=============================================================================
SCRIPT: Step Eight GDELT Write Country Weights.py
=============================================================================

INPUT:
- GDELT_rolling_window_weights.xlsx
- GDELT_Factors_MasterCSV.csv
- T2 Master.xlsx (country order for Country_Final)

OUTPUT:
- GDELT_Final_Country_Weights.xlsx
- GDELT_Country_Final.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Eight GDELT Write Country Weights.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step08_country_weights import run_country_weights

if __name__ == "__main__":
    run_country_weights(GDELT_POST5)
