"""
=============================================================================
SCRIPT: Step Six GDELT Create Country Alphas from Factor alphas.py
=============================================================================

INPUT FILES (run from repo root — same folder as script):
- GDELT_T60.xlsx           (from Step Four GDELT)
- GDELT_Top_20_Exposure.csv
- T2 Master.xlsx           (sheet 1MRet — country column order)

OUTPUT:
- GDELT_Country_Alphas.xlsx

PREREQUISITE: Step Five GDELT FAST (rolling weights) is not required; you need
Step Four GDELT T60 + Step Three exposure CSV.

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Six GDELT Create Country Alphas from Factor alphas.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step06_country_alphas import run_country_alphas

if __name__ == "__main__":
    run_country_alphas(GDELT_POST5)
