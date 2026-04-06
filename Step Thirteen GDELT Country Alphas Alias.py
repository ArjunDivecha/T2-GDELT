"""
=============================================================================
SCRIPT: Step Thirteen GDELT Country Alphas Alias.py
=============================================================================

Historical step numbering: same computation as **Step Six GDELT** (country alphas from
T60 × Top-20 exposure). Run Step Six for routine use; this script re-runs that logic
if you keep a 6→13 numbering habit from the archived T2 flow.

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Thirteen GDELT Country Alphas Alias.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step06_country_alphas import run_country_alphas

if __name__ == "__main__":
    run_country_alphas(GDELT_POST5)
