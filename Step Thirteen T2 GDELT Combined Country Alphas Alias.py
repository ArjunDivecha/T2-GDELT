"""
=============================================================================
SCRIPT: Step Thirteen T2 GDELT Combined Country Alphas Alias.py
=============================================================================

Same as **Step Six T2 GDELT Combined** — country alphas from combined T60 and exposure.

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Thirteen T2 GDELT Combined Country Alphas Alias.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step06_country_alphas import run_country_alphas

if __name__ == "__main__":
    run_country_alphas(COMBINED_POST5)
