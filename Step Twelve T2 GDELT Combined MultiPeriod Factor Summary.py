"""
=============================================================================
SCRIPT: Step Twelve T2 GDELT Combined MultiPeriod Factor Summary.py
=============================================================================

INPUT:  T2_GDELT_Combined_Optimizer.xlsx
OUTPUT: T2_GDELT_Combined_MultiPeriod_Factor_Summary.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Twelve T2 GDELT Combined MultiPeriod Factor Summary.py"
=============================================================================
"""

from gdelt_track_config import COMBINED_POST5
from gdelt_pipeline_step12_multiperiod import run_multiperiod_summary

if __name__ == "__main__":
    run_multiperiod_summary(COMBINED_POST5)
