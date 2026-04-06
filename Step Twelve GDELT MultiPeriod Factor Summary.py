"""
=============================================================================
SCRIPT: Step Twelve GDELT MultiPeriod Factor Summary.py
=============================================================================

Lightweight Step Twelve: trailing 3/6/12-month mean factor returns from the optimizer
workbook (last date snapshot). Not a full multi-horizon forecast model.

INPUT:  GDELT_Optimizer.xlsx (Monthly_Net_Returns sheet if present)
OUTPUT: GDELT_MultiPeriod_Factor_Summary.xlsx

VERSION: 1.0  LAST UPDATED: 2026-04-02
USAGE: python "Step Twelve GDELT MultiPeriod Factor Summary.py"
=============================================================================
"""

from gdelt_track_config import GDELT_POST5
from gdelt_pipeline_step12_multiperiod import run_multiperiod_summary

if __name__ == "__main__":
    run_multiperiod_summary(GDELT_POST5)
