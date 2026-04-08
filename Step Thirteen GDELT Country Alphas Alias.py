"""
=============================================================================
SCRIPT: Step Thirteen GDELT Country Alphas Alias.py
=============================================================================

Historical step numbering: same computation as **Step Six GDELT** (country alphas from
T60 × Top-20 exposure). Run Step Six for routine use; this script re-runs that logic
if you keep a 6→13 numbering habit from the archived T2 flow.

VERSION: 2.0 — standalone redirect
LAST UPDATED: 2026-04-08
USAGE: python "Step Thirteen GDELT Country Alphas Alias.py"
=============================================================================
"""

import importlib
import sys

if __name__ == "__main__":
    # Import and run Step Six GDELT directly
    mod_name = "Step Six GDELT Create Country Alphas from Factor alphas"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        "Step Six GDELT Create Country Alphas from Factor alphas.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
