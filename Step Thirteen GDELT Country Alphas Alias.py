"""
=============================================================================
SCRIPT NAME: Step Thirteen GDELT Country Alphas Alias.py
=============================================================================

DESCRIPTION:
    Compatibility alias that redirects execution to Step Six GDELT (Country
    Alphas from Factor Alphas). This script exists solely to support the
    historical 6-to-13 step numbering convention from the archived T2
    pipeline. It uses importlib to dynamically load and run the main()
    function from Step Six GDELT Create Country Alphas from Factor alphas.py.
    Run Step Six directly for routine use.

INPUT FILES:
    (none — this script delegates to Step Six GDELT, which reads its own
     input files)

OUTPUT FILES:
    (none — this script delegates to Step Six GDELT, which writes its own
     output files)

VERSION: 2.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - Python standard library (importlib, sys)

USAGE:
    python "Step Thirteen GDELT Country Alphas Alias.py"

NOTES:
    - Requires Step Six GDELT Create Country Alphas from Factor alphas.py
      in the same directory.
    - All documentation for inputs and outputs is on the Step Six file.
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
