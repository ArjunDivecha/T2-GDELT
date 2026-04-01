"""
GDELT workbook column labels → T2 Master / Normalized CSV country codes.
Aligned with Step One country_names where names differ.
"""

from __future__ import annotations

from typing import Dict

GDELT_TO_T2_COUNTRY: Dict[str, str] = {
    "U.S. NASDAQ": "NASDAQ",
    "China A": "ChinaA",
    "China H": "ChinaH",
}


def map_country_label(label: object) -> str:
    if label is None or (isinstance(label, float) and str(label) == "nan"):
        return ""
    s = str(label).strip()
    return GDELT_TO_T2_COUNTRY.get(s, s)
