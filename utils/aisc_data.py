"""
Loader for the AISC Shapes Database (pre-converted JSON files).

The app never reads the source .xls workbook at runtime; it only reads
the JSON files under data/aisc_sections/, generated once by
scripts/convert_aisc_excel.py.
"""

import json
from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "aisc_sections"


@st.cache_data
def load_manifest():
    """
    Load the list of available AISC shape families.
    """

    with open(DATA_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_family(family_key):
    """
    Load all shapes for a given family, indexed by shape name.
    """

    with open(DATA_DIR / f"{family_key}.json", encoding="utf-8") as f:
        return json.load(f)
