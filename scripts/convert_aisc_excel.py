"""
Convert the AISC Shapes Database Excel workbook into per-family JSON files.

This script runs standalone, outside the Streamlit app. It reads the
AISC .xls workbook once and writes one JSON file per shape family into
data/aisc_sections/, plus a manifest.json describing the families. The
app only ever reads those JSON files at runtime, never the .xls.

Re-run this script whenever the source AISC workbook is updated.

Requires `xlrd` (only for this script; not a runtime dependency of the app):
    pip install xlrd

Usage:
    python scripts/convert_aisc_excel.py <path_to_AISCProp*.xls>
"""

import json
import sys
from pathlib import Path

import xlrd

SHEET_TO_FAMILY = {
    "W, M, S & HP": "w_m_s_hp",
    "C & MC": "c_mc",
    "WT, MT, & ST": "wt_mt_st",
    "Single Angles": "single_angles",
    "Double Angles": "double_angles",
    "Rect.  & Square HSS": "hss_rect",
    "Round HSS & Pipes": "hss_round",
}

FAMILY_LABELS = {
    "w_m_s_hp": "W, M, S & HP Shapes",
    "c_mc": "Channels (C, MC)",
    "wt_mt_st": "Structural Tees (WT, MT, ST)",
    "single_angles": "Single Angles (L)",
    "double_angles": "Double Angles (2L)",
    "hss_rect": "Rectangular & Square HSS",
    "hss_round": "Round HSS & Pipes",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "aisc_sections"


def convert_sheet(sheet):
    """
    Convert a single worksheet into a dict indexed by shape name.
    """

    header = [
        sheet.cell_value(0, c)
        for c in range(sheet.ncols)
    ]

    data = {}

    for r in range(1, sheet.nrows):

        shape_name = sheet.cell_value(r, 0)
        row = {}

        for c in range(1, sheet.ncols):

            key = header[c]
            value = sheet.cell_value(r, c)

            if value == "---" or value == "":
                row[key] = None
            else:
                row[key] = value

        data[shape_name] = row

    return data


def main():

    if len(sys.argv) != 2:
        print("Usage: python scripts/convert_aisc_excel.py <path_to_AISC.xls>")
        sys.exit(1)

    xls_path = Path(sys.argv[1])
    workbook = xlrd.open_workbook(str(xls_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    for sheet_name, family_key in SHEET_TO_FAMILY.items():

        sheet = workbook.sheet_by_name(sheet_name)
        data = convert_sheet(sheet)

        out_file = OUTPUT_DIR / f"{family_key}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

        manifest.append(
            {
                "key": family_key,
                "label": FAMILY_LABELS[family_key],
                "file": out_file.name,
                "count": len(data),
            }
        )

        print(f"{family_key}: {len(data)} shapes -> {out_file}")

    manifest_path = OUTPUT_DIR / "manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
