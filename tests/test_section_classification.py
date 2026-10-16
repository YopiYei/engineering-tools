import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural.section_classification import (
    COMPACT,
    NONCOMPACT,
    SLENDER,
    aisc_shape_elements,
    classify_section,
    i_shape_elements,
    hss_round_elements,
)


def _family(key):
    with open(ROOT / "data" / "aisc_sections" / f"{key}.json", encoding="utf-8") as f:
        return json.load(f)


def test_tarea2_welded_beam_noncompact_flange_compact_web():
    # W29x12 built-up from the assignment: 12x1/2 flanges, 28x3/8 web, A36
    els = i_shape_elements(12, 0.5, 28, 0.375, 29000, 36, built_up=True)
    flange, web = els
    assert flange["lam"] == pytest.approx(12.0)
    assert flange["lam_p"] == pytest.approx(10.79, abs=0.01)
    assert flange["lam_r"] == pytest.approx(21.93, abs=0.01)
    assert flange["state"] == NONCOMPACT
    assert web["lam_p"] == pytest.approx(106.7, abs=0.1)
    assert web["state"] == COMPACT
    assert classify_section(els)["classification"] == NONCOMPACT


def test_aisc_known_w_shapes_fy50():
    # AISC Manual: W14x99 has a noncompact flange at Fy = 50 ksi; W14x61 is compact.
    shapes = _family("w_m_s_hp")
    r99 = classify_section(aisc_shape_elements("w_m_s_hp", shapes["W14X99"], 29000, 50))
    r61 = classify_section(aisc_shape_elements("w_m_s_hp", shapes["W14X61"], 29000, 50))
    assert r99["classification"] == NONCOMPACT
    assert r99["governing_element"] == "Flange"
    assert r61["classification"] == COMPACT


def test_known_noncompact_flange_shape_fy50():
    # AISC Table 1-1: W21x48 and W14x99 etc. Shapes with bf/2tf > 9.15 are noncompact flange.
    shapes = _family("w_m_s_hp")
    names = [n for n, p in shapes.items() if p["bf/(2*tf)"] > 9.15 and p["bf/(2*tf)"] < 24]
    assert names
    r = classify_section(aisc_shape_elements("w_m_s_hp", shapes[names[0]], 29000, 50))
    assert r["classification"] == NONCOMPACT


def test_weak_axis_ignores_web():
    els = i_shape_elements(12, 0.5, 28, 0.375, 29000, 36, axis="weak")
    assert [e["element"] for e in els] == ["Flange"]


def test_round_hss_slender():
    els = hss_round_elements(20, 0.1, 29000, 50)
    assert classify_section(els)["classification"] == SLENDER


def test_every_database_shape_classifies():
    for key in ["w_m_s_hp", "c_mc", "wt_mt_st", "single_angles", "double_angles", "hss_rect", "hss_round"]:
        for name, props in _family(key).items():
            for axis in ("strong", "weak"):
                els = aisc_shape_elements(key, props, 29000, 50, axis=axis)
                assert classify_section(els)["classification"] in (COMPACT, NONCOMPACT, SLENDER), name


# ---------------------------------------------------------
# Compression, Table B4.1a
# ---------------------------------------------------------

from calculations.structural.section_classification import (
    NONSLENDER,
    aisc_shape_compression,
    i_shape_compression,
)


def test_compression_tarea2_section():
    # flange: 0.64*sqrt(kc E / Fy), kc = 4/sqrt(74.67) = 0.463 -> lam_r = 0.64*sqrt(0.463*29000/36) = 12.36
    els = i_shape_compression(12, 0.5, 28, 0.375, 29000, 36, built_up=True)
    flange, web = els
    assert flange["lam_r"] == pytest.approx(12.36, abs=0.02)
    assert flange["state"] == NONSLENDER          # 12.0 <= 12.36
    assert web["lam_r"] == pytest.approx(1.49 * (29000 / 36) ** 0.5, rel=1e-6)
    # web: h/tw = 74.7 > 1.49*sqrt(E/Fy) = 42.3 -> slender web governs
    assert web["state"] == SLENDER
    r = classify_section(els)
    assert r["classification"] == SLENDER
    assert r["governing_element"] == "Web"


def test_compression_known_slender_w_shapes_fy50():
    # AISC Manual Table 1-1 footnote: W14x38 (and others) have slender flanges for compression at Fy = 50
    shapes = _family("w_m_s_hp")
    r = classify_section(aisc_shape_compression("w_m_s_hp", shapes["W14X38"], 29000, 50))
    assert r["classification"] == SLENDER
    assert r["governing_element"] == "Web"   # h/tw = 39.6 > 1.49*sqrt(E/Fy) = 35.9
    r = classify_section(aisc_shape_compression("w_m_s_hp", shapes["W14X99"], 29000, 50))
    assert r["classification"] == NONSLENDER


def test_compression_every_database_shape():
    for key in ["w_m_s_hp", "c_mc", "wt_mt_st", "single_angles", "double_angles", "hss_rect", "hss_round"]:
        for name, props in _family(key).items():
            els = aisc_shape_compression(key, props, 29000, 50)
            assert classify_section(els)["classification"] in (NONSLENDER, SLENDER), name
