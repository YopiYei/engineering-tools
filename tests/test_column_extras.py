"""Pruebas de las ampliaciones de COLUMN: interacción P–M, ductilidad vs. P, variantes e informe."""

import pathlib
import re
import sys
from io import BytesIO

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural.column_analysis import AnalysisSettings, MomentCurvatureAnalysis, ideal_flexural_strength
from calculations.structural.column_examples import SOESIANAWATI_UNITS, unit_model, unit_settings
from calculations.structural.column_interaction import (
    analysis_interaction,
    code_interaction_curve,
    default_axial_range,
    interaction_dataframe,
)
from calculations.structural.column_variants import Variant, run_variants, scaled_transverse, shifted_layers, variants_dataframe

UNIT_2 = SOESIANAWATI_UNITS[2]


@pytest.fixture(scope="module")
def model_and_settings():
    return unit_model(UNIT_2), unit_settings(UNIT_2)


# ---------------------------------------------------------------- interacción P-M

def test_code_curve_end_points_match_hand_calculation(model_and_settings):
    model, _ = model_and_settings
    curve = code_interaction_curve(model)
    area_steel, gross = model.section.steel_area_mm2, model.section.gross_area_mm2
    fc, fy = model.concrete.unconfined_strength, model.steel.yield_stress
    assert curve.tension_point[0] == pytest.approx(-area_steel * fy / 1e3, rel=1e-6)
    assert curve.compression_point[0] == pytest.approx((0.85 * fc * (gross - area_steel) + area_steel * fy) / 1e3, rel=1e-6)
    # sección simétrica: sin momento en tracción y compresión puras
    assert curve.tension_point[1] == pytest.approx(0.0, abs=1e-3)
    assert curve.compression_point[1] == pytest.approx(0.0, abs=1e-3)
    assert curve.tension_point[0] < curve.balanced_point[0] < curve.compression_point[0]


def test_code_curve_agrees_with_ideal_moment_at_given_loads(model_and_settings):
    model, _ = model_and_settings
    curve = code_interaction_curve(model, number_of_points=400)
    for axial_kn in (0.0, 1000.0, 2112.0, 4000.0):
        direct = ideal_flexural_strength(model, axial_kn * 1e3) / 1e6
        assert np.interp(axial_kn, curve.axial_kn, curve.moment_knm) == pytest.approx(direct, rel=0.005)


def test_balanced_point_is_the_maximum_moment_region(model_and_settings):
    model, _ = model_and_settings
    curve = code_interaction_curve(model)
    # el momento máximo ideal se alcanza cerca del punto balanceado (±15 % de la carga axial de compresión pura)
    index = int(np.argmax(curve.moment_knm))
    assert abs(curve.axial_kn[index] - curve.balanced_point[0]) < 0.15 * curve.compression_point[0]
    assert curve.balanced_point[1] > 0.97 * curve.maximum_moment_knm


def test_analysis_interaction_matches_a_direct_run(model_and_settings):
    model, settings = model_and_settings
    calls = []
    points = analysis_interaction(model, settings, [1000e3, 2112e3], progress=lambda i, n: calls.append((i, n)))
    assert calls == [(1, 2), (2, 2)]
    direct = MomentCurvatureAnalysis(model, AnalysisSettings(axial_load_n=2112e3)).run()
    assert points[1].max_moment == pytest.approx(direct.maximum.moment, rel=1e-9)
    assert points[1].ductility == pytest.approx(direct.ductility_of(direct.states[-1].phi), rel=1e-9)
    assert points[1].termination_key == direct.termination_key


def test_ductility_decreases_with_axial_load_when_hoops_govern(model_and_settings):
    model, settings = model_and_settings
    points = analysis_interaction(model, settings, [1500e3, 2500e3, 3500e3, 4500e3])
    governed = [p for p in points if p.termination_key == "hoop_fracture" and p.ductility is not None]
    assert len(governed) >= 3
    ductility = [p.ductility for p in governed]
    assert ductility == sorted(ductility, reverse=True)


def test_interaction_sweep_reports_failures_without_stopping(model_and_settings):
    model, settings = model_and_settings
    points = analysis_interaction(model, settings, [0.0, 1e9, 1000e3])
    assert points[1].error and points[1].max_moment is None
    assert points[0].error == "" and points[2].error == ""
    table = interaction_dataframe(points, model)
    assert len(table) == 3 and table["error"].astype(bool).tolist() == [False, True, False]


def test_default_axial_range_stays_inside_the_capacity(model_and_settings):
    model, settings = model_and_settings
    low, high = default_axial_range(model, settings)
    assert low == 0.0 and 0.0 < high < 1.0e8
    MomentCurvatureAnalysis(model, AnalysisSettings(axial_load_n=high))  # no debe lanzar


# ---------------------------------------------------------------- variantes

def test_base_variant_equals_a_direct_run(model_and_settings):
    model, settings = model_and_settings
    outcome = run_variants(model, settings, [Variant("Base")])[0]
    direct = MomentCurvatureAnalysis(model, settings).run()
    assert outcome.error == ""
    assert outcome.result.maximum.moment == pytest.approx(direct.maximum.moment, rel=1e-9)
    assert outcome.model.confinement.confined_strength == pytest.approx(model.confinement.confined_strength)


def test_closer_stirrups_give_more_confinement_and_ductility(model_and_settings):
    model, settings = model_and_settings
    base, tight, loose = run_variants(
        model, settings, [Variant("Base"), Variant("s menor", stirrup_spacing_mm=55.0), Variant("s mayor", stirrup_spacing_mm=120.0)]
    )
    assert loose.model.confinement.confined_strength < base.model.confinement.confined_strength < tight.model.confinement.confined_strength
    ductility = [o.result.ductility_of(o.result.states[-1].phi) for o in (loose, base, tight)]
    assert ductility == sorted(ductility)  # todas terminan por fractura del estribo con esta carga axial
    assert {o.result.termination_key for o in (base, tight, loose)} == {"hoop_fracture"}


def test_stirrup_diameter_scales_area_and_moves_bars_to_keep_their_distance(model_and_settings):
    model, settings = model_and_settings
    base_t = model.section.transverse
    scaled = scaled_transverse(base_t, Variant("x", stirrup_diameter_mm=10.0))
    assert scaled.area_x_mm2 == pytest.approx(base_t.area_x_mm2 * (10.0 / base_t.bar_diameter_mm) ** 2)
    layers = shifted_layers(model.section, 10.0)
    assert layers[0].depth_mm == pytest.approx(model.section.layers[0].depth_mm + 2.0)
    assert layers[-1].depth_mm == pytest.approx(model.section.layers[-1].depth_mm - 2.0)
    outcome = run_variants(model, settings, [Variant("Ø10", stirrup_diameter_mm=10.0)])[0]
    assert outcome.error == "" and outcome.result is not None


def test_invalid_variant_is_reported_and_others_still_run(model_and_settings):
    model, settings = model_and_settings
    outcomes = run_variants(model, settings, [Variant("imposible", stirrup_diameter_mm=90.0), Variant("Base")])
    assert outcomes[0].error and outcomes[0].result is None
    assert outcomes[1].result is not None
    table = variants_dataframe(outcomes)
    assert table.loc[0, "error"] and table.loc[1, "Mmax_kNm"] > 0


def test_manual_confinement_is_kept_when_requested(model_and_settings):
    model, settings = model_and_settings
    outcome = run_variants(model, settings, [Variant("s menor", stirrup_spacing_mm=55.0)], recompute_confinement=False)[0]
    assert outcome.model.confinement.confined_strength == pytest.approx(model.confinement.confined_strength)


# ---------------------------------------------------------------- informe

@pytest.fixture(scope="module")
def full_report_document(model_and_settings):
    from utils.column_report import build_column_report

    model, settings = model_and_settings
    result = MomentCurvatureAnalysis(model, settings).run()
    interaction = {
        "curve": code_interaction_curve(model),
        "points": analysis_interaction(model, settings, [500e3, 2000e3, 3500e3]),
        "current_axial_kn": settings.axial_load_n / 1e3,
    }
    variants = run_variants(model, settings, [Variant("Base"), Variant("s menor", stirrup_spacing_mm=60.0)])
    arrangement = {"octagonal": True, "legs_x": 2, "legs_y": 2, "description": "Cuadrado + octagonal"}
    return build_column_report(result, arrangement, interaction, variants, title="Informe de prueba", author="Tester")


def test_report_content_lists_inputs_results_and_optional_sections(full_report_document):
    from utils.column_report import Heading, Picture, Table

    headings = [b.text for b in full_report_document.blocks if isinstance(b, Heading)]
    assert headings[:3] == ["1. Datos de entrada", "1.1 Geometría y refuerzo", "1.2 Estribos y confinamiento"]
    for expected in ("2. Resultados", "3. Interpretación", "4. Diagrama de interacción P–M y ductilidad vs. carga axial",
                     "5. Comparación de configuraciones"):
        assert expected in headings
    assert len([b for b in full_report_document.blocks if isinstance(b, Picture)]) == 5
    limit_table = next(b for b in full_report_document.blocks if isinstance(b, Table) and b.headers[0] == "Estado límite")
    assert any("Momento máximo" in row[0] for row in limit_table.rows)


def test_report_without_optional_sections_has_fewer_blocks(model_and_settings):
    from utils.column_report import Heading, build_column_report

    model, settings = model_and_settings
    document = build_column_report(MomentCurvatureAnalysis(model, settings).run())
    headings = [b.text for b in document.blocks if isinstance(b, Heading)]
    assert "4. Alcance, hipótesis y referencias" in headings and not any("interacción" in h for h in headings)


def test_docx_can_be_read_back(full_report_document):
    docx = pytest.importorskip("docx")
    from utils.column_report import write_docx

    data = write_docx(full_report_document)
    document = docx.Document(BytesIO(data))
    texts = [p.text for p in document.paragraphs]
    assert "Informe de prueba" in texts
    assert any(t.startswith("2. Resultados") for t in texts)
    assert len(document.inline_shapes) == 5
    assert len(document.tables) >= 8


def test_pdf_is_valid_and_has_several_pages(full_report_document):
    pytest.importorskip("fpdf")
    from utils.column_report import write_pdf

    data = write_pdf(full_report_document)
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    pages = re.findall(rb"/Type\s*/Page[^s]", data)
    assert len(pages) >= 5


def test_interpretation_lines_mention_the_governing_limit_state(model_and_settings):
    from utils.column_report import interpretation_lines

    model, settings = model_and_settings
    lines = interpretation_lines(MomentCurvatureAnalysis(model, settings).run())
    assert any("Mmax" in line for line in lines)
    assert any("fractura del estribo" in line for line in lines)
    assert lines[-1].startswith("**Limitaciones:**")
