"""Pruebas de la respuesta de la columna: rótula plástica, fuerza lateral-desplazamiento y ductilidad.

Valores publicados:
    Mander (1984), Tabla 7.1: longitud de rótula plástica predicha con las Ec. 7.71-7.72.
    Soesianawati (1986), §8.2.8: desplazamientos máximos medidos en las Unidades 1 a 4 (contraste, no validación exacta:
    son ensayos cíclicos y el cálculo es monotónico).
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural.column_analysis import AnalysisSettings, MomentCurvatureAnalysis
from calculations.structural.column_examples import SOESIANAWATI_UNITS, unit_model, unit_settings
from calculations.structural.column_response import (
    ColumnData,
    compute_column_response,
    design_plastic_rotation,
    equivalent_plastic_hinge_length,
    first_yield_flexural_displacement,
    shear_displacement,
    yield_penetration_length,
)

LENGTH = 1750.0  # mm: altura libre de cada mitad de las columnas de ensayo (Fig. 2.3 de Soesianawati)


@pytest.fixture(scope="module")
def unit2_result():
    unit = SOESIANAWATI_UNITS[2]
    return MomentCurvatureAnalysis(unit_model(unit), unit_settings(unit)).run()


# ---------------------------------------------------------------- longitud de rótula (Tabla 7.1)

# (investigador, L/dimensión, dimensión [mm], db [mm], Lp predicha / dimensión publicada)
TABLE_7_1 = [
    ("Davey", 3.5, 500.0, 13.0, 0.44),
    ("Davey", 5.5, 500.0, 13.0, 0.56),
    ("Munro", 5.5, 500.0, 13.0, 0.56),
    ("Ng", 4.0, 250.0, 13.0, 0.70),
    ("Gill", 2.18, 550.0, 24.0, 0.41),
    ("Potangaroa", 2.0, 600.0, 24.0, 0.38),
    ("Ang", 4.0, 400.0, 16.0, 0.56),
    ("Autor, columnas huecas", 4.27, 750.0, 10.0, 0.39),
]


@pytest.mark.parametrize("name,length_ratio,size,bar_diameter,published", TABLE_7_1)
def test_hinge_length_matches_mander_table_7_1(name, length_ratio, size, bar_diameter, published):
    lp = equivalent_plastic_hinge_length(length_ratio * size, bar_diameter)
    assert lp / size == pytest.approx(published, abs=0.006)  # la tabla redondea a 2 decimales


def test_yield_penetration_and_design_rotation():
    assert yield_penetration_length(16.0) == pytest.approx(128.0)  # 32 sqrt(16)
    assert equivalent_plastic_hinge_length(1750.0, 16.0) == pytest.approx(128.0 + 105.0)
    assert equivalent_plastic_hinge_length(1750.0, 16.0, yield_penetration_mm=100.0) == pytest.approx(205.0)
    assert design_plastic_rotation(100e-6, 12e-6, 233.0) == pytest.approx(233.0 * 88e-6)
    assert design_plastic_rotation(5e-6, 12e-6, 233.0) == 0.0  # sin curvatura plástica
    with pytest.raises(ValueError):
        yield_penetration_length(0.0)


# ---------------------------------------------------------------- desplazamiento de primera fluencia (Ec. 7.75)

def test_first_yield_displacement_of_a_linear_section_is_my_l2_over_3ei():
    ei, my, length = 5.0e13, 4.0e8, 2000.0
    moments = np.linspace(0.0, my, 4001)
    assert first_yield_flexural_displacement(moments, moments / ei, length) == pytest.approx(my * length**2 / (3.0 * ei), rel=1e-6)


def test_first_yield_displacement_with_nonlinear_curvature():
    """phi = c M²: Delta = (L/My)² integral c M³ dM = c L² My²/4 (la curva real reduce Delta frente al triángulo)."""
    c, my, length = 1.0e-22, 4.0e8, 2000.0
    moments = np.linspace(0.0, my, 4000)
    expected = c * length**2 * my**2 / 4.0
    assert first_yield_flexural_displacement(moments, c * moments**2, length) == pytest.approx(expected, rel=1e-5)
    triangle = (c * my**2) * length**2 / 3.0
    assert expected < triangle


# ---------------------------------------------------------------- cortante (Ec. 7.83)

STIFFNESS = dict(k_uncracked=1.6e9, k_hinge=1.4e8, k_cracked=7.6e7)


def _shear(moment, max_moment, cracking=1.0e8, length=1750.0, hinge=400.0):
    return shear_displacement(moment, max_moment, length, cracking, hinge_length_mm=hinge, **STIFFNESS)


def test_shear_displacement_cases_a_b_c_are_continuous_and_ordered():
    cracking, length, hinge = 1.0e8, 1750.0, 400.0
    moment = 1.0e8  # M <= Mmax en todos los casos
    # (a) sin fisurar
    assert _shear(5e7, 5e7) == pytest.approx(5e7 / STIFFNESS["k_uncracked"])
    # frontera a/b: Mmax = Mcr
    assert _shear(moment, cracking) == pytest.approx(moment / STIFFNESS["k_uncracked"])
    assert _shear(moment, cracking * 1.000001) == pytest.approx(_shear(moment, cracking), rel=1e-4)
    # frontera b/c: Mmax = Mcr / (1 - Lh/L)
    boundary = cracking / (1.0 - hinge / length)
    assert _shear(moment, boundary * 0.999999) == pytest.approx(_shear(moment, boundary * 1.000001), rel=1e-4)
    # con más fisuración el mismo momento produce más deformación por cortante
    values = [_shear(moment, m) for m in (cracking, boundary, 1.5e8, 3.0e8, 4.0e8)]
    assert values == sorted(values)
    assert _shear(0.0, 3e8) == 0.0
    # sin Mcr (nunca se fisura) rige la rigidez sin fisurar
    assert shear_displacement(moment, 9e8, length, None, hinge_length_mm=hinge, **STIFFNESS) == pytest.approx(moment / STIFFNESS["k_uncracked"])


# ---------------------------------------------------------------- respuesta de la columna

def test_response_components_follow_the_equations(unit2_result):
    result = unit2_result
    response = compute_column_response(result, ColumnData(LENGTH))
    lpy = yield_penetration_length(16.0)
    my, phi_y = result.yield_moment_first, result.yield_curvature_first
    index = next(i for i, st in enumerate(result.states) if st.phi > 3.0 * phi_y)  # un paso claramente plástico
    state = result.states[index]
    plastic_zone = LENGTH * (1.0 - my / state.max_moment)  # Ec. 7.85
    rotation = (state.phi - phi_y * state.moment / my) * (plastic_zone / 3.0 + lpy)  # Ec. 7.87
    assert response.plastic_rotation[index] == pytest.approx(rotation, rel=1e-9)
    assert response.plastic_displacement[index] == pytest.approx(rotation * (LENGTH - 0.25 * plastic_zone), rel=1e-9)  # Ec. 7.88
    assert response.elastic_displacement[index] == pytest.approx(state.moment * LENGTH**2 / (3.0 * response.effective_stiffness), rel=1e-9)  # Ec. 7.77
    assert response.displacement[index] == pytest.approx(response.elastic_displacement[index] + response.plastic_displacement[index], rel=1e-12)
    assert response.shear_force[index] == pytest.approx(state.moment / LENGTH)  # Ec. 7.82


def test_yield_definitions_and_ductility_at_first_yield(unit2_result):
    result = unit2_result
    response = compute_column_response(result, ColumnData(LENGTH))
    assert response.first_yield_displacement == pytest.approx(response.first_yield_flexural_displacement)  # sin cortante
    assert response.yield_displacement == pytest.approx(response.first_yield_displacement * result.ideal_moment / result.yield_moment_first)  # Ec. 7.91
    # En la primera fluencia mu = Delta'y / Delta_y = My / Mi
    record = next(r for r in result.limit_states if r.key == "first_yield")
    index = response.state_index(record.state)
    assert response.ductility[index] == pytest.approx(result.yield_moment_first / result.ideal_moment, rel=1e-6)
    # Antes de la fluencia no hay parte plástica
    assert np.all(response.plastic_displacement[: index] == 0.0)
    # EI efectiva a primera fluencia: My L²/(3 Delta'y)
    assert response.effective_stiffness == pytest.approx(result.yield_moment_first * LENGTH**2 / (3.0 * response.first_yield_flexural_displacement))


def test_displacement_grows_with_curvature_and_scales_with_length(unit2_result):
    response = compute_column_response(unit2_result, ColumnData(LENGTH))
    assert np.all(np.diff(response.displacement) > 0.0)
    assert np.all(response.elastic_displacement >= 0) and np.all(response.plastic_displacement >= 0)
    short = compute_column_response(unit2_result, ColumnData(1000.0, yield_penetration_mm=100.0))
    long = compute_column_response(unit2_result, ColumnData(2000.0, yield_penetration_mm=100.0))
    # la flexión elástica y el desplazamiento de fluencia crecen con L²
    assert long.yield_displacement / short.yield_displacement == pytest.approx(4.0, rel=1e-9)
    assert long.shear_force.max() / short.shear_force.max() == pytest.approx(0.5, rel=1e-9)  # V = M/L


def test_shear_deformation_increases_displacements(unit2_result):
    without = compute_column_response(unit2_result, ColumnData(LENGTH))
    with_shear = compute_column_response(unit2_result, ColumnData(LENGTH, include_shear=True))
    assert np.all(without.shear_displacement == 0.0)
    assert with_shear.yield_displacement > without.yield_displacement
    assert np.all(with_shear.displacement >= without.displacement)
    assert with_shear.shear_stiffnesses["uncracked"] > with_shear.shear_stiffnesses["hinge"] > with_shear.shear_stiffnesses["cracked"] > 0.0
    # la zona de rótula tiene estribos más juntos que fuera de ella, por eso es más rígida a cortante que la zona fisurada exterior
    assert with_shear.shear_displacement.max() > 0.0


def test_ultimate_state_selection_and_design_rotation(unit2_result):
    response = compute_column_response(unit2_result, ColumnData(LENGTH))
    final = response.ultimate()
    assert final["index"] == len(unit2_result.states) - 1
    spalling = response.ultimate("cover_spalling")
    assert spalling["displacement"] < final["displacement"]  # el descascaramiento ocurre antes que la fractura del estribo
    assert final["design_plastic_rotation"] == pytest.approx(
        response.plastic_hinge_length_mm * (final["curvature"] - unit2_result.yield_curvature))
    with pytest.raises(ValueError):
        response.ultimate("core_ultimate")  # no se alcanzó (no se pidió)


def test_design_ultimate_criterion_is_more_conservative_than_energy_balance():
    unit = SOESIANAWATI_UNITS[2]
    model = unit_model(unit)
    energy = MomentCurvatureAnalysis(model, unit_settings(unit)).run()
    design = MomentCurvatureAnalysis(model, unit_settings(unit, use_design_ultimate=True)).run()
    assert design.termination_key in ("design_core_strain", "design_moment_drop")
    assert design.states[-1].phi < energy.states[-1].phi
    # εmax = εspall + 0.04 (K - 1) en el borde del núcleo (Ec. 7.70)
    k = model.core.peak_stress / model.concrete.unconfined_strength
    last = design.states[-1]
    edge = last.eps0 + last.phi * model.section.core_depth_mm / 2.0
    if design.termination_key == "design_core_strain":
        assert edge == pytest.approx(model.concrete.spall_strain + 0.04 * (k - 1.0), rel=1e-6)


def test_first_cracking_precedes_first_yield(unit2_result):
    records = {r.key: r for r in unit2_result.limit_states}
    assert records["first_cracking"].reached
    assert records["first_cracking"].moment < records["first_yield"].moment
    assert records["first_cracking"].curvature < records["first_yield"].curvature


@pytest.mark.parametrize("number,measured_mm", [(2, 115.0), (3, 68.0), (4, 46.0)])
def test_ultimate_displacement_is_close_to_the_measured_maximum(number, measured_mm):
    """Contraste con los desplazamientos máximos medidos (Soesianawati §8.2.8) en las unidades con P = 0.3 f'c Ag.

    Los ensayos son cíclicos y el cálculo monotónico; la comparación solo comprueba el orden de magnitud (±25 %).
    La Unidad 1 queda fuera: con carga axial baja rige εsu del acero y el cálculo da el doble del medido (91 mm).
    """
    unit = SOESIANAWATI_UNITS[number]
    result = MomentCurvatureAnalysis(unit_model(unit), unit_settings(unit)).run()
    response = compute_column_response(result, ColumnData(LENGTH, include_shear=True))
    assert response.ultimate()["displacement"] == pytest.approx(measured_mm, rel=0.25)


def test_invalid_inputs_are_rejected(unit2_result):
    for kwargs in ({"length_mm": 0.0}, {"length_mm": -5.0}, {"length_mm": 1000.0, "form_factor": 0.0},
                   {"length_mm": 1000.0, "hinge_length_mm": -1.0}, {"length_mm": 1000.0, "yield_penetration_mm": 0.0}):
        with pytest.raises(ValueError):
            ColumnData(**kwargs)
    unit = SOESIANAWATI_UNITS[1]
    compression_controlled = MomentCurvatureAnalysis(unit_model(unit), AnalysisSettings(axial_load_n=5000e3)).run()
    with pytest.raises(ValueError, match="fluencia"):
        compute_column_response(compression_controlled, ColumnData(LENGTH))


def test_response_table_columns_and_units(unit2_result):
    response = compute_column_response(unit2_result, ColumnData(LENGTH, include_shear=True))
    table = response.to_dataframe()
    assert {"lateral_force_kN", "displacement_mm", "displacement_ductility", "plastic_rotation_rad", "drift_ratio"} <= set(table.columns)
    assert table["lateral_force_kN"].max() == pytest.approx(unit2_result.maximum.moment / LENGTH / 1e3, rel=1e-6)
    assert table["drift_ratio"].iloc[-1] == pytest.approx(response.displacement[-1] / LENGTH)


def test_report_with_column_response_adds_the_section_and_a_figure(unit2_result):
    from utils.column_report import Heading, Picture, Table, build_column_report, write_docx, write_pdf

    response = compute_column_response(unit2_result, ColumnData(LENGTH, include_shear=True))
    document = build_column_report(unit2_result, column={"response": response, "ultimate_key": None})
    headings = [b.text for b in document.blocks if isinstance(b, Heading)]
    assert "4. De la sección a la columna: rótula plástica y curva fuerza–desplazamiento" in headings
    assert len([b for b in document.blocks if isinstance(b, Picture)]) == 4
    table = next(b for b in document.blocks if isinstance(b, Table) and b.headers[0] == "Parámetro" and any("Lpy" in r[0] for r in b.rows))
    assert any("Ductilidad de desplazamiento" in row[0] for row in table.rows)
    docx = pytest.importorskip("docx")
    assert write_docx(document)[:2] == b"PK" and write_pdf(document)[:5] == b"%PDF-"
