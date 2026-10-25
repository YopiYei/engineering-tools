"""Pruebas del análisis momento-curvatura tipo COLUMN (Mander, 1984).

Valores publicados usados como referencia:
    Soesianawati (1986), Tabla 5.1: f'cc y ecc de Mander para las Unidades 1 a 4.
    Soesianawati (1986), Tabla 8.3: Mi y Mmax/Mi del análisis monotónico de Mander.
    Mander (1984), p. 65: ejemplo numérico de resistencia confinada (lectura de gráfico).
"""

import pathlib
import sys
from dataclasses import replace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural import moment_curvature as reference_module
from calculations.structural.column_analysis import (
    AnalysisSettings,
    ConcreteProperties,
    MomentCurvatureAnalysis,
    build_column_model,
    ideal_flexural_strength,
)
from calculations.structural.column_confinement import (
    ConfinementResult,
    confined_concrete_strength,
    confinement_effectiveness,
    mander_closed_form_strength,
)
from calculations.structural.column_examples import (
    HD16_STEEL,
    SOESIANAWATI_UNITS,
    unit_model,
    unit_section,
    unit_settings,
)
from calculations.structural.column_materials import PopovicsConcrete, ReinforcingSteel
from calculations.structural.column_section import BarLayer, RectangularSection, TransverseReinforcement, bar_area, build_mesh

UNITS = sorted(SOESIANAWATI_UNITS)


@pytest.fixture(scope="module")
def unit_results():
    """Un análisis completo por unidad de ensayo (se reutiliza entre pruebas)."""
    results = {}
    for number in UNITS:
        unit = SOESIANAWATI_UNITS[number]
        results[number] = MomentCurvatureAnalysis(unit_model(unit), unit_settings(unit)).run()
    return results


# ---------------------------------------------------------------- materiales

def test_concrete_peak_and_initial_modulus():
    concrete = PopovicsConcrete(40.0, 0.002, 31622.0)
    stress, tangent = concrete.stress_and_tangent(np.array([0.002, 1e-9]))
    assert stress[0] == pytest.approx(40.0)
    assert tangent[0] == pytest.approx(0.0, abs=1e-6)  # pendiente nula en el pico
    assert tangent[1] == pytest.approx(31622.0, rel=1e-3)  # Ec en el origen


def test_cover_descends_linearly_to_zero_at_spalling():
    cover = PopovicsConcrete(30.0, 0.002, 27386.0, spall_strain=0.008)
    f2 = cover.stress_and_tangent(np.array(0.004))[0]
    f_mid = cover.stress_and_tangent(np.array(0.006))[0]
    assert f_mid == pytest.approx(f2 / 2.0)  # punto medio de la recta de 0.004 a 0.008
    assert cover.stress_and_tangent(np.array(0.008))[0] == pytest.approx(0.0)
    assert cover.stress_and_tangent(np.array(0.012))[0] == 0.0


def test_concrete_tension_is_linear_then_cracks():
    concrete = PopovicsConcrete(30.0, 0.002, 27386.0, tensile_strength=3.0)
    cracking = 3.0 / 27386.0
    assert concrete.stress_and_tangent(np.array(-0.5 * cracking))[0] == pytest.approx(-1.5)
    assert concrete.stress_and_tangent(np.array(-2.0 * cracking))[0] == 0.0
    already_cracked = concrete.stress_and_tangent(np.array(-0.5 * cracking), cracked=np.array(True))[0]
    assert already_cracked == 0.0


@pytest.mark.parametrize("strain", [-0.003, 0.0005, 0.0015, 0.0024, 0.0035, 0.0055])
def test_concrete_tangent_matches_finite_difference(strain):
    concrete = PopovicsConcrete(45.0, 0.0025, 31000.0, spall_strain=0.009)
    h = 1e-8
    numeric = (concrete.stress_and_tangent(strain + h)[0] - concrete.stress_and_tangent(strain - h)[0]) / (2 * h)
    assert concrete.stress_and_tangent(strain)[1] == pytest.approx(numeric, rel=1e-4, abs=1.0)


def test_steel_key_points_and_symmetry():
    steel = HD16_STEEL
    for strain, expected in [(steel.yield_strain, steel.yield_stress), (steel.hardening_strain, steel.yield_stress),
                             (steel.ultimate_strain, steel.ultimate_stress)]:
        assert float(steel.stress_and_tangent(strain)[0]) == pytest.approx(expected)
        assert float(steel.stress_and_tangent(-strain)[0]) == pytest.approx(-expected)
    # Ec. 2.10 y 2.9a: el módulo tangente en el inicio del endurecimiento es Esh
    assert float(steel.stress_and_tangent(steel.hardening_strain * 1.0000001)[1]) == pytest.approx(steel.hardening_modulus, rel=1e-3)


@pytest.mark.parametrize("strain", [0.001, 0.0015, 0.005, 0.02, 0.06])
def test_steel_tangent_matches_finite_difference(strain):
    h = 1e-8
    steel = HD16_STEEL
    numeric = (steel.stress_and_tangent(strain + h)[0] - steel.stress_and_tangent(strain - h)[0]) / (2 * h)
    assert float(steel.stress_and_tangent(strain)[1]) == pytest.approx(float(numeric), rel=1e-3, abs=1.0)


def test_invalid_materials_are_rejected():
    with pytest.raises(ValueError):
        PopovicsConcrete(100.0, 0.002, 50000.0)  # Ec = Esec: r no definido
    with pytest.raises(ValueError):
        PopovicsConcrete(30.0, 0.002, 27000.0, spall_strain=0.003)  # espall <= 2 eco
    with pytest.raises(ValueError):
        ReinforcingSteel(400.0, 300.0, 200000.0, 0.01, 0.1, 5000.0)  # fsu < fy
    with pytest.raises(ValueError):
        ReinforcingSteel(400.0, 600.0, 200000.0, 0.001, 0.1, 5000.0)  # esh < ey


# ---------------------------------------------------------------- confinamiento

def test_no_confinement_returns_unconfined_strength():
    assert confined_concrete_strength(30.0, 0.0, 0.0) == 30.0


def test_unconfined_strength_is_a_fixed_point_of_the_surface():
    # Con confinamiento casi nulo el resultado tiende a f'co (comprueba la calibración de la superficie)
    assert confined_concrete_strength(30.0, 1e-4, 1e-4) == pytest.approx(30.0, abs=0.01)


def test_mander_thesis_numerical_example():
    """Mander (1984), p. 65: f'co = 30, f'ly = 2.7 y f'lx = 5.1 MPa; la tesis lee 1.65 f'co = 49.5 MPa del gráfico."""
    fcc = confined_concrete_strength(30.0, 5.1, 2.7)
    assert fcc == pytest.approx(49.5, rel=0.03)


@pytest.mark.parametrize("lateral", [0.5, 1.0, 2.5, 5.0])
def test_equal_confinement_matches_closed_form(lateral):
    """La fórmula cerrada (Soesianawati Ec. 5.17) es la solución de la superficie para f'lx = f'ly."""
    assert confined_concrete_strength(35.0, lateral, lateral) == pytest.approx(mander_closed_form_strength(35.0, lateral), rel=2e-3)


def test_more_confinement_gives_more_strength():
    assert confined_concrete_strength(30.0, 4.0, 4.0) > confined_concrete_strength(30.0, 2.0, 4.0) > confined_concrete_strength(30.0, 2.0, 2.0)


@pytest.mark.parametrize("number", UNITS)
def test_confinement_matches_soesianawati_table_5_1(number):
    unit = SOESIANAWATI_UNITS[number]
    confinement = unit_model(unit).confinement
    assert confinement.confined_strength == pytest.approx(unit.published_fcc_mpa, rel=0.025)
    assert confinement.confined_strain == pytest.approx(unit.published_ecc, rel=0.10)


def test_closer_stirrups_increase_effectiveness():
    unit = SOESIANAWATI_UNITS[1]
    section = unit_section(unit)
    tighter = replace(section, transverse=replace(section.transverse, spacing_mm=60.0))
    assert 0.0 < confinement_effectiveness(section) < confinement_effectiveness(tighter) < 1.0


# ---------------------------------------------------------------- sección y malla

def test_mesh_conserves_concrete_and_steel_area():
    section = unit_section(SOESIANAWATI_UNITS[1])
    mesh = build_mesh(section, 60)
    concrete_area = mesh.core_area.sum() + mesh.cover_area.sum()
    assert concrete_area + mesh.bar_area.sum() == pytest.approx(section.gross_area_mm2, rel=1e-9)
    assert section.number_of_bars == 12
    assert mesh.bar_area.sum() == pytest.approx(section.steel_area_mm2)
    assert np.all(mesh.core_area >= 0) and np.all(mesh.cover_area >= 0)


def test_invalid_sections_are_rejected():
    stirrup = TransverseReinforcement.from_legs(10.0, 100.0, 300.0, 2, 2)
    with pytest.raises(ValueError):
        RectangularSection.with_perimeter_bars(300, 300, 160, 3, 3, 20, stirrup)  # sin núcleo
    with pytest.raises(ValueError):
        RectangularSection.with_perimeter_bars(300, 300, 40, 12, 3, 25, stirrup)  # barras traslapadas
    with pytest.raises(ValueError):
        RectangularSection.with_perimeter_bars(300, 300, 40, 1, 3, 20, stirrup)  # menos de 2 barras por cara
    with pytest.raises(ValueError):
        TransverseReinforcement.from_legs(10.0, 100.0, 300.0, 1, 2)


# ---------------------------------------------------------------- análisis

@pytest.mark.parametrize("number", UNITS)
def test_every_step_satisfies_axial_equilibrium(unit_results, number):
    """Equilibrio interno = carga axial aplicada en todos los incrementos, dentro de la tolerancia."""
    result = unit_results[number]
    tolerance = result.settings.equilibrium_tolerance * result.model.concrete.unconfined_strength * result.model.section.gross_area_mm2
    for state in result.states:
        assert state.converged
        assert abs(state.axial_force - result.settings.axial_load_n) <= tolerance * 1.0001
        assert state.moment >= -1e-6


def test_initial_state_is_pure_axial_compression():
    unit = SOESIANAWATI_UNITS[1]
    analysis = MomentCurvatureAnalysis(unit_model(unit), unit_settings(unit))
    state = analysis.initial_state()
    assert state.phi == 0.0
    assert state.moment == pytest.approx(0.0, abs=1.0)  # la sección es simétrica
    assert state.eps0 > 0.0


def test_initial_stiffness_matches_transformed_section():
    """A curvatura pequeña M/phi = EI de la sección transformada (concreto + acero, sin fisurar)."""
    unit = SOESIANAWATI_UNITS[1]
    model = unit_model(unit)
    mesh = model.mesh
    ec, es = model.core.elastic_modulus, model.steel.elastic_modulus
    expected = ec * np.sum((mesh.core_area + mesh.cover_area) * mesh.y**2) + es * np.sum(mesh.bar_area * mesh.bar_y**2)
    analysis = MomentCurvatureAnalysis(model, unit_settings(unit))
    start = analysis.initial_state()
    phi = 2e-8  # 1/mm: deformaciones << eco, rango lineal
    state = analysis.advance(start, phi)
    # el concreto con P > 0 es algo más rígido que Ec en el origen: se compara la rigidez de flexión
    assert state.moment / phi == pytest.approx(expected, rel=0.01)


def test_matches_existing_moment_curvature_module():
    """Verificación cruzada con `moment_curvature.py` (P = 0, concreto no confinado, mismos materiales).

    Los dos códigos son independientes; el existente no resta el concreto desplazado por las barras,
    por eso se le entrega el acero neto (fs - fc) para que ambos modelos sean equivalentes.
    """
    fc, fy, es = 28.0, 420.0, 200000.0
    steel = ReinforcingSteel(fy, 1.5 * fy, es, 0.01, 0.12, es / 60.0)
    stirrup = TransverseReinforcement.from_legs(10.0, 100.0, 420.0, 2, 2)
    section = RectangularSection.with_perimeter_bars(300.0, 500.0, 40.0, 3, 2, 25.0, stirrup)
    concrete = ConcreteProperties(fc, 0.002, 0.006, tensile_strength=0.0, elastic_modulus=4700.0 * fc**0.5)
    unconfined = ConfinementResult(1.0, 0.0, 0.0, 0.0, 0.0, fc, 0.002, 1.0)  # núcleo = concreto no confinado
    model = build_column_model(section, concrete, steel, confinement=unconfined, number_of_strips=100)
    analysis = MomentCurvatureAnalysis(model, AnalysisSettings(axial_load_n=0.0))

    concrete_fn = lambda e: reference_module.concrete_mander_unconfined(e, fc, 0.002, 0.006)
    net_steel_fn = lambda e: reference_module.steel_mander(e, fy, es) - concrete_fn(e)
    depths = list(500.0 / 2.0 - section.bar_layers()[0])
    areas = list(section.bar_layers()[2])
    reference = reference_module.moment_curvature(
        300.0, 500.0, list(zip(depths, areas)), concrete_fn, net_steel_fn, 0.0035, num_steps=200, num_fibres=400
    )

    state = analysis.initial_state()
    checked = 0
    for phi in np.linspace(2e-6, 0.8 * reference["phi"].max(), 12):
        state = analysis.advance(state, float(phi))
        expected = np.interp(phi, reference["phi"], reference["moment"])
        assert state.moment == pytest.approx(expected, rel=0.01)
        checked += 1
    assert checked == 12


@pytest.mark.parametrize("number", UNITS)
def test_ideal_moment_matches_soesianawati_table_8_3(number):
    """Mi con el método de la norma y la resistencia de cilindros del día del ensayo (nota (a) de la tabla)."""
    unit = SOESIANAWATI_UNITS[number]
    mi = ideal_flexural_strength(unit_model(unit), unit.axial_load_n, concrete_strength=unit.cylinder_strength_test_mpa) / 1e6
    assert mi == pytest.approx(unit.ideal_moment_knm, rel=0.02)


@pytest.mark.parametrize("number", UNITS)
def test_maximum_moment_matches_soesianawati_table_8_3(unit_results, number):
    """Mmax del análisis monotónico de Mander = (Mmax/Mi publicado) x Mi publicado, con razón redondeada a 2 decimales."""
    unit = SOESIANAWATI_UNITS[number]
    expected = unit.overstrength_mander * unit.ideal_moment_knm
    assert unit_results[number].maximum.moment / 1e6 == pytest.approx(expected, rel=0.02)


def test_results_do_not_depend_on_mesh_or_increment():
    unit = SOESIANAWATI_UNITS[1]
    reference = MomentCurvatureAnalysis(unit_model(unit, number_of_strips=240), unit_settings(unit, increment_factor=0.25)).run()
    for strips, factor in ((60, 1.0), (120, 0.5)):
        result = MomentCurvatureAnalysis(unit_model(unit, number_of_strips=strips), unit_settings(unit, increment_factor=factor)).run()
        assert result.maximum.moment == pytest.approx(reference.maximum.moment, rel=0.005)
        assert result.yield_curvature_first == pytest.approx(reference.yield_curvature_first, rel=0.01)
        assert result.states[-1].phi == pytest.approx(reference.states[-1].phi, rel=0.02)


def test_axial_load_effects():
    """Sin carga axial, con compresión moderada y con compresión alta."""
    unit = SOESIANAWATI_UNITS[1]
    model = unit_model(unit)
    runs = {p: MomentCurvatureAnalysis(model, AnalysisSettings(axial_load_n=p)).run() for p in (0.0, 744e3, 2112e3, 3500e3)}
    moments = [runs[p].maximum.moment for p in sorted(runs)]
    assert moments == sorted(moments)  # hasta la zona balanceada, la compresión aumenta el momento
    # Con carga alta rige la fractura del estribo y la ductilidad disponible disminuye al aumentar P.
    # (Con P = 0 y P moderada el límite es esu del acero, que no es comparable entre ambos.)
    governed_by_hoops = [p for p in sorted(runs) if runs[p].termination_key == "hoop_fracture"]
    assert governed_by_hoops == [2112e3, 3500e3]
    ductility = [runs[p].ductility_of(runs[p].states[-1].phi) for p in governed_by_hoops]
    assert ductility[0] > ductility[1]
    assert all(r.warnings == [] for r in runs.values())


def test_very_high_axial_load_reports_that_steel_never_yields():
    unit = SOESIANAWATI_UNITS[1]
    result = MomentCurvatureAnalysis(unit_model(unit), AnalysisSettings(axial_load_n=5000e3)).run()
    assert result.yield_curvature is None
    assert any("fluencia" in message for message in result.warnings)


def test_limit_states_are_ordered_and_defined(unit_results):
    result = unit_results[2]
    records = {r.key: r for r in result.limit_states}
    order = ["cover_peak", "first_yield", "code_strain", "max_moment", "cover_spalling", "hoop_fracture"]
    curvatures = [records[k].curvature for k in order]
    assert all(records[k].reached for k in order)
    assert curvatures == sorted(curvatures)
    assert result.yield_curvature > 0 and result.yield_curvature_first > 0
    assert result.termination_key == "hoop_fracture"
    # la deformación que define cada evento se cumple en el estado refinado
    first_yield = records["first_yield"].state
    y_tension = result.model.mesh.bar_y[0]
    assert -(first_yield.eps0 + first_yield.phi * y_tension) == pytest.approx(result.model.steel.yield_strain, rel=1e-6)


def test_hoop_fracture_terminates_when_energy_balance_is_reached(unit_results):
    result = unit_results[2]
    analysis = MomentCurvatureAnalysis(result.model, result.settings)
    assert result.states[-1].core_work == pytest.approx(analysis.hoop_energy_capacity, rel=1e-6)


def test_user_ultimate_strain_terminates_the_analysis():
    unit = SOESIANAWATI_UNITS[1]
    ecu = 0.012
    result = MomentCurvatureAnalysis(unit_model(unit), unit_settings(unit, core_ultimate_strain=ecu)).run()
    assert result.termination_key == "core_ultimate"
    last = result.states[-1]
    assert last.eps0 + last.phi * result.model.section.core_depth_mm / 2.0 == pytest.approx(ecu, rel=1e-6)


def test_dataframe_columns_and_units():
    unit = SOESIANAWATI_UNITS[1]
    result = MomentCurvatureAnalysis(unit_model(unit), unit_settings(unit)).run()
    table = result.to_dataframe()
    assert {"curvature_urad_per_mm", "moment_kNm", "neutral_axis_depth_mm", "converged", "limit_state"} <= set(table.columns)
    assert table["moment_kNm"].max() == pytest.approx(result.maximum.moment / 1e6)
    assert table["limit_state"].str.contains("Primera fluencia").any()


# ---------------------------------------------------------------- entradas inválidas

def test_axial_load_above_capacity_is_rejected():
    unit = SOESIANAWATI_UNITS[1]
    with pytest.raises(ValueError, match="carga axial"):
        MomentCurvatureAnalysis(unit_model(unit), AnalysisSettings(axial_load_n=1e9))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"increment_schedule": "otro"},
        {"increment_schedule": "uniform"},
        {"increment_factor": 0.0},
        {"equilibrium_tolerance": -1.0},
        {"moment_drop_ratio": 1.5},
        {"work_fraction_core": 0.0},
        {"core_ultimate_strain": -0.01},
    ],
)
def test_invalid_analysis_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AnalysisSettings(**kwargs)


# ---------------------------------------------------------------- búsqueda de raíces

def test_bisect_accepts_descending_interval_and_rejects_no_sign_change():
    from calculations.structural.root_finding import bisect, maximize_bounded

    assert bisect(lambda x: x**3 - 2.0, 3.0, 0.0, xtol=1e-12) == pytest.approx(2.0 ** (1.0 / 3.0), abs=1e-10)
    assert bisect(lambda x: x**3 - 2.0, 0.0, 3.0, xtol=1e-12) == pytest.approx(2.0 ** (1.0 / 3.0), abs=1e-10)
    with pytest.raises(ValueError):
        bisect(lambda x: x**2 + 1.0, -1.0, 1.0)
    assert maximize_bounded(lambda x: -(x - 0.7) ** 2, 0.0, 2.0, xtol=1e-10) == pytest.approx(0.7, abs=1e-7)


# ---------------------------------------------------------------- capas de barras generales

def _asymmetric_model():
    """Sección tipo viga: 2 barras de 16 mm arriba y 3 de 25 mm abajo (diámetros y número distintos)."""
    fy, es = 420.0, 200000.0
    steel = ReinforcingSteel(fy, 1.5 * fy, es, 0.01, 0.12, es / 60.0)
    stirrup = TransverseReinforcement.from_legs(10.0, 100.0, 420.0, 2, 2)
    layers = (BarLayer(58.0, 2, 16.0), BarLayer(437.5, 3, 25.0))
    section = RectangularSection(300.0, 500.0, 40.0, layers, stirrup)
    return build_column_model(section, ConcreteProperties(28.0, 0.002, 0.008), steel)


def test_general_layers_reproduce_the_perimeter_constructor():
    unit = SOESIANAWATI_UNITS[1]
    perimeter = unit_section(unit)
    manual = RectangularSection(
        perimeter.width_mm, perimeter.depth_mm, perimeter.clear_cover_mm,
        tuple(reversed(tuple(BarLayer(l.depth_mm, l.count, l.diameter_mm) for l in perimeter.layers))),  # orden de entrada no importa
        perimeter.transverse,
    )
    assert manual.layers == perimeter.layers
    assert confinement_effectiveness(manual) == pytest.approx(confinement_effectiveness(perimeter), rel=1e-12)
    assert [layer.count for layer in perimeter.layers] == [4, 2, 2, 4]


def test_asymmetric_section_ideal_moment_matches_hand_calculation():
    """P = 0: Mi ~ T (d - a/2) con T = As fy y a = T/(0.85 f'c b); el acero comprimido suma poco."""
    model = _asymmetric_model()
    steel_force = 3 * bar_area(25.0) * 420.0
    a = steel_force / (0.85 * 28.0 * 300.0)
    hand = steel_force * (437.5 - a / 2.0) / 1e6
    assert ideal_flexural_strength(model, 0.0) / 1e6 == pytest.approx(hand, rel=0.02)


def test_asymmetric_section_initial_stiffness_includes_axial_flexure_coupling():
    """Con P = 0 y refuerzo asimétrico, M/phi = EI - EZ²/EA (el eje neutro se desplaza del centroide)."""
    model = _asymmetric_model()
    mesh = model.mesh
    ec, es = model.core.elastic_modulus, model.steel.elastic_modulus
    concrete_area = mesh.core_area + mesh.cover_area
    ea = ec * concrete_area.sum() + es * mesh.bar_area.sum()
    ez = ec * (concrete_area * mesh.y).sum() + es * (mesh.bar_area * mesh.bar_y).sum()
    ei = ec * (concrete_area * mesh.y**2).sum() + es * (mesh.bar_area * mesh.bar_y**2).sum()
    analysis = MomentCurvatureAnalysis(model, AnalysisSettings(axial_load_n=0.0))
    phi = 2e-8
    state = analysis.advance(analysis.initial_state(), phi)
    assert state.moment / phi == pytest.approx(ei - ez**2 / ea, rel=1e-3)
    assert state.axial_force == pytest.approx(0.0, abs=analysis.tolerance * 1.001)


def test_asymmetric_section_runs_to_a_limit_state():
    result = MomentCurvatureAnalysis(_asymmetric_model(), AnalysisSettings(axial_load_n=0.0)).run()
    assert all(state.converged for state in result.states)
    assert result.maximum.moment > result.ideal_moment  # endurecimiento del acero
    assert result.yield_curvature is not None


@pytest.mark.parametrize(
    "layers",
    [
        (BarLayer(5.0, 2, 16.0), BarLayer(300.0, 2, 16.0)),  # capa superior fuera del estribo
        (BarLayer(60.0, 2, 16.0), BarLayer(65.0, 2, 16.0)),  # capas que se traslapan
        (BarLayer(60.0, 2, 16.0),),  # una sola capa
        (BarLayer(60.0, 12, 25.0), BarLayer(300.0, 2, 25.0)),  # barras traslapadas en la capa
    ],
)
def test_invalid_bar_layers_are_rejected(layers):
    stirrup = TransverseReinforcement.from_legs(10.0, 100.0, 300.0, 2, 2)
    with pytest.raises(ValueError):
        RectangularSection(300.0, 360.0, 40.0, layers, stirrup)


def test_invalid_bar_layer_values_are_rejected():
    with pytest.raises(ValueError):
        BarLayer(50.0, 1, 16.0)
    with pytest.raises(ValueError):
        BarLayer(50.0, 2, 0.0)


# ---------------------------------------------------------------- dibujo de la sección

def test_section_figure_draws_every_bar_stirrup_and_dimension():
    from visualization.column_plots import section_figure

    unit_section_1 = unit_section(SOESIANAWATI_UNITS[1])
    figure = section_figure(unit_section_1, octagonal=True)
    circles = [shape for shape in figure.layout.shapes if shape.type == "circle"]
    assert len(circles) == unit_section_1.number_of_bars == 12
    # las barras se dibujan con su diámetro real, dentro del contorno de la sección
    radius = 16.0 / 2.0
    assert all(abs((c.x1 - c.x0) / 2.0 - radius) < 1e-9 for c in circles)
    assert all(0.0 < c.x0 and c.x1 < 400.0 and 0.0 < c.y0 and c.y1 < 400.0 for c in circles)
    texts = " ".join(a.text for a in figure.layout.annotations if a.text)
    for expected in ("b = 400", "h = 400", "b″ = 367", "d″ = 367", "4 Ø16", "2 Ø16", "recubrimiento libre r = 13"):
        assert expected in texts

    stirrup = TransverseReinforcement.from_legs(10.0, 100.0, 420.0, 4, 3)
    beam = RectangularSection(300.0, 500.0, 40.0, (BarLayer(58.0, 2, 16.0), BarLayer(437.5, 3, 25.0)), stirrup)
    drawing = section_figure(beam, legs_x=4, legs_y=3)
    assert len([shape for shape in drawing.layout.shapes if shape.type == "circle"]) == 5
    legend = [trace.name for trace in drawing.data if trace.name]
    assert {"Núcleo confinado", "Barras longitudinales"} <= set(legend)
