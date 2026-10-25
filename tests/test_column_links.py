"""Pruebas del enlace entre paneles: «Concrete Stress-Strain» y «Section Properties» alimentan a COLUMN."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural import concrete_properties as panel
from calculations.structural.concrete_link import COMPRESSION_MODELS, TENSION_MODELS, concrete_from_panel
from utils.shared_state import (
    CONCRETE_KEY,
    RECTANGLE_KEY,
    clear_rectangle,
    publish_concrete,
    publish_rectangle,
    read_concrete,
    read_rectangle,
)


# ---------------------------------------------------------------- módulos puros

@pytest.mark.parametrize("model", sorted(COMPRESSION_MODELS))
@pytest.mark.parametrize("fc", [25.0, 40.0, 70.0])
def test_concrete_from_panel_uses_the_panel_curves(model, fc):
    linked = concrete_from_panel(fc, model, "collins_mitchell")
    curves = panel.generate_curves(fc, model, "collins_mitchell", num_points=400)
    peak = int(np.argmax(curves["stress_comp"]))
    assert linked.unconfined_strength == fc
    assert linked.strain_at_peak == pytest.approx(curves["strain_comp"][peak])
    assert linked.elastic_modulus == pytest.approx(4700.0 * fc**0.5)  # todos los modelos del panel usan 4700 sqrt(f'c)
    assert linked.tensile_strength == pytest.approx(np.max(curves["stress_tens"]))
    assert linked.compression_stress.max() == pytest.approx(fc, rel=0.02)


def test_tension_model_changes_the_tensile_strength():
    values = {t: concrete_from_panel(30.0, "collins_mitchell", t).tensile_strength for t in TENSION_MODELS}
    assert len(set(round(v, 4) for v in values.values())) == len(values)
    assert all(v > 0 for v in values.values())


def test_invalid_panel_selection_is_rejected():
    with pytest.raises(ValueError):
        concrete_from_panel(30.0, "otro", "collins_mitchell")
    with pytest.raises(ValueError):
        concrete_from_panel(30.0, "collins_mitchell", "otro")


def test_shared_state_round_trip_and_unit_conversion():
    state = {}
    assert read_concrete(state) is None and read_rectangle(state) is None
    publish_concrete(35, "kent_park", "fields_bischoff", state)
    assert read_concrete(state) == {"fc": 35.0, "compression_model": "kent_park", "tension_model": "fields_bischoff"}
    for unit, factor in (("mm", 1.0), ("cm", 10.0), ("m", 1000.0), ("in", 25.4)):
        publish_rectangle(30, 50, unit, state)
        rectangle = read_rectangle(state)
        assert rectangle["width_mm"] == pytest.approx(30 * factor) and rectangle["height_mm"] == pytest.approx(50 * factor)
    clear_rectangle(state)
    assert read_rectangle(state) is None and CONCRETE_KEY in state and RECTANGLE_KEY not in state
    with pytest.raises(ValueError):
        publish_rectangle(1, 1, "ft", state)


# ---------------------------------------------------------------- páginas (AppTest)

def _page(path, **state):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / path), default_timeout=120)
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


def test_concrete_panel_publishes_fc_and_models():
    app = _page("pages/structural/concrete_curves.py")
    assert not app.exception
    shared = app.session_state[CONCRETE_KEY]
    assert shared == {"fc": 30.0, "compression_model": "collins_mitchell", "tension_model": "collins_mitchell"}


def test_section_properties_publishes_the_rectangle_and_clears_it_for_other_shapes():
    app = _page("pages/structural/section_properties.py")
    assert not app.exception
    assert app.session_state[RECTANGLE_KEY] == {"width": 10.0, "height": 20.0, "unit": "cm"}
    section_type = next(box for box in app.selectbox if box.label == "Select section type")
    section_type.select("Circle").run()
    assert not app.exception and RECTANGLE_KEY not in app.session_state


def test_column_panel_takes_concrete_and_geometry_from_the_other_panels():
    shared = dict(
        shared_concrete={"fc": 35.0, "compression_model": "kent_park", "tension_model": "fields_bischoff"},
        shared_rectangle={"width": 30.0, "height": 50.0, "unit": "cm"},
    )
    app = _page("pages/structural/column_mphi.py", **shared)
    assert not app.exception
    next(r for r in app.radio if r.label == "Origen de b y h").set_value("Tomar de «Section Properties»")
    next(r for r in app.radio if r.label == "Origen de los datos del concreto").set_value("Tomar de «Concrete Stress-Strain»")
    app.run()
    assert not app.exception and not app.error

    def value(label):
        return next(n.value for n in app.number_input if n.label.startswith(label))

    expected = concrete_from_panel(35.0, "kent_park", "fields_bischoff")
    assert value("Ancho, b") == pytest.approx(300.0) and value("Peralte, h") == pytest.approx(500.0)
    assert value("f'co") == pytest.approx(35.0)
    assert value("εco") == pytest.approx(expected.strain_at_peak)
    assert value("Ec (MPa)") == pytest.approx(expected.elastic_modulus)
    assert value("f't (MPa)") == pytest.approx(expected.tensile_strength)
    next(b for b in app.button if "Ejecutar" in b.label).click()
    app.run()
    assert not app.exception and not app.error
    result = app.session_state["column_mphi_result"]["result"]
    assert result.model.section.width_mm == pytest.approx(300.0) and result.model.section.depth_mm == pytest.approx(500.0)
    assert result.model.concrete.unconfined_strength == pytest.approx(35.0)
    assert result.model.concrete.strain_at_peak == pytest.approx(expected.strain_at_peak)
    assert result.model.core.elastic_modulus == pytest.approx(expected.elastic_modulus)
    assert result.model.cover.tensile_strength == pytest.approx(expected.tensile_strength)
    panel_curve = app.session_state["column_mphi_result"]["panel_curve"]
    assert panel_curve is not None and "Kent" in panel_curve[2]


def test_column_panel_falls_back_to_manual_values_when_nothing_is_shared():
    app = _page("pages/structural/column_mphi.py")
    next(r for r in app.radio if r.label == "Origen de b y h").set_value("Tomar de «Section Properties»")
    next(r for r in app.radio if r.label == "Origen de los datos del concreto").set_value("Tomar de «Concrete Stress-Strain»")
    app.run()
    assert not app.exception
    assert any("aún no tiene un rectángulo" in info.value for info in app.info)
    assert any("aún no publicó datos" in info.value for info in app.info)
    assert next(n.value for n in app.number_input if n.label.startswith("Ancho, b")) == pytest.approx(400.0)


def test_column_panel_shows_the_column_response_section():
    app = _page("pages/structural/column_mphi.py")
    next(b for b in app.button if "Ejecutar" in b.label).click()
    app.run()
    assert not app.exception and not app.error
    labels = [m.label for m in app.metric]
    for expected in ("Lp = Lpy + 0.06 L (mm)", "EI efectiva (MN·m²)", "Ductilidad de desplazamiento μΔ = Δu/Δy"):
        assert expected in labels
    assert any("Sección 10" in s.value or "10. De la sección a la columna" in s.value for s in app.subheader)
    stored = app.session_state["column_mphi_column"]
    assert stored["response"].column_length_mm == pytest.approx(1750.0)


def test_report_includes_the_column_response_when_requested():
    pytest.importorskip("docx")
    app = _page("pages/structural/column_mphi.py")
    next(b for b in app.button if "Ejecutar" in b.label).click()
    app.run()
    next(b for b in app.button if b.key == "report_run").click()
    app.run()
    assert not app.exception and not app.error
    report = app.session_state["column_mphi_report"]
    assert report["docx"][:2] == b"PK" and report["pdf"][:5] == b"%PDF-"
