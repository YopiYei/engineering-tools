import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural.moment_curvature import (
    PARK_BEAMS,
    PSI_TO_MPA,
    concrete_mander_unconfined,
    concrete_park_paulay,
    curvature_ductility,
    park_beam_curve,
    steel_mander,
)

# Peak M/bd2 [lb/in2] of Fig. 6.6 (Park & Paulay), read from the figure
# (approx. 1370, 1300, 1200, 920, 870, 460) and reproduced by this code.
PEAKS_PSI = {1: 1366, 2: 1315, 3: 1197, 4: 916, 5: 865, 6: 469, 7: 466}


@pytest.mark.parametrize("beam", sorted(PARK_BEAMS))
def test_park_fig_6_6_peaks(beam):
    curve = park_beam_curve(beam)
    peak = curve["m_norm"].max() / PSI_TO_MPA
    assert peak == pytest.approx(PEAKS_PSI[beam], rel=0.01)


def test_more_tension_steel_means_more_moment_less_curvature():
    c3 = park_beam_curve(3)
    c7 = park_beam_curve(7)
    assert c3["m_norm"].max() > 2 * c7["m_norm"].max()
    assert c3["phi_d"][-1] < c7["phi_d"][-1]


def test_compression_steel_increases_curvature_capacity():
    assert park_beam_curve(6)["phi_d"][-1] > park_beam_curve(7)["phi_d"][-1]


def test_mander_hardening_raises_moment_of_underreinforced_beams():
    park = park_beam_curve(7, "park", "elastoplastic")["m_norm"].max()
    mander = park_beam_curve(7, "mander", "mander")["m_norm"].max()
    assert 1.05 < mander / park < 1.20


def test_concrete_models_peak_at_e0():
    assert float(concrete_park_paulay(0.002, 27.6)) == pytest.approx(27.6)
    assert float(concrete_mander_unconfined(0.002, 27.6)) == pytest.approx(27.6, rel=1e-3)
    assert float(concrete_mander_unconfined(0.007, 27.6)) == 0.0


def test_mander_steel_plateau_and_hardening():
    fy, es = 275.8, 206843.0
    assert float(steel_mander(0.005, fy, es)) == pytest.approx(fy)
    assert float(steel_mander(0.06, fy, es)) > fy
    assert float(steel_mander(-0.06, fy, es)) < -fy


def test_ductility_is_finite():
    curve = park_beam_curve(7)
    _, mu = curvature_ductility(curve["phi"], curve["moment"])
    assert 5 < mu < 20
