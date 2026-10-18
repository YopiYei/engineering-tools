import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural.shrinkage_cracking import (
    australian_min_ratios,
    gilbert_crack_width,
    kgf_cm2_to_mpa,
)

# Gilbert (1992) numerical examples: L = 5 m, e = 150 mm, db = 12 mm,
# ft = 2.0 MPa, Ec = 25 000 MPa, fy = 400 MPa.
PAPER = dict(length_mm=5000, thickness_mm=150, bar_diameter_mm=12, fy_mpa=400.0,
             ft_mpa=2.0, ec_mpa=25000.0)


@pytest.mark.parametrize(
    "rho, w_expected, yielded",
    [(0.005, 0.31, False), (0.0025, 1.37, True), (0.0035, 0.53, False), (0.0063, 0.21, False)],
)
def test_gilbert_paper_examples(rho, w_expected, yielded):
    result = gilbert_crack_width(rho=rho, **PAPER)
    assert result["w_mm"] == pytest.approx(w_expected, abs=0.01)
    assert result["yielded"] is yielded


def test_gilbert_case_a_cracks():
    result = gilbert_crack_width(rho=0.005, **PAPER)
    assert result["n_cracks"] == 6
    assert result["spacing_mm"] == pytest.approx(833.3, abs=0.1)


def test_stronger_concrete_can_yield_the_steel():
    """e = 10 cm, L = 4 m, rho = 14/fy: f'c = 175 does not yield, 350 does."""
    fy = kgf_cm2_to_mpa(4200)
    rho = australian_min_ratios(4200)[0]
    weak = gilbert_crack_width(4000, 100, rho, 8, fy, fc_mpa=kgf_cm2_to_mpa(175))
    strong = gilbert_crack_width(4000, 100, rho, 8, fy, fc_mpa=kgf_cm2_to_mpa(350))
    assert not weak["yielded"] and weak["n_cracks"] == 4
    assert weak["w_mm"] == pytest.approx(0.36, abs=0.01)
    assert strong["yielded"] and strong["n_cracks"] == 1
    assert strong["w_mm"] == pytest.approx(1.18, abs=0.01)


def test_21_over_fy_never_yields_in_study_range():
    fy = kgf_cm2_to_mpa(4200)
    rho = australian_min_ratios(4200)[1]
    for fc in (175, 210, 280, 350):
        for length, thickness, db in ((4000, 50, 6), (8000, 100, 8), (20000, 150, 12)):
            r = gilbert_crack_width(length, thickness, rho, db, fy, fc_mpa=kgf_cm2_to_mpa(fc))
            assert not r["yielded"]
            assert 0.1 < r["w_mm"] < 0.4


def test_invalid_input():
    with pytest.raises(ValueError):
        gilbert_crack_width(5000, 150, 0.0, 12, 400.0, fc_mpa=25.0)
