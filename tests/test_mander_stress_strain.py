import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calculations.structural.mander_stress_strain import (
    build_curve,
    confined_strength_from_lateral_stresses,
    cover_stress,
    with_ultimate_strain,
)


def test_paper_example_confined_strength():
    # Mander et al. (1988), ejemplo junto a la Fig. 4: f'co = 30, f'lx = 2.7, f'ly = 5.1 -> f'cc = 1.65 f'co
    fcc = confined_strength_from_lateral_stresses(30.0, 2.7, 5.1)
    assert fcc / 30.0 == pytest.approx(1.65, abs=0.03)


def test_curve_peaks_at_fcc_and_ecc():
    fco, fcc = 30.0, 45.0
    curve = build_curve(fco, fcc)
    assert curve.stress.max() == pytest.approx(fcc, rel=1e-3)
    assert curve.strain[np.argmax(curve.stress)] == pytest.approx(curve.confined_peak_strain, rel=0.02)
    # Ec. 5 con eco = 0.002: ecc = 0.002 [1 + 5 (1.5 - 1)] = 0.007
    assert curve.confined_peak_strain == pytest.approx(0.007)


def test_unconfined_curve_is_special_case():
    curve = build_curve(30.0)
    assert curve.confined_peak_strain == pytest.approx(0.002)
    assert curve.secant_modulus == pytest.approx(30.0 / 0.002)
    assert curve.power_r == pytest.approx(curve.elastic_modulus / (curve.elastic_modulus - curve.secant_modulus))


def test_cover_reaches_zero_at_spalling_strain():
    curve = build_curve(30.0)
    stress = cover_stress(np.array([0.002, 0.006, 0.008]), 30.0, 0.002, curve.power_r, 0.006)
    assert stress[0] == pytest.approx(30.0)
    assert stress[1] == pytest.approx(0.0, abs=1e-9)
    assert stress[2] == 0.0


def test_ultimate_strain_grows_with_confinement_steel():
    curve = build_curve(30.0, 45.0)
    light = with_ultimate_strain(curve, transverse_volumetric_ratio=0.01).ultimate_strain
    heavy = with_ultimate_strain(curve, transverse_volumetric_ratio=0.02).ultimate_strain
    assert 0.004 < light < heavy < 0.1


def test_invalid_inputs():
    with pytest.raises(ValueError):
        build_curve(30.0, 20.0)
