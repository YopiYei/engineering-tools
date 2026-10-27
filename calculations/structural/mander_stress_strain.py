"""Curva esfuerzo-deformación de Mander para concreto no confinado y confinado.

Referencia: Mander, J. B., Priestley, M. J. N., & Park, R. (1988). Theoretical
stress-strain model for confined concrete. J. Struct. Eng., ASCE, 114(8), 1804-1826.

    Ec. 3   fc = f'cc * x * r / (r - 1 + x^r)            x = ec / ecc
    Ec. 5   ecc = eco * [1 + 5 (f'cc/f'co - 1)]
    Ec. 6   r = Ec / (Ec - Esec)
    Ec. 8   Esec = f'cc / ecc
    Ec. 64  110 rho_st + 0.017 sqrt(f'co) = Ucc + Usc    (deformación última, primer estribo roto)

El concreto no confinado es el caso particular f'cc = f'co, ecc = eco. El recubrimiento
(concreto no confinado de una sección) sigue la misma curva hasta 2 eco y luego baja
en línea recta hasta cero en la deformación de descascaramiento esp.

La resistencia confinada f'cc la entrega la superficie de falla de cinco parámetros
(Willam-Warnke), ya implementada en `column_confinement`; aquí solo se arma la curva.

Unidades: MPa y mm/mm (deformaciones adimensionales).
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from calculations.structural.column_confinement import confine_rectangular_core, confined_concrete_strength, confined_strain
from calculations.structural.column_section import RectangularSection
from calculations.structural.root_finding import bisect

UNCONFINED_PEAK_STRAIN = 0.002  # eco, valor usual (Mander et al. 1988, tras Ec. 5)
SPALLING_STRAIN = 0.006  # esp, valor usual del concreto de recubrimiento (Fig. 1 y 3)
STRAIN_RATIO_R = 5.0  # R en la Ec. 5 (Richart et al. 1928)
TRANSVERSE_STEEL_ENERGY = 110.0  # Usf [MJ/m³], energía de la barra transversal hasta su rotura (Ec. 62)
UNCONFINED_ENERGY_FACTOR = 0.017  # Uco = 0.017 sqrt(f'co) [MJ/m³] (Ec. 63)

# Módulo tangente del concreto Ec [MPa] como función de f'co [MPa].
ELASTIC_MODULUS_MODELS = {
    "mander": ("Mander (1988): 5000·√f'co", lambda fco: 5000.0 * sqrt(fco)),
    "aci": ("ACI 318: 4700·√f'co", lambda fco: 4700.0 * sqrt(fco)),
}


@dataclass(frozen=True)
class ManderCurve:
    """Curva esfuerzo-deformación de Mander con sus parámetros característicos."""

    strain: np.ndarray  # ec [mm/mm]
    stress: np.ndarray  # fc [MPa]
    unconfined_strength: float  # f'co
    confined_strength: float  # f'cc
    unconfined_peak_strain: float  # eco
    confined_peak_strain: float  # ecc
    elastic_modulus: float  # Ec
    secant_modulus: float  # Esec = f'cc / ecc
    power_r: float  # r = Ec / (Ec - Esec)
    ultimate_strain: float | None = None  # ecu, solo si se calculó (Ec. 64)


# =========================================================================
# Parámetros del modelo
# =========================================================================

def elastic_modulus(unconfined_strength: float, model: str = "mander") -> float:
    """Módulo tangente Ec [MPa] según el modelo elegido de `ELASTIC_MODULUS_MODELS`."""
    if model not in ELASTIC_MODULUS_MODELS:
        raise ValueError(f"Modelo de Ec desconocido: {model}")
    return ELASTIC_MODULUS_MODELS[model][1](unconfined_strength)


def secant_modulus(confined_strength: float, confined_peak_strain: float) -> float:
    """Esec = f'cc / ecc, Ec. 8: pendiente de la secante al punto de máximo esfuerzo."""
    return confined_strength / confined_peak_strain


def power_r(elastic_modulus_value: float, secant_modulus_value: float) -> float:
    """r = Ec / (Ec - Esec), Ec. 6."""
    if secant_modulus_value >= elastic_modulus_value:
        raise ValueError(
            "Esec debe ser menor que Ec: revise f'co, eco o el modelo de Ec "
            "(con eco = 0.002 y f'co alto, Ec = f'co/eco ya no se cumple)."
        )
    return elastic_modulus_value / (elastic_modulus_value - secant_modulus_value)


# =========================================================================
# Curva
# =========================================================================

def mander_stress(
    strain,
    confined_strength: float,
    confined_peak_strain: float,
    power_r_value: float,
):
    """Esfuerzo de compresión fc [MPa] de la Ec. 3 para una deformación (o arreglo) ec >= 0."""
    strain_ratio = np.maximum(np.asarray(strain, dtype=float), 0.0) / confined_peak_strain
    return confined_strength * strain_ratio * power_r_value / (power_r_value - 1.0 + strain_ratio**power_r_value)


def cover_stress(
    strain,
    unconfined_strength: float,
    unconfined_peak_strain: float,
    power_r_value: float,
    spalling_strain: float = SPALLING_STRAIN,
):
    """Esfuerzo del concreto no confinado (recubrimiento) [MPa].

    Ec. 3 con f'co y eco hasta 2 eco; después, recta hasta cero en esp.
    """
    strain = np.asarray(strain, dtype=float)
    stress = mander_stress(strain, unconfined_strength, unconfined_peak_strain, power_r_value)

    double_peak_strain = 2.0 * unconfined_peak_strain
    stress_at_double_peak = float(mander_stress(double_peak_strain, unconfined_strength, unconfined_peak_strain, power_r_value))
    descending_line = stress_at_double_peak * (spalling_strain - strain) / (spalling_strain - double_peak_strain)

    return np.where(strain > double_peak_strain, np.maximum(descending_line, 0.0), stress)


def build_curve(
    unconfined_strength: float,
    confined_strength: float | None = None,
    unconfined_peak_strain: float = UNCONFINED_PEAK_STRAIN,
    elastic_modulus_model: str = "mander",
    max_strain: float | None = None,
    num_points: int = 400,
) -> ManderCurve:
    """Curva de Mander para f'co y f'cc dados.

    Con `confined_strength = None` (o igual a f'co) resulta la curva del concreto
    no confinado: ecc = eco y la Ec. 3 se evalúa con f'co.

    max_strain: deformación final de la curva; por defecto 5 ecc.
    """
    if unconfined_strength <= 0 or unconfined_peak_strain <= 0:
        raise ValueError("f'co y eco deben ser positivos.")
    if confined_strength is None:
        confined_strength = unconfined_strength
    if confined_strength < unconfined_strength:
        raise ValueError("f'cc no puede ser menor que f'co.")

    confined_peak_strain = confined_strain(unconfined_peak_strain, unconfined_strength, confined_strength)
    ec = elastic_modulus(unconfined_strength, elastic_modulus_model)
    esec = secant_modulus(confined_strength, confined_peak_strain)
    r = power_r(ec, esec)

    last_strain = 5.0 * confined_peak_strain if max_strain is None else max_strain
    strain = np.linspace(0.0, last_strain, num_points)
    stress = mander_stress(strain, confined_strength, confined_peak_strain, r)

    return ManderCurve(
        strain=strain,
        stress=stress,
        unconfined_strength=unconfined_strength,
        confined_strength=confined_strength,
        unconfined_peak_strain=unconfined_peak_strain,
        confined_peak_strain=confined_peak_strain,
        elastic_modulus=ec,
        secant_modulus=esec,
        power_r=r,
    )


# =========================================================================
# Resistencia confinada
# =========================================================================

def confined_strength_from_lateral_stresses(unconfined_strength: float, lateral_stress_x: float, lateral_stress_y: float) -> float:
    """f'cc [MPa] de los esfuerzos laterales efectivos f'lx y f'ly (superficie de 5 parámetros, Fig. 4)."""
    return confined_concrete_strength(unconfined_strength, lateral_stress_x, lateral_stress_y)


def confined_strength_from_section(section: RectangularSection, unconfined_strength: float, unconfined_peak_strain: float = UNCONFINED_PEAK_STRAIN):
    """Confinamiento de un núcleo rectangular: devuelve el `ConfinementResult` (ke, f'lx, f'ly, f'cc, ecc)."""
    return confine_rectangular_core(section, unconfined_strength, unconfined_peak_strain)


# =========================================================================
# Deformación última (primer estribo roto), Ec. 64
# =========================================================================

def _area_under_curve(strain: np.ndarray, stress: np.ndarray) -> float:
    """Área bajo la curva [MJ/m³] por trapecios (MPa × mm/mm = MJ/m³)."""
    return float(np.sum(0.5 * (stress[1:] + stress[:-1]) * np.diff(strain)))


def ultimate_strain(
    curve: ManderCurve,
    transverse_volumetric_ratio: float,
    longitudinal_volumetric_ratio: float = 0.0,
    longitudinal_yield_stress: float = 0.0,
    longitudinal_elastic_modulus: float = 200000.0,
    transverse_steel_energy: float = TRANSVERSE_STEEL_ENERGY,
) -> float:
    """Deformación ecu en la que se rompe el primer estribo, por balance de energía (Ec. 64).

        Usf * rho_st = Ucc + Usc - Uco

    Ucc  = área bajo la curva confinada hasta ecu (Ec. 3)
    Usc  = rho_cc * área bajo la curva del acero longitudinal (elastoplástico) hasta ecu
    Uco  = 0.017 sqrt(f'co)   (Ec. 63)

    Parámetros
    ----------
    transverse_volumetric_ratio:   rho_st, volumen de estribos / volumen del núcleo.
    longitudinal_volumetric_ratio: rho_cc, acero longitudinal / área del núcleo (0 = se desprecia Usc).
    longitudinal_yield_stress:     fy longitudinal [MPa], solo si rho_cc > 0.
    """
    if transverse_volumetric_ratio <= 0:
        raise ValueError("rho_st debe ser positivo para calcular la deformación última.")

    energy_supply = transverse_steel_energy * transverse_volumetric_ratio + UNCONFINED_ENERGY_FACTOR * sqrt(curve.unconfined_strength)

    def energy_demand(strain_value: float) -> float:
        strains = np.linspace(0.0, strain_value, 600)
        concrete_energy = _area_under_curve(strains, mander_stress(strains, curve.confined_strength, curve.confined_peak_strain, curve.power_r))
        steel_stress = np.minimum(longitudinal_elastic_modulus * strains, longitudinal_yield_stress)
        steel_energy = longitudinal_volumetric_ratio * _area_under_curve(strains, steel_stress)
        return concrete_energy + steel_energy

    def balance(strain_value: float) -> float:
        return energy_demand(strain_value) - energy_supply

    lower_strain, upper_strain = 1e-5, 0.2
    if balance(upper_strain) < 0.0:
        raise ValueError("La energía de los estribos excede la capacidad de la curva hasta ec = 0.2: revise rho_st.")
    return bisect(balance, lower_strain, upper_strain, xtol=1e-8)


def with_ultimate_strain(curve: ManderCurve, **kwargs) -> ManderCurve:
    """Copia de `curve` con `ultimate_strain` calculado y la curva extendida hasta ecu."""
    ecu = ultimate_strain(curve, **kwargs)
    strain = np.linspace(0.0, max(ecu, curve.strain[-1]), len(curve.strain))
    stress = mander_stress(strain, curve.confined_strength, curve.confined_peak_strain, curve.power_r)
    return ManderCurve(
        strain=strain,
        stress=stress,
        unconfined_strength=curve.unconfined_strength,
        confined_strength=curve.confined_strength,
        unconfined_peak_strain=curve.unconfined_peak_strain,
        confined_peak_strain=curve.confined_peak_strain,
        elastic_modulus=curve.elastic_modulus,
        secant_modulus=curve.secant_modulus,
        power_r=curve.power_r,
        ultimate_strain=ecu,
    )
