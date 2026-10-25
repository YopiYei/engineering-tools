"""Concreto confinado por estribos rectangulares según Mander (1984), Cap. 3.

Referencias (Mander, J. B., *Seismic Design of Bridge Piers*, Research Report
84-2, Univ. of Canterbury, 1984):
    Ec. 3.16-3.17   coeficiente de efectividad del confinamiento ke
    Ec. 3.18-3.21   esfuerzos laterales efectivos f'lx, f'ly
    Ec. 3.29-3.31   superficie de resistencia última (Willam-Warnke, datos SW)
    §3.4.5.1        procedimiento numérico de 6 pasos para f'cc
    Ec. 3.32        deformación en f'cc: ecc = eco [1 + R (f'cc/f'co - 1)], R = 5

Unidades: MPa y mm. Convención del procedimiento de 6 pasos: tensión positiva,
como en la tesis; el resultado f'cc se entrega como valor positivo.

Se incluye además la fórmula cerrada para confinamiento igual en ambas
direcciones (Soesianawati, 1986, Ec. 5.17), que sirve como verificación.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from calculations.structural.column_section import RectangularSection
from calculations.structural.root_finding import bisect

STRAIN_RATIO_R = 5.0  # R = k2/k1 supuesto en la tesis (Ec. 3.32)


@dataclass(frozen=True)
class ConfinementResult:
    """Resultado del confinamiento de un núcleo rectangular."""

    effectiveness: float  # ke
    rho_x: float
    rho_y: float
    lateral_stress_x: float  # f'lx [MPa]
    lateral_stress_y: float  # f'ly [MPa]
    confined_strength: float  # f'cc [MPa]
    confined_strain: float  # ecc
    strength_ratio: float  # f'cc / f'co


def confinement_effectiveness(section: RectangularSection) -> float:
    """Coeficiente ke = Ae/Acc, Ec. 3.17.

    ke = [1 - sum(w'^2)/(6 bc dc)] (1 - 0.5 s'/bc)(1 - 0.5 s'/dc) / (1 - rho_cc)

    El primer factor descuenta las parábolas de concreto no confinado entre
    barras apoyadas (arqueo horizontal); los dos siguientes, el arqueo vertical
    entre estribos; el denominador excluye el área de las barras del núcleo.
    """
    bc, dc = section.core_width_mm, section.core_depth_mm
    clear_hoop_spacing = section.transverse.spacing_mm - section.transverse.bar_diameter_mm
    if clear_hoop_spacing <= 0:
        raise ValueError("El espaciamiento de estribos debe ser mayor que el diámetro del estribo.")
    plan_factor = 1.0 - section.sum_squared_clear_spacings() / (6.0 * bc * dc)
    elevation_factor = (1.0 - 0.5 * clear_hoop_spacing / bc) * (1.0 - 0.5 * clear_hoop_spacing / dc)
    effectiveness = plan_factor * elevation_factor / (1.0 - section.longitudinal_ratio_core)
    if not 0.0 < effectiveness <= 1.0:
        raise ValueError(f"Coeficiente de efectividad fuera de rango (ke = {effectiveness:.3f}): revise los estribos.")
    return effectiveness


def lateral_confining_stresses(section: RectangularSection, effectiveness: float):
    """f'lx = ke rho_x fyh y f'ly = ke rho_y fyh, Ec. 3.18 (rho_x = Asx/(s dc), rho_y = Asy/(s bc))."""
    t = section.transverse
    rho_x = t.area_x_mm2 / (t.spacing_mm * section.core_depth_mm)
    rho_y = t.area_y_mm2 / (t.spacing_mm * section.core_width_mm)
    return rho_x, rho_y, effectiveness * rho_x * t.yield_stress, effectiveness * rho_y * t.yield_stress


def _new_major_stress(estimate: float, fco: float, sigma_1: float, sigma_2: float) -> float:
    """Pasos 2 a 5 del procedimiento: recalcula sigma_3 a partir de una estimación.

    Esfuerzos con tensión positiva y sigma_3 <= sigma_2 <= sigma_1 <= 0.
    """
    sigma_oct = (sigma_1 + sigma_2 + estimate) / 3.0  # (h)
    tau_oct = sqrt((sigma_1 - sigma_2) ** 2 + (sigma_2 - estimate) ** 2 + (estimate - sigma_1) ** 2) / 3.0  # (i)
    cos_theta = (sigma_1 - sigma_oct) / (sqrt(2.0) * tau_oct)  # (j)

    s = sigma_oct / fco  # (k)
    t_meridian = 0.069232 - 0.661091 * s - 0.049350 * s**2  # (l)  Ec. 3.30a
    c_meridian = 0.122965 - 1.150502 * s - 0.315545 * s**2  # (m)  Ec. 3.30b

    d = 4.0 * (c_meridian**2 - t_meridian**2) * cos_theta**2  # (n)
    # (o) Willam-Warnke: interpolación elíptica entre los meridianos de tensión y compresión
    tau_ratio = c_meridian * (
        0.5 * d / cos_theta
        + (2.0 * t_meridian - c_meridian) * sqrt(d + 5.0 * t_meridian**2 - 4.0 * t_meridian * c_meridian)
    ) / (d + (2.0 * t_meridian - c_meridian) ** 2)
    tau_ultimate = fco * tau_ratio  # (p)

    # (q) despeje de sigma_3 en la definición de tau_oct
    return (sigma_1 + sigma_2) / 2.0 - sqrt(4.5 * tau_ultimate**2 - 0.75 * (sigma_1 - sigma_2) ** 2)


def confined_concrete_strength(unconfined_strength: float, lateral_stress_x: float, lateral_stress_y: float) -> float:
    """Resistencia confinada f'cc [MPa] por el procedimiento de 6 pasos (§3.4.5.1).

    En vez de iterar a punto fijo (paso 6) se resuelve sigma_3 = g(sigma_3) con
    bisección; la solución es la misma.
    """
    fco = unconfined_strength
    if fco <= 0 or lateral_stress_x < 0 or lateral_stress_y < 0:
        raise ValueError("f'co debe ser positivo y los esfuerzos laterales no negativos.")
    if max(lateral_stress_x, lateral_stress_y) == 0.0:
        return fco

    # Paso 1: sigma_1 es el esfuerzo lateral menor y sigma_2 el mayor (tensión positiva)
    sigma_1 = -min(lateral_stress_x, lateral_stress_y)
    sigma_2 = -max(lateral_stress_x, lateral_stress_y)

    def residual(estimate):
        return _new_major_stress(estimate, fco, sigma_1, sigma_2) - estimate

    def safe_residual(estimate):
        # Fuera del dominio de la superficie la raíz es imaginaria: se descarta ese punto
        try:
            return residual(estimate)
        except (ValueError, ZeroDivisionError):
            return float("nan")

    # Se busca el cambio de signo más cercano a -f'co (la resistencia no confinada)
    grid = np.linspace(-fco * (1.0 + 1e-9), -15.0 * fco, 600)
    values = np.array([safe_residual(x) for x in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(values[i]) and np.isfinite(values[i + 1]) and values[i] * values[i + 1] <= 0.0:
            return -bisect(residual, grid[i], grid[i + 1], xtol=1e-10)
    raise ValueError("No se encontró la resistencia confinada: revise los esfuerzos de confinamiento.")


def confined_strain(unconfined_strain: float, unconfined_strength: float, confined_strength: float, r: float = STRAIN_RATIO_R) -> float:
    """Deformación en f'cc: ecc = eco [1 + R (f'cc/f'co - 1)], Ec. 3.32."""
    return unconfined_strain * (1.0 + r * (confined_strength / unconfined_strength - 1.0))


def mander_closed_form_strength(unconfined_strength: float, lateral_stress: float) -> float:
    """Fórmula cerrada para confinamiento igual en x e y (Soesianawati 1986, Ec. 5.17).

    f'cc = f'co (-1.254 + 2.254 sqrt(1 + 7.94 f'l/f'co) - 2 f'l/f'co)
    Es una aproximación de la superficie de la tesis de Mander; se usa solo
    como verificación independiente.
    """
    ratio = lateral_stress / unconfined_strength
    return unconfined_strength * (-1.254 + 2.254 * sqrt(1.0 + 7.94 * ratio) - 2.0 * ratio)


def confine_rectangular_core(section: RectangularSection, unconfined_strength: float, unconfined_strain: float = 0.002) -> ConfinementResult:
    """Propiedades del núcleo confinado: ke, f'lx, f'ly, f'cc y ecc."""
    ke = confinement_effectiveness(section)
    rho_x, rho_y, f_lx, f_ly = lateral_confining_stresses(section, ke)
    fcc = confined_concrete_strength(unconfined_strength, f_lx, f_ly)
    return ConfinementResult(
        effectiveness=ke,
        rho_x=rho_x,
        rho_y=rho_y,
        lateral_stress_x=f_lx,
        lateral_stress_y=f_ly,
        confined_strength=fcc,
        confined_strain=confined_strain(unconfined_strain, unconfined_strength, fcc),
        strength_ratio=fcc / unconfined_strength,
    )
