"""De la sección a la columna: rótula plástica, curva fuerza lateral-desplazamiento y ductilidad.

Extiende el diagrama momento-curvatura del extremo crítico de una columna en voladizo
(longitud L desde la sección crítica hasta el punto de inflexión, carga lateral V = M/L) a una
curva fuerza-desplazamiento aproximada, como en la sección 7.4.4 de Mander (1984), con
carga axial constante. Es una reconstrucción de esa teoría; no incluye la carga cíclica.

Ecuaciones de Mander (1984, Research Report 84-2):
    7.67   theta_p = Lp (phi_u - phi_y)                      rotación plástica de diseño
    7.71   Lpy = 32 sqrt(db)  [mm]                           penetración de fluencia
    7.72   Lp = Lpy + 0.06 L                                 longitud equivalente de rótula
    7.74   Delta = Delta_e + Delta_s + Delta_p
    7.75   Delta = integral de phi(l) l dl                   (momentos de área de la curvatura)
    7.76   EI_eff = My L² / (3 Delta_y_flexión)              rigidez efectiva a primera fluencia
    7.77   Delta_e = M L² / (3 EI_eff)
    7.78   Kve = 0.4 Ec Aq / f                               cortante sin fisurar
    7.81   Kv = 1 / (1/(Es Av d/s) + 4/(Ec Aq))              cortante fisurado (fisuras a 45°)
    7.82   M = V L
    7.83   Delta_s según tres zonas (sin fisurar, fisurada y rótula)
    7.84   phi_p = phi - phi'_y M/My
    7.85   Lpc = L - L My/Mmax
    7.87   theta_p = (phi - phi'_y M/My)(Lpc/3 + Lpy)
    7.88   Delta_p = theta_p (L - 0.25 Lpc)
    7.90   mu = Delta / Delta_y
    7.91   Delta_y = Delta'_y Mi/My,  Delta'_y = Delta_e + Delta_s a primera fluencia (7.92)

Nota sobre 7.83c: el texto impreso usa Kve en la tercera zona, pero esa zona es la fisurada fuera de
la rótula (Fig. 7.14), de rigidez Kvc; se sigue la figura y la descripción del texto.

Unidades: N, mm, MPa. Momento en N·mm, curvatura en 1/mm, fuerza lateral V en N.
"""

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from calculations.structural.column_analysis import (
    CODE_STEEL_MODULUS,
    MomentCurvatureAnalysis,
    MomentCurvatureResult,
    SectionState,
)

YIELD_PENETRATION_COEFFICIENT = 32.0  # Ec. 7.71 (db en mm)
HINGE_SPREAD_RATIO = 0.06  # Ec. 7.72: propagación de la plasticidad, 6 % de L
SHEAR_MODULUS_RATIO = 0.4  # G = 0.4 Ec (§7.4.4b)
PRE_YIELD_SAMPLES = 120  # puntos para integrar la curvatura previa a la fluencia


# =========================================================================================
# Longitud de rótula y rotación plástica de diseño
# =========================================================================================


def yield_penetration_length(bar_diameter_mm: float) -> float:
    """Longitud de penetración de la fluencia Lpy = 32 sqrt(db) [mm] (Ec. 7.71)."""
    if bar_diameter_mm <= 0:
        raise ValueError("El diámetro de barra debe ser positivo.")
    return YIELD_PENETRATION_COEFFICIENT * sqrt(bar_diameter_mm)


def equivalent_plastic_hinge_length(column_length_mm: float, bar_diameter_mm: float, yield_penetration_mm: float | None = None) -> float:
    """Longitud equivalente de rótula plástica Lp = Lpy + 0.06 L [mm] (Ec. 7.72)."""
    if column_length_mm <= 0:
        raise ValueError("La longitud de la columna debe ser positiva.")
    lpy = yield_penetration_length(bar_diameter_mm) if yield_penetration_mm is None else yield_penetration_mm
    return lpy + HINGE_SPREAD_RATIO * column_length_mm


def design_plastic_rotation(ultimate_curvature: float, yield_curvature: float, plastic_hinge_length_mm: float) -> float:
    """Rotación plástica de diseño theta_p = Lp (phi_u - phi_y) (Ec. 7.67)."""
    return plastic_hinge_length_mm * max(ultimate_curvature - yield_curvature, 0.0)


# =========================================================================================
# Datos y resultados
# =========================================================================================


@dataclass(frozen=True)
class ColumnData:
    """Datos de la columna en voladizo equivalente.

    length_mm:                 L, de la sección crítica al punto de inflexión.
    yield_penetration_mm:      Lpy; None -> 32 sqrt(db) con el diámetro de la capa de barras traccionada.
    include_shear:             suma la deformación por cortante (Ec. 7.78-7.83).
    shear_area_mm2:            Aq (área de alma b·d); None -> b por el peralte efectivo.
    form_factor:               f de la Ec. 7.78 (1.2 en secciones rectangulares).
    hinge_length_mm:           Lh, longitud de la zona confinada de rótula; None -> h.
    outside_hoop_spacing_mm:   espaciamiento de estribos fuera de la rótula; None -> 2 s.
    hoop_shear_area_mm2:       Av, área de estribos que resiste el cortante; None -> acero transversal paralelo a y.
    """

    length_mm: float
    yield_penetration_mm: float | None = None
    include_shear: bool = False
    shear_area_mm2: float | None = None
    form_factor: float = 1.2
    hinge_length_mm: float | None = None
    outside_hoop_spacing_mm: float | None = None
    hoop_shear_area_mm2: float | None = None

    def __post_init__(self):
        if self.length_mm <= 0:
            raise ValueError("La longitud de la columna L debe ser positiva.")
        for name in ("yield_penetration_mm", "shear_area_mm2", "hinge_length_mm", "outside_hoop_spacing_mm", "hoop_shear_area_mm2"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} debe ser positivo.")
        if self.form_factor <= 0:
            raise ValueError("El factor de forma debe ser positivo.")


@dataclass
class ColumnResponse:
    """Respuesta fuerza-desplazamiento de la columna. Arreglos alineados con los estados del análisis M-phi."""

    data: ColumnData
    result: MomentCurvatureResult
    column_length_mm: float
    yield_penetration_mm: float
    plastic_hinge_length_mm: float  # Lp, Ec. 7.72
    first_yield_flexural_displacement: float  # integral de la curvatura previa a la fluencia (Ec. 7.75)
    first_yield_displacement: float  # Delta'_y = Delta_e + Delta_s (Ec. 7.92)
    yield_displacement: float  # Delta_y = Delta'_y Mi/My (Ec. 7.91)
    effective_stiffness: float  # EI_eff [N·mm²], Ec. 7.76
    curvature: np.ndarray
    moment: np.ndarray
    shear_force: np.ndarray  # V = M/L
    elastic_displacement: np.ndarray
    shear_displacement: np.ndarray
    plastic_displacement: np.ndarray
    displacement: np.ndarray
    plastic_rotation: np.ndarray
    ductility: np.ndarray  # mu = Delta/Delta_y
    shear_stiffnesses: dict = field(default_factory=dict)

    @property
    def yield_force(self) -> float:
        """V a la primera fluencia, My/L."""
        return self.result.yield_moment_first / self.column_length_mm

    def at_state(self, index: int) -> dict:
        """Magnitudes del paso `index` (N, mm, rad)."""
        return {
            "curvature": self.curvature[index], "moment": self.moment[index], "shear_force": self.shear_force[index],
            "elastic_displacement": self.elastic_displacement[index], "shear_displacement": self.shear_displacement[index],
            "plastic_displacement": self.plastic_displacement[index], "displacement": self.displacement[index],
            "plastic_rotation": self.plastic_rotation[index], "ductility": self.ductility[index],
        }

    def state_index(self, state: SectionState) -> int:
        """Posición de `state` en la lista de estados del análisis."""
        for i, candidate in enumerate(self.result.states):
            if candidate is state:
                return i
        raise ValueError("El estado no pertenece a este análisis.")

    def ultimate(self, key: str | None = None) -> dict:
        """Magnitudes en el estado límite `key` (None: el que terminó el análisis), con su curvatura y su rotación de diseño."""
        result = self.result
        if key is None:
            index = len(result.states) - 1
        else:
            record = next((r for r in result.limit_states if r.key == key and r.reached), None)
            if record is None:
                raise ValueError(f"El estado límite «{key}» no se alcanzó en este análisis.")
            index = self.state_index(record.state)
        values = self.at_state(index)
        phi_u = values["curvature"]
        values["design_plastic_rotation"] = design_plastic_rotation(phi_u, result.yield_curvature, self.plastic_hinge_length_mm)
        values["index"] = index
        return values

    def to_dataframe(self):
        """Tabla por incremento. Fuerza en kN, momento en kN·m, curvatura en µrad/mm, desplazamiento en mm, rotación en rad."""
        import pandas as pd

        return pd.DataFrame({
            "step": np.arange(len(self.curvature)),
            "curvature_urad_per_mm": self.curvature * 1e6,
            "moment_kNm": self.moment / 1e6,
            "lateral_force_kN": self.shear_force / 1e3,
            "elastic_displacement_mm": self.elastic_displacement,
            "shear_displacement_mm": self.shear_displacement,
            "plastic_displacement_mm": self.plastic_displacement,
            "displacement_mm": self.displacement,
            "plastic_rotation_rad": self.plastic_rotation,
            "displacement_ductility": self.ductility,
            "drift_ratio": self.displacement / self.column_length_mm,
        })


# =========================================================================================
# Cálculo
# =========================================================================================


def first_yield_flexural_displacement(moments, curvatures, length_mm: float) -> float:
    """Desplazamiento del extremo libre por flexión con la curva M-phi real previa a la fluencia (Ec. 7.75).

    Con momento lineal M(l) = My l/L, Delta = integral de phi(M) l dl = (L/My)² integral de phi M dM,
    integrada por la regla del trapecio sobre los puntos (M, phi) hasta My.
    """
    moments = np.asarray(moments, dtype=float)
    curvatures = np.asarray(curvatures, dtype=float)
    moment_yield = moments[-1]
    if moment_yield <= 0:
        raise ValueError("El momento de primera fluencia debe ser positivo.")
    integrand = curvatures * moments
    area = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(moments))
    return (length_mm / moment_yield) ** 2 * float(area)


def shear_displacement(moment, max_moment, length_mm, cracking_moment, k_uncracked, k_hinge, k_cracked, hinge_length_mm) -> float:
    """Deformación por cortante Delta_s [mm] de la columna (Ec. 7.83a, b y c).

    moment:         M actual en la sección crítica [N·mm].
    max_moment:     Mmax, el máximo alcanzado hasta ahora; fija cuánto de la columna está fisurada.
    cracking_moment: Mcr; None si la sección nunca se fisura.
    k_*:            rigideces a cortante Kve, Kvh y Kvc [N] de las zonas sin fisurar, de rótula y fisurada.
    """
    if moment <= 0:
        return 0.0
    if cracking_moment is None or max_moment <= cracking_moment:
        return moment / k_uncracked  # (a) sin fisuras
    ratio = cracking_moment / max_moment
    if max_moment <= cracking_moment / (1.0 - hinge_length_mm / length_mm):
        # (b) fisuración solo dentro de la zona de rótula
        return moment / k_uncracked * ratio + moment / k_hinge * (1.0 - ratio)
    # (c) la fisuración pasa la zona de rótula: sin fisurar + rótula + fisurada fuera de la rótula
    hinge_fraction = hinge_length_mm / length_mm
    return (moment / k_uncracked * ratio + moment / k_hinge * hinge_fraction
            + moment / k_cracked * (1.0 - ratio - hinge_fraction))


def _pre_yield_curve(result: MomentCurvatureResult):
    """Puntos (M, phi) desde cero hasta la primera fluencia, con muestreo fino y uniforme en phi."""
    analysis = MomentCurvatureAnalysis(result.model, result.settings)
    state = analysis.initial_state()
    phi_yield = result.yield_curvature_first
    moments, curvatures = [state.moment], [0.0]
    for phi in np.linspace(0.0, phi_yield, PRE_YIELD_SAMPLES + 1)[1:]:
        state = analysis.advance(state, float(phi))
        moments.append(state.moment)
        curvatures.append(state.phi)
    moments[-1] = result.yield_moment_first  # el último punto es exactamente la primera fluencia
    return np.array(moments), np.array(curvatures)


def compute_column_response(result: MomentCurvatureResult, data: ColumnData) -> ColumnResponse:
    """Curva fuerza lateral-desplazamiento de la columna a partir del análisis momento-curvatura."""
    if result.yield_curvature is None or result.yield_curvature_first is None or not result.yield_moment_first:
        raise ValueError(
            "La respuesta de la columna requiere la fluencia del acero (phi'y, My, phi_y), que no ocurre en este análisis "
            "(sección controlada por compresión)."
        )
    model, section = result.model, result.model.section
    length = data.length_mm
    moment_yield, phi_yield_first = result.yield_moment_first, result.yield_curvature_first

    bar_diameter = section.layers[-1].diameter_mm  # capa traccionada (inferior)
    lpy = data.yield_penetration_mm if data.yield_penetration_mm is not None else yield_penetration_length(bar_diameter)
    lp = lpy + HINGE_SPREAD_RATIO * length

    # --- Flexión elástica: EI efectivo a primera fluencia (Ec. 7.75-7.77) ---
    pre_moments, pre_curvatures = _pre_yield_curve(result)
    flexural_yield = first_yield_flexural_displacement(pre_moments, pre_curvatures, length)
    effective_stiffness = moment_yield * length**2 / (3.0 * flexural_yield)  # Ec. 7.76

    # --- Cortante (opcional): rigideces de las zonas (Ec. 7.78-7.83) ---
    shear = {}
    cracking = next((r for r in result.limit_states if r.key == "first_cracking" and r.reached), None)
    cracking_moment = None if cracking is None else cracking.moment
    if data.include_shear:
        ec = model.core.elastic_modulus
        effective_depth = section.layers[-1].depth_mm
        shear_area = data.shear_area_mm2 or section.width_mm * effective_depth
        hoop_area = data.hoop_shear_area_mm2 or section.transverse.area_y_mm2
        hinge_spacing = section.transverse.spacing_mm
        outside_spacing = data.outside_hoop_spacing_mm or 2.0 * hinge_spacing
        concrete_term = 4.0 / (ec * shear_area)

        def cracked(spacing):  # Ec. 7.81, con Es = 200000 MPa como en el listado de COLUMN
            return 1.0 / (1.0 / (CODE_STEEL_MODULUS * hoop_area * effective_depth / spacing) + concrete_term)

        shear = {
            "uncracked": SHEAR_MODULUS_RATIO * ec * shear_area / data.form_factor,  # Ec. 7.78
            "hinge": cracked(hinge_spacing),
            "cracked": cracked(outside_spacing),
            "hinge_length_mm": data.hinge_length_mm or section.depth_mm,
        }

    def shear_part_of(moment, max_moment):
        if not shear:
            return 0.0
        return shear_displacement(moment, max_moment, length, cracking_moment, shear["uncracked"], shear["hinge"],
                                  shear["cracked"], shear["hinge_length_mm"])

    # --- Desplazamiento de primera fluencia y de fluencia (Ec. 7.91-7.92) ---
    first_yield_total = flexural_yield + shear_part_of(moment_yield, moment_yield)  # Delta'_y = Delta_e + Delta_s
    yield_displacement = first_yield_total * result.ideal_moment / moment_yield

    # --- Respuesta en cada paso del análisis M-phi ---
    states = result.states
    curvature = np.array([st.phi for st in states])
    moment = np.array([st.moment for st in states])
    elastic = np.zeros(len(states))
    shear_part = np.zeros(len(states))
    plastic = np.zeros(len(states))
    rotation = np.zeros(len(states))
    for i, st in enumerate(states):
        elastic[i] = st.moment * length**2 / (3.0 * effective_stiffness)  # Ec. 7.77
        shear_part[i] = shear_part_of(st.moment, st.max_moment)
        if st.phi >= phi_yield_first:  # ya fluyó: la parte plástica sigue aunque M descienda tras el pico
            plastic_curvature = max(st.phi - phi_yield_first * st.moment / moment_yield, 0.0)  # Ec. 7.84
            plastic_zone = length * (1.0 - moment_yield / st.max_moment)  # Ec. 7.85
            rotation[i] = plastic_curvature * (plastic_zone / 3.0 + lpy)  # Ec. 7.87
            plastic[i] = rotation[i] * (length - 0.25 * plastic_zone)  # Ec. 7.88
    displacement = elastic + shear_part + plastic

    return ColumnResponse(
        data=data, result=result, column_length_mm=length, yield_penetration_mm=lpy, plastic_hinge_length_mm=lp,
        first_yield_flexural_displacement=flexural_yield, first_yield_displacement=first_yield_total,
        yield_displacement=yield_displacement, effective_stiffness=effective_stiffness, curvature=curvature, moment=moment,
        shear_force=moment / length, elastic_displacement=elastic, shear_displacement=shear_part, plastic_displacement=plastic,
        displacement=displacement, plastic_rotation=rotation, ductility=displacement / yield_displacement,
        shear_stiffnesses=shear,
    )
