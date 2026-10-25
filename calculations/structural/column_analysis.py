"""Análisis momento-curvatura monotónico de una sección rectangular (reconstrucción tipo COLUMN).

Es una RECONSTRUCCIÓN NUMÉRICA basada en la literatura, no una traducción de
COLUMN: se siguió la descripción de la sección 7.4.2 de Mander (1984) y las
ecuaciones de los Cap. 2 y 3, y se consultó el listado FORTRAN (Apéndice A)
para constantes y salvaguardas. No se implementan las ramas cíclicas, la
penetración de fluencia, la deformación por cortante ni los efectos de
velocidad de deformación de COLUMN.

Fuente principal: Mander, J. B. (1984), *Seismic Design of Bridge Piers*,
Research Report 84-2, Univ. of Canterbury.

Algoritmo (sección 7.4.2, "Algorithm 2", pp. 284-286):
    1. Se impone un incremento de curvatura  phi_n = phi_(n-1) + dphi        (7.57)
    2. Se corrige la deformación del eje de referencia para mantener el
       equilibrio axial:  d(eps0) = (P_externa - N) / EA   (7.58, con signos
       adaptados a compresión positiva)
    3. Se recalculan N y M con la nueva distribución de deformaciones         (7.42, 7.43)
       eps(y) = eps0 + phi y                                                  (7.41)
    4. Se repite hasta |N - P| < tolerancia y se pasa al siguiente incremento.
    EA = suma(E_t A) es la rigidez axial tangente actual                      (7.45)

Convención: compresión positiva; y desde la mitad del peralte, positivo hacia
la cara comprimida. Unidades: N, mm, MPa; curvatura en 1/mm.
"""

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from calculations.structural.column_confinement import ConfinementResult, confine_rectangular_core
from calculations.structural.column_materials import PopovicsConcrete, ReinforcingSteel
from calculations.structural.column_section import RectangularSection, SectionMesh, build_mesh
from calculations.structural.root_finding import bisect, maximize_bounded

# --- Constantes tomadas de la tesis / del listado de COLUMN -------------------------------
CODE_ULTIMATE_STRAIN = 0.003  # deformación del concreto para la resistencia ideal Mi (sección 7.4.2)
CODE_STEEL_MODULUS = 200000.0  # MPa; el listado (A27) usa 200000 en el modo de resistencia ideal
HOOP_WORK_CAPACITY = 105.0  # MJ/m³ = MPa; energía de fractura de la barra del estribo (Ec. 7.61, 3.73)
CONCRETE_WORK_FACTOR = 0.017  # Ec. 7.61: energía del concreto no confinado = 0.017 sqrt(f'c) MJ/m³
# Incrementos de curvatura de COLUMN (p. 286): 0.2, 0.5 y 1.0 phi_y
INCREMENT_STAGES = ((2.0, 0.2), (7.0, 0.5), (np.inf, 1.0))  # (hasta phi/phi_y, incremento/phi_y)


@dataclass(frozen=True)
class ConcreteProperties:
    """Datos del concreto no confinado (tarjeta 2 de COLUMN).

    unconfined_strength: f'co [MPa], resistencia in situ.
    strain_at_peak:      eco (0.002 típico).
    spall_strain:        deformación a la que el recubrimiento deja de resistir (dato del usuario).
    tensile_strength:    f't [MPa]. None -> 0.6 sqrt(f'co), valor típico (supuesto, no de COLUMN).
    elastic_modulus:     Ec [MPa]. None -> 5000 sqrt(f'co) (sección 3.6.2 de la tesis).
    """

    unconfined_strength: float
    strain_at_peak: float = 0.002
    spall_strain: float = 0.008
    tensile_strength: float | None = None
    elastic_modulus: float | None = None

    @property
    def resolved_tensile_strength(self) -> float:
        return 0.6 * self.unconfined_strength**0.5 if self.tensile_strength is None else self.tensile_strength

    @property
    def resolved_elastic_modulus(self) -> float:
        return 5000.0 * self.unconfined_strength**0.5 if self.elastic_modulus is None else self.elastic_modulus


@dataclass(frozen=True)
class ColumnModel:
    """Sección discretizada con sus materiales: todo lo necesario para integrar."""

    section: RectangularSection
    mesh: SectionMesh
    cover: PopovicsConcrete
    core: PopovicsConcrete
    steel: ReinforcingSteel
    concrete: ConcreteProperties
    confinement: ConfinementResult


def build_column_model(
    section: RectangularSection,
    concrete: ConcreteProperties,
    steel: ReinforcingSteel,
    confinement: ConfinementResult | None = None,
    number_of_strips: int = 60,
) -> ColumnModel:
    """Arma el modelo. Si no se da `confinement` se calcula con el método del Cap. 3.

    COLUMN recibe f'cc y ecc como datos (tarjeta 3); aquí pueden fijarse pasando
    un ConfinementResult propio.
    """
    if confinement is None:
        confinement = confine_rectangular_core(section, concrete.unconfined_strength, concrete.strain_at_peak)
    ec = concrete.resolved_elastic_modulus
    ft = concrete.resolved_tensile_strength
    cover = PopovicsConcrete(
        concrete.unconfined_strength, concrete.strain_at_peak, ec, ft, spall_strain=concrete.spall_strain
    )
    core = PopovicsConcrete(confinement.confined_strength, confinement.confined_strain, ec, ft)
    return ColumnModel(section, build_mesh(section, number_of_strips), cover, core, steel, concrete, confinement)


@dataclass(frozen=True)
class AnalysisSettings:
    """Parámetros del análisis.

    axial_load_n: carga axial externa [N], compresión positiva.
    increment_schedule: "column" (0.2/0.5/1.0 phi_y, p. 286) o "uniform".
    uniform_increment: dphi [1/mm] si el esquema es "uniform".
    increment_factor: multiplica todos los incrementos (< 1 refina la curva).
    core_ultimate_strain: ecu del núcleo (dato de COLUMN, tarjeta 3); None lo desactiva.
    use_hoop_fracture: terminar cuando el trabajo en el núcleo supera la energía del
                       estribo (balance de energía, Ec. 7.61-7.66).
    hoop_length_mm: Ls, longitud del estribo crítico; None -> perímetro del estribo perimetral.
    work_fraction_core / work_fraction_steel: Wc y Ws (Ec. 7.62); 1.0 = el estribo
                       perimetral resiste todo el trabajo.
    max_curvature_ductility: termina cuando phi/phi_y lo alcanza; None lo desactiva.
    moment_drop_ratio: termina cuando M cae a esta fracción de Mmax; None lo desactiva.
    equilibrium_tolerance: tolerancia de |N - P| como fracción de f'co Ag (COLUMN usa 5e-4).
    """

    axial_load_n: float = 0.0
    increment_schedule: str = "column"
    uniform_increment: float | None = None
    increment_factor: float = 1.0
    core_ultimate_strain: float | None = None
    use_hoop_fracture: bool = True
    hoop_length_mm: float | None = None
    work_fraction_core: float = 1.0
    work_fraction_steel: float = 1.0
    max_curvature_ductility: float | None = None
    moment_drop_ratio: float | None = None
    equilibrium_tolerance: float = 1e-5
    max_iterations: int = 15
    max_steps: int = 3000

    def __post_init__(self):
        if self.increment_schedule not in ("column", "uniform"):
            raise ValueError("El esquema de incrementos debe ser 'column' o 'uniform'.")
        if self.increment_schedule == "uniform" and not (self.uniform_increment and self.uniform_increment > 0):
            raise ValueError("Con esquema uniforme se requiere un incremento de curvatura positivo.")
        if self.increment_factor <= 0:
            raise ValueError("El factor de incremento debe ser positivo.")
        if self.equilibrium_tolerance <= 0 or self.max_iterations < 3 or self.max_steps < 10:
            raise ValueError("Tolerancia, iteraciones o pasos máximos inválidos.")
        for name in ("core_ultimate_strain", "max_curvature_ductility", "moment_drop_ratio", "hoop_length_mm"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} debe ser positivo.")
        if self.moment_drop_ratio is not None and self.moment_drop_ratio >= 1.0:
            raise ValueError("La fracción de caída de momento debe ser menor que 1.")
        if not (0 < self.work_fraction_core <= 1 and 0 < self.work_fraction_steel <= 1):
            raise ValueError("Las fracciones de trabajo Wc y Ws deben estar en (0, 1].")


@dataclass(frozen=True)
class FibreHistory:
    """Historia mínima por lámina: máxima compresión y máxima tracción alcanzadas."""

    max_compression: np.ndarray
    max_tension: np.ndarray


@dataclass(frozen=True)
class SectionState:
    """Estado de la sección en un valor de curvatura (resultado de un paso convergido)."""

    phi: float  # curvatura [1/mm]
    eps0: float  # deformación en el eje de referencia (mitad del peralte)
    axial_force: float  # N [N]
    moment: float  # M [N*mm] respecto a la mitad del peralte
    axial_stiffness: float  # EA tangente [N]
    first_moment_stiffness: float  # EZ tangente [N*mm]
    iterations: int
    residual: float  # N - P [N]
    converged: bool
    history: FibreHistory
    max_moment: float  # momento máximo alcanzado hasta este estado
    core_work: float  # trabajo acumulado en el núcleo y las barras comprimidas [N] (= MJ/m x 1e6)
    eps_lamina: np.ndarray
    sigma_core: np.ndarray
    eps_bars: np.ndarray
    sigma_bars: np.ndarray

    def strain_at(self, y_mm):
        return self.eps0 + self.phi * np.asarray(y_mm)


@dataclass(frozen=True)
class LimitState:
    """Estado límite o evento: ocurre cuando `measure(state)` pasa de negativo a >= 0."""

    key: str
    label: str
    definition: str
    terminal: bool
    measure: Callable[[SectionState], float]


@dataclass(frozen=True)
class LimitStateRecord:
    key: str
    label: str
    definition: str
    terminal: bool
    reached: bool
    curvature: float | None = None  # [1/mm]
    moment: float | None = None  # [N*mm]
    state: SectionState | None = None


@dataclass
class MomentCurvatureResult:
    """Curva momento-curvatura y la información para interpretarla."""

    model: ColumnModel
    settings: AnalysisSettings
    states: list
    limit_states: list
    yield_curvature_first: float | None  # phi'_y: primera fluencia del acero [1/mm]
    yield_moment_first: float | None  # M_y [N*mm]
    ideal_moment: float  # Mi [N*mm]
    yield_curvature: float | None  # phi_y = phi'_y Mi / My (Ec. 7.60) [1/mm]
    termination_key: str
    termination_message: str
    warnings: list = field(default_factory=list)

    # --- vistas convenientes (arreglos en unidades de trabajo del módulo) ---
    @property
    def curvature(self):
        return np.array([s.phi for s in self.states])

    @property
    def moment(self):
        return np.array([s.moment for s in self.states])

    @property
    def maximum(self):
        """Estado de momento máximo (refinado)."""
        return max(self.states, key=lambda s: s.moment)

    def ductility_of(self, curvature: float | None):
        if curvature is None or not self.yield_curvature:
            return None
        return curvature / self.yield_curvature

    def to_dataframe(self):
        """Tabla por incremento. Curvatura en µrad/mm (= 1e-3 1/m, como en la tesis), momento en kN·m, fuerza en kN."""
        import pandas as pd

        section = self.model.section
        half_depth = section.depth_mm / 2.0
        half_core = section.core_depth_mm / 2.0
        bar_y = self.model.mesh.bar_y
        events = {}
        for record in self.limit_states:
            if record.reached:
                events.setdefault(id(record.state), []).append(record.label)
        rows = []
        for i, s in enumerate(self.states):
            phi = s.phi
            rows.append(
                {
                    "step": i,
                    "curvature_urad_per_mm": phi * 1e6,
                    "moment_kNm": s.moment / 1e6,
                    "axial_strain_eps0": s.eps0,
                    "neutral_axis_depth_mm": (half_depth + s.eps0 / phi) if phi > 0 else np.nan,
                    "strain_extreme_cover": s.eps0 + phi * half_depth,
                    "strain_core_edge": s.eps0 + phi * half_core,
                    "strain_bar_compression": s.eps0 + phi * bar_y[-1],
                    "strain_bar_tension": s.eps0 + phi * bar_y[0],
                    "curvature_ductility": self.ductility_of(phi),
                    "iterations": s.iterations,
                    "axial_residual_kN": s.residual / 1e3,
                    "converged": s.converged,
                    "limit_state": "; ".join(events.get(id(s), [])),
                }
            )
        return pd.DataFrame(rows)


# =========================================================================================
# Integración de la sección
# =========================================================================================


def _integrate(model: ColumnModel, eps0: float, phi: float, history: FibreHistory):
    """N, M y rigideces tangentes para una distribución de deformaciones (Ec. 7.41-7.48).

    N = suma(f_c A_c) + suma(f_s A_s)        M = suma(f A y)
    EA = suma(E_t A)                          EZ = suma(E_t A y)
    """
    mesh = model.mesh
    eps = eps0 + phi * mesh.y
    # Hipótesis 3 y 4 (sección 7.4.2): el recubrimiento descascarado y el concreto fisurado no resisten
    spalled = history.max_compression >= model.cover.spall_strain
    cracked_cover = history.max_tension > model.cover.cracking_strain
    cracked_core = history.max_tension > model.core.cracking_strain
    sigma_core, tangent_core = model.core.stress_and_tangent(eps, cracked=cracked_core)
    sigma_cover, tangent_cover = model.cover.stress_and_tangent(eps, cracked=cracked_cover, spalled=spalled)

    eps_bars = eps0 + phi * mesh.bar_y
    sigma_bars, tangent_bars = model.steel.stress_and_tangent(eps_bars)

    force_core = sigma_core * mesh.core_area
    force_cover = sigma_cover * mesh.cover_area
    force_bars = sigma_bars * mesh.bar_area
    axial = force_core.sum() + force_cover.sum() + force_bars.sum()
    moment = ((force_core + force_cover) * mesh.y).sum() + (force_bars * mesh.bar_y).sum()

    stiff_concrete = tangent_core * mesh.core_area + tangent_cover * mesh.cover_area
    stiff_bars = tangent_bars * mesh.bar_area
    ea = stiff_concrete.sum() + stiff_bars.sum()
    ez = (stiff_concrete * mesh.y).sum() + (stiff_bars * mesh.bar_y).sum()
    return axial, moment, ea, ez, eps, sigma_core, eps_bars, sigma_bars


class MomentCurvatureAnalysis:
    """Ejecuta el análisis para un modelo y unos parámetros dados."""

    def __init__(self, model: ColumnModel, settings: AnalysisSettings):
        self.model = model
        self.settings = settings
        section = model.section
        self.axial_load = settings.axial_load_n
        self.tolerance = settings.equilibrium_tolerance * model.concrete.unconfined_strength * section.gross_area_mm2
        # Piso de la rigidez axial para que Newton no divida por ~0 en tramos plastificados
        self.stiffness_floor = 1e-3 * model.core.elastic_modulus * section.gross_area_mm2
        self.hoop_length = settings.hoop_length_mm or 2.0 * (section.core_width_mm + section.core_depth_mm)
        t = section.transverse
        # Capacidad de energía del estribo crítico (Ec. 7.61a); MPa x mm² = N
        self.hoop_energy_capacity = (
            HOOP_WORK_CAPACITY * t.bar_area_mm2 * self.hoop_length / t.spacing_mm
            + settings.work_fraction_core
            * model.mesh.core_area.sum()
            * CONCRETE_WORK_FACTOR
            * model.concrete.unconfined_strength**0.5
        )
        self._check_axial_load()

    # ----- equilibrio -----
    def _check_axial_load(self):
        """Verifica que la carga axial sea compatible con la capacidad axial de la sección."""
        zero_history = FibreHistory(np.zeros(len(self.model.mesh.y)), np.zeros(len(self.model.mesh.y)))
        strains = np.linspace(-0.02, 0.05, 701)
        forces = np.array([_integrate(self.model, e, 0.0, zero_history)[0] for e in strains])
        if not forces.min() <= self.axial_load <= forces.max():
            raise ValueError(
                f"La carga axial P = {self.axial_load / 1e3:,.0f} kN está fuera del intervalo que la sección "
                f"puede equilibrar ({forces.min() / 1e3:,.0f} a {forces.max() / 1e3:,.0f} kN)."
            )
        self.axial_capacity = float(forces.max())

    def _equilibrium(self, phi, eps0_guess, history):
        """Resuelve eps0 tal que N(eps0, phi) = P por Newton con retroceso (Ec. 7.49 y 7.58)."""
        eps0 = float(eps0_guess)
        axial, moment, ea, ez, *rest = _integrate(self.model, eps0, phi, history)
        imbalance = axial - self.axial_load
        iterations = 0
        while abs(imbalance) > self.tolerance and iterations < self.settings.max_iterations:
            iterations += 1
            step = -imbalance / max(ea, self.stiffness_floor)
            step = float(np.clip(step, -0.01, 0.01))
            for _ in range(12):  # retroceso: no se acepta un paso que empeore el equilibrio
                trial = _integrate(self.model, eps0 + step, phi, history)
                if abs(trial[0] - self.axial_load) < abs(imbalance) or abs(step) < 1e-12:
                    break
                step *= 0.5
            eps0 += step
            axial, moment, ea, ez, *rest = trial
            imbalance = axial - self.axial_load
        converged = abs(imbalance) <= self.tolerance
        if not converged:
            root = self._bracketed_root(phi, eps0_guess, history)
            if root is not None:
                eps0 = root
                axial, moment, ea, ez, *rest = _integrate(self.model, eps0, phi, history)
                imbalance = axial - self.axial_load
                converged = abs(imbalance) <= self.tolerance
        return eps0, (axial, moment, ea, ez, *rest), iterations, imbalance, converged

    def _bracketed_root(self, phi, eps0_guess, history):
        """Respaldo robusto cuando Newton falla (ramas descendentes): raíz más cercana a la anterior."""
        grid = eps0_guess + np.linspace(-0.02, 0.02, 401)
        values = np.array([_integrate(self.model, e, phi, history)[0] - self.axial_load for e in grid])
        crossings = np.where(values[:-1] * values[1:] <= 0.0)[0]
        if len(crossings) == 0:
            return None
        nearest = crossings[np.argmin(np.abs(grid[crossings] - eps0_guess))]
        return bisect(lambda e: _integrate(self.model, e, phi, history)[0] - self.axial_load, grid[nearest], grid[nearest + 1], xtol=1e-14)

    def _make_state(self, phi, eps0, integration, iterations, imbalance, converged, previous: SectionState | None, history_in):
        axial, moment, ea, ez, eps, sigma_core, eps_bars, sigma_bars = integration
        history = FibreHistory(np.maximum(history_in.max_compression, eps), np.maximum(history_in.max_tension, -eps))
        work = 0.0 if previous is None else previous.core_work + self._work_increment(previous, eps, sigma_core, eps_bars, sigma_bars)
        return SectionState(
            phi=phi,
            eps0=eps0,
            axial_force=axial,
            moment=moment,
            axial_stiffness=ea,
            first_moment_stiffness=ez,
            iterations=iterations,
            residual=imbalance,
            converged=converged,
            history=history,
            max_moment=moment if previous is None else max(previous.max_moment, moment),
            core_work=work,
            eps_lamina=eps,
            sigma_core=sigma_core,
            eps_bars=eps_bars,
            sigma_bars=sigma_bars,
        )

    def _work_increment(self, previous, eps, sigma_core, eps_bars, sigma_bars):
        """Incremento de trabajo del núcleo y de las barras comprimidas (Ec. 7.64a, regla trapezoidal).

        Solo cuentan los incrementos de compresión (de > 0) en elementos comprimidos,
        que son los que resiste el estribo crítico durante la rotación positiva.
        """
        s = self.settings
        mesh = self.model.mesh
        d_core = eps - previous.eps_lamina
        core = np.where((d_core > 0) & (sigma_core > 0), 0.5 * (previous.sigma_core + sigma_core) * d_core, 0.0)
        d_bar = eps_bars - previous.eps_bars
        bars = np.where((d_bar > 0) & (sigma_bars > 0), 0.5 * (previous.sigma_bars + sigma_bars) * d_bar, 0.0)
        return s.work_fraction_core * (core * mesh.core_area).sum() + s.work_fraction_steel * (bars * mesh.bar_area).sum()

    def initial_state(self) -> SectionState:
        """Estado a curvatura cero: deformación axial uniforme que equilibra P."""
        n = len(self.model.mesh.y)
        history = FibreHistory(np.zeros(n), np.zeros(n))
        eps0, integration, iterations, imbalance, converged = self._equilibrium(0.0, 0.0, history)
        if not converged:
            raise ValueError("No se pudo equilibrar la carga axial a curvatura cero.")
        return self._make_state(0.0, eps0, integration, iterations, imbalance, converged, None, history)

    def advance(self, previous: SectionState, phi: float) -> SectionState:
        """Un paso: impone la curvatura `phi` y corrige eps0 hasta equilibrar N = P (Algoritmo 2)."""
        if previous.axial_stiffness > self.stiffness_floor:
            guess = previous.eps0 - previous.first_moment_stiffness / previous.axial_stiffness * (phi - previous.phi)
        else:
            guess = previous.eps0
        eps0, integration, iterations, imbalance, converged = self._equilibrium(phi, guess, previous.history)
        return self._make_state(phi, eps0, integration, iterations, imbalance, converged, previous, previous.history)

    # ----- estados límite -----
    def limit_states(self, ideal_yield_ductility_limit=None) -> list:
        """Lista de estados límite/eventos con su criterio explícito."""
        m, s = self.model, self.settings
        section = m.section
        half_depth = section.depth_mm / 2.0
        half_core = section.core_depth_mm / 2.0
        y_tension, y_comp = m.mesh.bar_y[0], m.mesh.bar_y[-1]
        steel = m.steel

        def extreme_cover(st): return st.eps0 + st.phi * half_depth
        def core_edge(st): return st.eps0 + st.phi * half_core
        def tension_bar(st): return -(st.eps0 + st.phi * y_tension)
        def bar_magnitude(st): return max(abs(st.eps0 + st.phi * y_tension), abs(st.eps0 + st.phi * y_comp))

        states = [
            LimitState("first_yield", "Primera fluencia del acero",
                       "La barra más traccionada alcanza ey = fy/Es.", False, lambda st: tension_bar(st) - steel.yield_strain),
            LimitState("code_strain", "Deformación 0.003 en la fibra extrema",
                       "El concreto de la fibra extrema alcanza 0.003 (referencia del método de diseño de la norma).", False,
                       lambda st: extreme_cover(st) - CODE_ULTIMATE_STRAIN),
            LimitState("cover_peak", "Recubrimiento en su esfuerzo máximo",
                       "La fibra extrema del recubrimiento alcanza eco (f'co): inicio de la rama descendente.", False,
                       lambda st: extreme_cover(st) - m.concrete.strain_at_peak),
            LimitState("cover_spalling", "Descascaramiento del recubrimiento",
                       "La fibra extrema del recubrimiento alcanza la deformación de descascaramiento ingresada.", False,
                       lambda st: extreme_cover(st) - m.concrete.spall_strain),
            LimitState("core_peak", "Núcleo en su esfuerzo máximo",
                       "La fibra extrema del núcleo (eje del estribo) alcanza ecc (f'cc).", False,
                       lambda st: core_edge(st) - m.core.peak_strain),
            LimitState("steel_hardening", "Inicio del endurecimiento del acero",
                       "La barra más traccionada alcanza esh.", False, lambda st: tension_bar(st) - steel.hardening_strain),
        ]
        if s.core_ultimate_strain is not None:
            states.append(LimitState("core_ultimate", "Deformación última del núcleo (dato)",
                                     f"La fibra extrema del núcleo alcanza ecu = {s.core_ultimate_strain:g} ingresada.", True,
                                     lambda st: core_edge(st) - s.core_ultimate_strain))
        if s.use_hoop_fracture:
            states.append(LimitState("hoop_fracture", "Fractura del estribo (balance de energía)",
                                     "El trabajo hecho sobre el núcleo y las barras comprimidas supera la energía de "
                                     "deformación del estribo crítico (Ec. 7.61-7.66).", True,
                                     lambda st: st.core_work - self.hoop_energy_capacity))
        states.append(LimitState("steel_ultimate", "Deformación última del acero",
                                 "La barra más deformada alcanza esu.", True, lambda st: bar_magnitude(st) - steel.ultimate_strain))
        if s.max_curvature_ductility is not None and ideal_yield_ductility_limit is not None:
            states.append(LimitState("ductility_limit", "Límite de ductilidad de curvatura",
                                     f"phi/phi_y alcanza {s.max_curvature_ductility:g} (límite ingresado).", True,
                                     lambda st: st.phi - ideal_yield_ductility_limit))
        if s.moment_drop_ratio is not None:
            ratio = s.moment_drop_ratio
            states.append(LimitState("moment_drop", f"Caída del momento al {ratio:.0%} de Mmax",
                                     f"M desciende a {ratio:.0%} del momento máximo alcanzado.", True,
                                     lambda st: ratio * st.max_moment - st.moment if st.max_moment > 0 else -1.0))
        return states

    # ----- marcha de curvatura -----
    def _increment(self, phi, phi_y):
        s = self.settings
        if s.increment_schedule == "uniform":
            return s.uniform_increment * s.increment_factor
        for upper, fraction in INCREMENT_STAGES:
            if phi < upper * phi_y:
                return fraction * phi_y * s.increment_factor
        raise AssertionError("unreachable")

    def _refine(self, previous: SectionState, current: SectionState, limit: LimitState) -> SectionState:
        """Ubica por bisección la curvatura en que `limit` ocurre exactamente entre dos pasos."""
        def g(phi):
            return limit.measure(self.advance(previous, phi))
        phi = bisect(g, previous.phi, current.phi, xtol=1e-14)
        return self.advance(previous, phi)

    def _march(self, phi_y, limits, stop_key=None):
        """Avanza la curvatura hasta un estado límite terminal (o `stop_key`)."""
        states = [self.initial_state()]
        pending = {limit.key: limit for limit in limits}
        reached: dict[str, SectionState] = {}
        termination = ("max_steps", f"Se alcanzó el máximo de {self.settings.max_steps} pasos.")
        warnings = []

        for _ in range(self.settings.max_steps):
            previous = states[-1]
            dphi = self._increment(previous.phi, phi_y)
            current = None
            for _halving in range(8):  # si no converge se reduce el incremento a la mitad
                candidate = self.advance(previous, previous.phi + dphi)
                if candidate.converged and np.isfinite(candidate.moment):
                    current = candidate
                    break
                dphi *= 0.5
            if current is None:
                termination = ("no_convergence", "No convergió el equilibrio axial ni reduciendo el incremento de curvatura.")
                warnings.append(f"Sin convergencia en phi = {previous.phi * 1000:.4g} 1/m; el análisis se detuvo allí.")
                break

            crossings = []
            for key, limit in pending.items():
                if limit.measure(previous) < 0.0 <= limit.measure(current):
                    crossings.append((self._refine(previous, current, limit), limit))
            crossings.sort(key=lambda item: item[0].phi)

            stop = None
            for state, limit in crossings:
                states.append(state)
                reached[limit.key] = state
                del pending[limit.key]
                if limit.terminal or limit.key == stop_key:
                    stop = limit
                    break
            if stop is not None:
                termination = (stop.key, stop.label)
                break
            states.append(current)
        return states, reached, termination, warnings

    def run(self) -> MomentCurvatureResult:
        """Ejecuta el análisis completo: primera fluencia, Mi, phi_y y curva hasta el estado límite terminal."""
        warnings = []
        ideal = ideal_flexural_strength(self.model, self.axial_load)

        # Pasada previa: primera fluencia (phi'_y, My) con incrementos finos y uniformes
        probe = self.model.steel.yield_strain / self.model.section.depth_mm / 5.0
        pre_settings = replace(self.settings, increment_schedule="uniform", uniform_increment=probe, increment_factor=1.0)
        pre = MomentCurvatureAnalysis(self.model, pre_settings)
        pre_limits = [lim for lim in pre.limit_states() if lim.key in ("first_yield", "hoop_fracture", "steel_ultimate", "core_ultimate")]
        _, hits, _, _ = pre._march(probe, pre_limits, stop_key="first_yield")
        phi_y_first = hits["first_yield"].phi if "first_yield" in hits else None
        moment_y = hits["first_yield"].moment if "first_yield" in hits else None

        if phi_y_first is None or moment_y is None or moment_y <= 0.0:
            phi_y = None
            warnings.append(
                "El acero no alcanzó la fluencia antes del estado límite terminal: no se define phi_y ni la ductilidad."
            )
            scale = probe * 5.0
        else:
            phi_y = phi_y_first * ideal / moment_y  # Ec. 7.60
            scale = phi_y

        # Si se pidió un límite de ductilidad se convierte en curvatura
        duct_limit = None
        if self.settings.max_curvature_ductility is not None and phi_y is not None:
            duct_limit = self.settings.max_curvature_ductility * phi_y
        elif self.settings.max_curvature_ductility is not None:
            warnings.append("El límite de ductilidad no se aplicó porque phi_y no está definida.")

        limits = self.limit_states(duct_limit)
        states, reached, (term_key, term_label), march_warnings = self._march(scale, limits)
        warnings.extend(march_warnings)

        states, max_state = self._refine_maximum(states)
        if len(states) < 3:
            warnings.append("Se obtuvieron muy pocos puntos de la curva.")
        reached["max_moment"] = max_state

        records = []
        for limit in limits:
            state = reached.get(limit.key)
            records.append(LimitStateRecord(limit.key, limit.label, limit.definition, limit.terminal, state is not None,
                                            None if state is None else state.phi, None if state is None else state.moment, state))
        records.append(LimitStateRecord("max_moment", "Momento máximo", "Máximo de M(phi) de la curva calculada (refinado).",
                                        False, True, max_state.phi, max_state.moment, max_state))
        records.sort(key=lambda r: (r.curvature is None, r.curvature or 0.0))
        return MomentCurvatureResult(
            model=self.model, settings=self.settings, states=states, limit_states=records,
            yield_curvature_first=phi_y_first, yield_moment_first=moment_y, ideal_moment=ideal, yield_curvature=phi_y,
            termination_key=term_key, termination_message=term_label, warnings=warnings,
        )

    def _refine_maximum(self, states):
        """Refina el momento máximo entre los pasos vecinos al máximo discreto."""
        k = int(np.argmax([st.moment for st in states]))
        if 0 < k < len(states) - 1:
            start, lo, hi = states[k - 1], states[k - 1].phi, states[k + 1].phi
            phi_max = maximize_bounded(lambda p: self.advance(start, p).moment, lo, hi, xtol=1e-12)
            best = self.advance(start, phi_max)
            if best.converged and best.moment > states[k].moment:
                return sorted(states + [best], key=lambda st: st.phi), best
        return states, states[k]


# =========================================================================================
# Resistencia ideal (método de la norma) y utilidades
# =========================================================================================


def ideal_flexural_strength(model: ColumnModel, axial_load_n: float, concrete_strength: float | None = None) -> float:
    """Momento ideal Mi [N*mm] con el método de diseño que usa COLUMN (modo 1 del listado).

    Deformación 0.003 en la fibra extrema, bloque rectangular 0.85 f'c de profundidad
    a = beta1 c con beta1 = 0.85 - 0.008 (f'c - 30) acotado a [0.65, 0.85] (listado, línea
    "BETAI"), acero elastoplástico con Es = 200000 MPa y área de concreto neta de barras.
    El momento es respecto a la mitad del peralte. `concrete_strength` permite usar una
    f'c distinta de f'co del modelo (p. ej. la resistencia de cilindros el día del ensayo).
    """
    section, steel = model.section, model.steel
    fc = model.concrete.unconfined_strength if concrete_strength is None else concrete_strength
    beta1 = float(np.clip(0.85 - 0.008 * (fc - 30.0), 0.65, 0.85))
    h, b = section.depth_mm, section.width_mm
    bar_y, _, layer_area = section.bar_layers()
    depth_from_top = h / 2.0 - bar_y

    def forces(c):
        a = min(beta1 * c, h)
        in_block = depth_from_top <= a
        steel_strain = CODE_ULTIMATE_STRAIN * (c - depth_from_top) / c
        steel_stress = np.clip(CODE_STEEL_MODULUS * steel_strain, -steel.yield_stress, steel.yield_stress)
        concrete = 0.85 * fc * (b * a - layer_area[in_block].sum())
        axial = concrete + (steel_stress * layer_area).sum()
        moment = 0.85 * fc * (b * a * (h / 2.0 - a / 2.0) - (layer_area[in_block] * bar_y[in_block]).sum())
        moment += (steel_stress * layer_area * bar_y).sum()
        return axial, moment

    lo, hi = 1e-3 * h, 50.0 * h
    f_lo, f_hi = forces(lo)[0] - axial_load_n, forces(hi)[0] - axial_load_n
    if f_lo * f_hi > 0:
        raise ValueError("No se pudo calcular la resistencia ideal con la carga axial dada.")
    c = bisect(lambda x: forces(x)[0] - axial_load_n, lo, hi, xtol=1e-9)
    return forces(c)[1]


def strain_profile(result: MomentCurvatureResult, index: int):
    """Deformación en los bordes de la sección en el paso `index`: (y [mm], deformación)."""
    state = result.states[index]
    h = result.model.section.depth_mm
    y = np.array([-h / 2.0, h / 2.0])
    return y, state.strain_at(y)
