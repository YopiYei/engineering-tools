"""Diagrama de interacción P-M y ductilidad en función de la carga axial.

Dos curvas complementarias:

1. Curva ideal (método de la norma): para cada profundidad de eje neutro c se calcula el
   par (N, M) con deformación 0.003 en la fibra extrema y bloque rectangular de esfuerzos
   (`column_analysis.code_forces`). Es rápida y da los puntos notables: tracción pura,
   punto balanceado y compresión pura. Es la misma resistencia ideal Mi que usa COLUMN.
2. Curvas de COLUMN: para cada carga axial P se ejecuta el análisis momento-curvatura
   completo y se extraen el momento máximo, la curvatura de fluencia y la ductilidad.

Convención: compresión positiva; N, mm, MPa; momento respecto a la mitad del peralte.
"""

from dataclasses import dataclass, replace

import numpy as np

from calculations.structural.column_analysis import (
    CODE_ULTIMATE_STRAIN,
    CODE_STEEL_MODULUS,
    AnalysisSettings,
    ColumnModel,
    MomentCurvatureAnalysis,
    code_forces,
)


@dataclass(frozen=True)
class CodeInteractionCurve:
    """Curva ideal N-M y sus puntos notables (unidades: kN y kN·m)."""

    axial_kn: np.ndarray
    moment_knm: np.ndarray
    tension_point: tuple  # (N, M) con c -> 0: toda la armadura en tracción a fy
    balanced_point: tuple  # (N, M) con fluencia de la capa inferior y 0.003 arriba
    compression_point: tuple  # (N, M) con c -> infinito: compresión pura

    @property
    def maximum_moment_knm(self) -> float:
        return float(self.moment_knm.max())


def code_interaction_curve(model: ColumnModel, number_of_points: int = 120) -> CodeInteractionCurve:
    """Curva N-M ideal barriendo la profundidad del eje neutro de muy pequeña a muy grande."""
    if number_of_points < 10:
        raise ValueError("Use al menos 10 puntos para la curva ideal.")
    h = model.section.depth_mm
    c_values = np.geomspace(1e-3 * h, 1e3 * h, number_of_points)
    pairs = np.array([code_forces(model, c) for c in c_values])
    # El punto balanceado: la capa inferior fluye (e = fy/Es) cuando la fibra superior llega a 0.003
    bottom_depth = model.section.layers[-1].depth_mm
    yield_strain = model.steel.yield_stress / CODE_STEEL_MODULUS
    c_balanced = CODE_ULTIMATE_STRAIN * bottom_depth / (CODE_ULTIMATE_STRAIN + yield_strain)
    tension = code_forces(model, 1e-9 * h)
    compression = code_forces(model, 1e6 * h)
    balanced = code_forces(model, c_balanced)
    return CodeInteractionCurve(
        axial_kn=pairs[:, 0] / 1e3,
        moment_knm=pairs[:, 1] / 1e6,
        tension_point=(tension[0] / 1e3, tension[1] / 1e6),
        balanced_point=(balanced[0] / 1e3, balanced[1] / 1e6),
        compression_point=(compression[0] / 1e3, compression[1] / 1e6),
    )


@dataclass(frozen=True)
class InteractionPoint:
    """Resultado del análisis completo para una carga axial."""

    axial_load_n: float
    ideal_moment: float | None = None  # Mi [N*mm]
    max_moment: float | None = None  # Mmax [N*mm]
    curvature_at_max: float | None = None  # [1/mm]
    first_yield_curvature: float | None = None  # phi'y [1/mm]
    yield_curvature: float | None = None  # phi_y [1/mm]
    ultimate_curvature: float | None = None  # phi al terminar [1/mm]
    ductility: float | None = None  # phi_u / phi_y
    termination_key: str = ""
    termination_label: str = ""
    error: str = ""


def default_axial_range(model: ColumnModel, settings: AnalysisSettings, fraction: float = 0.8):
    """Rango de carga axial [N] para el barrido: de 0 a `fraction` de la capacidad axial del modelo."""
    capacity = MomentCurvatureAnalysis(model, replace(settings, axial_load_n=0.0)).axial_capacity
    return 0.0, fraction * capacity


def analysis_interaction(model: ColumnModel, settings: AnalysisSettings, axial_loads, progress=None) -> list:
    """Ejecuta el análisis momento-curvatura para cada carga axial de `axial_loads` [N].

    Un fallo en una carga (por ejemplo, fuera de la capacidad) no detiene el barrido: ese punto
    queda con el mensaje en `error`. `progress(i, n)` se llama después de cada punto si se entrega.
    """
    points = []
    loads = list(axial_loads)
    for i, load in enumerate(loads, start=1):
        try:
            result = MomentCurvatureAnalysis(model, replace(settings, axial_load_n=float(load))).run()
            maximum = result.maximum
            final = result.states[-1]
            points.append(InteractionPoint(
                axial_load_n=float(load), ideal_moment=result.ideal_moment, max_moment=maximum.moment,
                curvature_at_max=maximum.phi, first_yield_curvature=result.yield_curvature_first,
                yield_curvature=result.yield_curvature, ultimate_curvature=final.phi,
                ductility=result.ductility_of(final.phi), termination_key=result.termination_key,
                termination_label=result.termination_message,
            ))
        except ValueError as error:
            points.append(InteractionPoint(axial_load_n=float(load), error=str(error)))
        if progress is not None:
            progress(i, len(loads))
    return points


def interaction_dataframe(points, model: ColumnModel):
    """Tabla del barrido. Fuerza en kN, momento en kN·m, curvatura en µrad/mm."""
    import pandas as pd

    gross = model.concrete.unconfined_strength * model.section.gross_area_mm2
    scale = lambda value, factor: None if value is None else value * factor
    return pd.DataFrame([
        {
            "axial_kN": p.axial_load_n / 1e3,
            "axial_ratio": p.axial_load_n / gross,
            "Mi_kNm": scale(p.ideal_moment, 1e-6),
            "Mmax_kNm": scale(p.max_moment, 1e-6),
            "curvature_at_Mmax_urad_per_mm": scale(p.curvature_at_max, 1e6),
            "phi_y_urad_per_mm": scale(p.yield_curvature, 1e6),
            "phi_u_urad_per_mm": scale(p.ultimate_curvature, 1e6),
            "ductility": p.ductility,
            "termination": p.termination_label,
            "error": p.error,
        }
        for p in points
    ])
