"""Comparación de variantes de una misma columna (estribos, carga axial, resistencia del concreto).

Una variante cambia algunos parámetros de la configuración base (los que se dejan en None
quedan iguales) y se analiza con los mismos criterios de terminación. Sirve para ver, por ejemplo,
cuánto aumenta la ductilidad al reducir el espaciamiento de los estribos.

Un error en una variante (por ejemplo, un estribo tan grueso que las barras ya no caben) no
detiene a las demás: se guarda el mensaje en `VariantResult.error`.
"""

from dataclasses import dataclass, replace

from calculations.structural.column_analysis import (
    AnalysisSettings,
    ColumnModel,
    MomentCurvatureAnalysis,
    MomentCurvatureResult,
    build_column_model,
)
from calculations.structural.column_section import BarLayer, RectangularSection, TransverseReinforcement


@dataclass(frozen=True)
class Variant:
    """Cambios respecto a la configuración base. None = igual a la base."""

    name: str
    stirrup_spacing_mm: float | None = None
    stirrup_diameter_mm: float | None = None
    stirrup_yield_mpa: float | None = None
    axial_load_n: float | None = None
    unconfined_strength: float | None = None


@dataclass
class VariantResult:
    variant: Variant
    model: ColumnModel | None = None
    result: MomentCurvatureResult | None = None
    error: str = ""


def scaled_transverse(base: TransverseReinforcement, variant: Variant) -> TransverseReinforcement:
    """Acero transversal de la variante: al cambiar el diámetro, las áreas cambian con (d/d0)²."""
    diameter = variant.stirrup_diameter_mm or base.bar_diameter_mm
    factor = (diameter / base.bar_diameter_mm) ** 2
    return replace(
        base,
        bar_diameter_mm=diameter,
        spacing_mm=variant.stirrup_spacing_mm or base.spacing_mm,
        yield_stress=variant.stirrup_yield_mpa or base.yield_stress,
        area_x_mm2=base.area_x_mm2 * factor,
        area_y_mm2=base.area_y_mm2 * factor,
    )


def shifted_layers(section: RectangularSection, new_stirrup_diameter_mm: float) -> tuple:
    """Capas de barras para otro diámetro de estribo, conservando su distancia al estribo.

    Si el estribo engruesa en d, la capa superior baja d y la inferior sube d (las barras siguen
    pegadas al estribo); las capas intermedias se reparten proporcionalmente entre ambas.
    """
    delta = new_stirrup_diameter_mm - section.transverse.bar_diameter_mm
    top, bottom = section.layers[0].depth_mm, section.layers[-1].depth_mm
    new_top, new_bottom = top + delta, bottom - delta
    scale = (new_bottom - new_top) / (bottom - top)
    return tuple(BarLayer(new_top + (layer.depth_mm - top) * scale, layer.count, layer.diameter_mm) for layer in section.layers)


def run_variants(
    base_model: ColumnModel,
    base_settings: AnalysisSettings,
    variants,
    recompute_confinement: bool = True,
) -> list:
    """Analiza cada variante. Con `recompute_confinement=False` se conserva el f'cc y el ecc de la base
    (cuando fueron ingresados a mano, los estribos de la variante no los modifican)."""
    strips = len(base_model.mesh.y)
    outcomes = []
    for variant in variants:
        try:
            transverse = scaled_transverse(base_model.section.transverse, variant)
            layers = base_model.section.layers
            if transverse.bar_diameter_mm != base_model.section.transverse.bar_diameter_mm:
                layers = shifted_layers(base_model.section, transverse.bar_diameter_mm)
            section = replace(base_model.section, layers=layers, transverse=transverse)
            concrete = base_model.concrete
            if variant.unconfined_strength:
                concrete = replace(concrete, unconfined_strength=variant.unconfined_strength)
            confinement = None if recompute_confinement else base_model.confinement
            model = build_column_model(section, concrete, base_model.steel, confinement, number_of_strips=strips)
            axial = base_settings.axial_load_n if variant.axial_load_n is None else variant.axial_load_n
            result = MomentCurvatureAnalysis(model, replace(base_settings, axial_load_n=axial)).run()
            outcomes.append(VariantResult(variant, model, result))
        except ValueError as error:
            outcomes.append(VariantResult(variant, error=str(error)))
    return outcomes


def variants_dataframe(outcomes):
    """Resumen comparativo. Momento en kN·m, curvatura en µrad/mm, fuerza en kN."""
    import pandas as pd

    rows = []
    for outcome in outcomes:
        row = {"name": outcome.variant.name, "error": outcome.error}
        if outcome.result is not None:
            result, model = outcome.result, outcome.model
            final = result.states[-1]
            row.update({
                "spacing_mm": model.section.transverse.spacing_mm,
                "stirrup_diameter_mm": model.section.transverse.bar_diameter_mm,
                "stirrup_fy_mpa": model.section.transverse.yield_stress,
                "axial_kN": result.settings.axial_load_n / 1e3,
                "fco_mpa": model.concrete.unconfined_strength,
                "ke": model.confinement.effectiveness,
                "fcc_mpa": model.confinement.confined_strength,
                "ecc": model.confinement.confined_strain,
                "Mi_kNm": result.ideal_moment / 1e6,
                "Mmax_kNm": result.maximum.moment / 1e6,
                "phi_y_urad_per_mm": None if result.yield_curvature is None else result.yield_curvature * 1e6,
                "phi_u_urad_per_mm": final.phi * 1e6,
                "ductility": result.ductility_of(final.phi),
                "termination": result.termination_message,
            })
        rows.append(row)
    return pd.DataFrame(rows)
