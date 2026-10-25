"""Columnas de ensayo de Soesianawati (1986): datos de entrada y resultados publicados.

Fuente: Soesianawati, M. T. (1986), *Limited Ductility Design of Reinforced
Concrete Columns*, Research Report 86-10, Univ. of Canterbury.
    Tabla 2.1             detalles de las Unidades 1 a 4
    Tabla 3.1             resistencias de cilindros y edad de ensayo
    Fig. 5.8    (p. 49)   acero longitudinal HD16
    Tabla 5.1   (p. 45)   f'cc y ecc de Mander
    Tabla 8.3   (p. 140)  Mi y Mmax/Mi de los análisis momento-curvatura
(Números de página impresos; el PDF digital lleva 14 páginas más.)

Sección de todas las unidades: 400 x 400 mm, 12 barras HD16 (4 por cara),
recubrimiento de 13 mm hasta el estribo, conjuntos de estribos cuadrados y
octagonales superpuestos (Fig. 2.2): Ash = (2 + sqrt 2) Asb en cada dirección.
La carga axial es Pe = k f'c Ag con f'c de la Tabla 5.1 (resistencia el día del ensayo).

Los f'co de Mander se obtienen con la Ec. 5.21 de ese informe (efecto de la edad):
f'co = t / (2.5 + 0.93 t) f'c(28 días), eco = 0.002 (4 + 0.85 t)/(2.5 + 0.93 t).
"""

from dataclasses import dataclass

from calculations.structural.column_analysis import AnalysisSettings, ColumnModel, ConcreteProperties, build_column_model
from calculations.structural.column_materials import ReinforcingSteel
from calculations.structural.column_section import RectangularSection, TransverseReinforcement, equivalent_octagonal_area_mm2

SECTION_SIZE_MM = 400.0
CLEAR_COVER_MM = 13.0
LONGITUDINAL_DIAMETER_MM = 16.0
BARS_PER_FACE = 4

# Acero HD16 (Fig. 5.8)
HD16_STEEL = ReinforcingSteel(
    yield_stress=446.0,
    ultimate_stress=702.0,
    elastic_modulus=213411.0,
    hardening_strain=0.0098,
    ultimate_strain=0.098,
    hardening_modulus=8608.0,
)


@dataclass(frozen=True)
class ColumnUnit:
    name: str
    axial_ratio: float  # Pe / (f'c Ag)
    stirrup_diameter_mm: float
    stirrup_yield_mpa: float
    stirrup_spacing_mm: float
    cylinder_strength_test_mpa: float  # f'c el día del ensayo (Tabla 5.1)
    cylinder_strength_28d_mpa: float  # f'c a 28 días (Tabla 3.1)
    age_days: float  # edad de ensayo (Tabla 3.1)
    ideal_moment_knm: float  # Mi, Tabla 8.3 (norma NZS 3101:1982)
    overstrength_mander: float  # Mmax/Mi con el modelo de Mander, análisis monotónico (Tabla 8.3)
    published_fcc_mpa: float  # Tabla 5.1, modelo de Mander
    published_ecc: float  # Tabla 5.1, modelo de Mander

    @property
    def axial_load_n(self) -> float:
        return self.axial_ratio * self.cylinder_strength_test_mpa * SECTION_SIZE_MM**2

    @property
    def age_factor_strength(self) -> float:
        return self.age_days / (2.5 + 0.93 * self.age_days)

    @property
    def unconfined_strength_mpa(self) -> float:
        return self.age_factor_strength * self.cylinder_strength_28d_mpa

    @property
    def strain_at_peak(self) -> float:
        return 0.002 * (4.0 + 0.85 * self.age_days) / (2.5 + 0.93 * self.age_days)


SOESIANAWATI_UNITS = {
    1: ColumnUnit("Unidad 1", 0.1, 7.0, 364.0, 85.0, 46.5, 43.5, 40.0, 302.0, 1.09, 51.44, 0.00358),
    2: ColumnUnit("Unidad 2", 0.3, 8.0, 360.0, 78.0, 44.0, 40.0, 48.0, 405.0, 1.05, 51.20, 0.00457),
    3: ColumnUnit("Unidad 3", 0.3, 7.0, 364.0, 91.0, 44.0, 43.5, 33.0, 406.0, 1.06, 50.74, 0.00358),
    4: ColumnUnit("Unidad 4", 0.3, 6.0, 255.0, 94.0, 40.0, 40.0, 38.0, 383.0, 1.06, 44.02, 0.00284),
}


def unit_section(unit: ColumnUnit) -> RectangularSection:
    area = equivalent_octagonal_area_mm2(unit.stirrup_diameter_mm)
    transverse = TransverseReinforcement(unit.stirrup_diameter_mm, unit.stirrup_spacing_mm, unit.stirrup_yield_mpa, area, area)
    return RectangularSection.with_perimeter_bars(
        SECTION_SIZE_MM, SECTION_SIZE_MM, CLEAR_COVER_MM, BARS_PER_FACE, BARS_PER_FACE, LONGITUDINAL_DIAMETER_MM, transverse
    )


def unit_model(unit: ColumnUnit, spall_strain: float = 0.008, number_of_strips: int = 60) -> ColumnModel:
    """Modelo de una unidad de ensayo con los datos publicados."""
    concrete = ConcreteProperties(unit.unconfined_strength_mpa, unit.strain_at_peak, spall_strain)
    return build_column_model(unit_section(unit), concrete, HD16_STEEL, number_of_strips=number_of_strips)


def unit_settings(unit: ColumnUnit, **overrides) -> AnalysisSettings:
    return AnalysisSettings(axial_load_n=unit.axial_load_n, **overrides)
