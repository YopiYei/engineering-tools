"""Enlace entre el panel «Concrete Stress-Strain» y el análisis COLUMN.

El panel de curvas del concreto publica el f'c y los modelos de compresión y de tracción que el
usuario eligió (vista de un solo f'c). Este módulo traduce esa elección a los datos del concreto no
confinado de COLUMN, tomándolos de las curvas del panel y no de valores supuestos:

    f'co  = f'c del panel
    eco   = deformación en el esfuerzo máximo de la curva de compresión del modelo elegido
    Ec    = módulo de elasticidad del modelo elegido
    f't   = esfuerzo máximo de la curva de tracción del modelo elegido

COLUMN sigue usando su propia ley (Popovics con confinamiento de Mander); la curva del panel se
conserva para compararla con la de COLUMN.
"""

from dataclasses import dataclass

import numpy as np

from calculations.structural import concrete_properties as panel

COMPRESSION_MODELS = {
    "collins_mitchell": ("Collins & Mitchell", panel.concrete_collins_mitchell),
    "de_nicolo": ("De Nicolo, Pani y Pozzo", panel.concrete_de_nicolo_pani_pozzo),
    "kent_park": ("Kent & Park", panel.concrete_kent_park),
    "desayi_krishnan": ("Desayi & Krishnan", panel.concrete_desayi_krishnan),
}
TENSION_MODELS = {
    "collins_mitchell": "Collins & Mitchell",
    "restrepo_rodriguez": "Restrepo-Rodríguez",
    "fields_bischoff": "Fields & Bischoff",
}
CURVE_POINTS = 400


@dataclass(frozen=True)
class PanelConcrete:
    """Datos del concreto tomados del panel «Concrete Stress-Strain»."""

    unconfined_strength: float  # f'c [MPa]
    strain_at_peak: float  # deformación en el esfuerzo máximo de compresión
    elastic_modulus: float  # Ec [MPa]
    tensile_strength: float  # esfuerzo máximo de la curva de tracción [MPa]
    compression_label: str
    tension_label: str
    compression_strain: np.ndarray  # curva de compresión del panel
    compression_stress: np.ndarray

    @property
    def description(self) -> str:
        return (f"f'c = {self.unconfined_strength:g} MPa; compresión: {self.compression_label}; tracción: {self.tension_label}")


def concrete_from_panel(fc_mpa: float, compression_model: str, tension_model: str) -> PanelConcrete:
    """Datos de COLUMN a partir de f'c y de los modelos elegidos en el panel de curvas del concreto."""
    if compression_model not in COMPRESSION_MODELS:
        raise ValueError(f"Modelo de compresión desconocido: {compression_model}")
    if tension_model not in TENSION_MODELS:
        raise ValueError(f"Modelo de tracción desconocido: {tension_model}")
    label, parameters = COMPRESSION_MODELS[compression_model]
    curves = panel.generate_curves(fc_mpa, compression_model, tension_model, num_points=CURVE_POINTS)
    strain, stress = np.asarray(curves["strain_comp"], dtype=float), np.asarray(curves["stress_comp"], dtype=float)
    return PanelConcrete(
        unconfined_strength=float(fc_mpa),
        strain_at_peak=float(strain[int(np.argmax(stress))]),
        elastic_modulus=float(parameters(fc_mpa)["Ec"]),
        tensile_strength=float(np.max(curves["stress_tens"])),
        compression_label=label,
        tension_label=TENSION_MODELS[tension_model],
        compression_strain=strain,
        compression_stress=stress,
    )
