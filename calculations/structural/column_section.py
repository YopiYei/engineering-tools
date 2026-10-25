"""Geometría de una sección rectangular de concreto armado y su discretización en láminas.

Sigue la idealización de la sección 5.2 (Soesianawati, 1986) y 7.4.2 (Mander, 1984):
la sección se divide en láminas ("laminae") paralelas al eje neutro; cada
lámina contiene un área de concreto de recubrimiento y un área de concreto del
núcleo confinado; las barras longitudinales se agrupan en capas ("levels").

Convenciones:
    - Unidades: mm y mm².
    - El eje x es el ancho b; el eje y es el peralte h. La flexión es respecto al
      eje x, de modo que las láminas son horizontales.
    - Las capas de barras se definen por su profundidad: distancia de la cara
      comprimida (superior) al centro de las barras.
    - y se mide desde la mitad del peralte, positivo hacia la cara comprimida. Es el
      centroide de la sección bruta de concreto, que es el eje de referencia de COLUMN
      (listado, variable YFORC); el momento se calcula respecto a ese eje aunque el
      refuerzo no sea simétrico.
    - El núcleo confinado llega hasta el eje del estribo perimetral, igual que
      en COLUMN ("centreline of hoops").
"""

from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np

BAR_AREA_FACTOR = pi / 4.0


def bar_area(diameter_mm: float) -> float:
    """Área de una barra circular [mm²]."""
    return BAR_AREA_FACTOR * diameter_mm**2


@dataclass(frozen=True)
class BarLayer:
    """Una capa de barras longitudinales del mismo diámetro, equiespaciadas a lo ancho.

    depth_mm:    distancia de la cara comprimida al centro de las barras.
    count:       número de barras de la capa (>= 2: una a cada lado).
    diameter_mm: diámetro de las barras.
    """

    depth_mm: float
    count: int
    diameter_mm: float

    def __post_init__(self):
        if self.count != int(self.count) or self.count < 2:
            raise ValueError("Cada capa de barras debe tener al menos 2 barras (número entero).")
        if self.diameter_mm <= 0 or self.depth_mm <= 0:
            raise ValueError("La profundidad y el diámetro de cada capa de barras deben ser positivos.")

    @property
    def area_mm2(self) -> float:
        return self.count * bar_area(self.diameter_mm)


@dataclass(frozen=True)
class TransverseReinforcement:
    """Acero transversal (estribos y ganchos) de un conjunto de estribos.

    bar_diameter_mm: diámetro de la barra del estribo.
    spacing_mm:      espaciamiento centro a centro de los conjuntos de estribos, s.
    yield_stress:    fyh [MPa].
    area_x_mm2:      área total de acero transversal paralelo al eje x que cruza el
                     núcleo, por conjunto de estribos (incluye las ramas del estribo
                     perimetral y de los ganchos suplementarios). Define rho_x.
    area_y_mm2:      ídem paralelo al eje y. Define rho_y.
    supported_bars_top_bottom: barras de la capa superior (e inferior) apoyadas en una
                     esquina de estribo o gancho, contando las de esquina. None = todas.
    supported_bars_sides: barras apoyadas en cada cara lateral (una por capa si todas
                     están apoyadas). None = todas.
    """

    bar_diameter_mm: float
    spacing_mm: float
    yield_stress: float
    area_x_mm2: float
    area_y_mm2: float
    supported_bars_top_bottom: int | None = None
    supported_bars_sides: int | None = None

    def __post_init__(self):
        if min(self.bar_diameter_mm, self.spacing_mm, self.yield_stress) <= 0:
            raise ValueError("El diámetro, el espaciamiento y fyh de los estribos deben ser positivos.")
        if self.area_x_mm2 <= 0 or self.area_y_mm2 <= 0:
            raise ValueError("Las áreas de acero transversal en x e y deben ser positivas.")

    @classmethod
    def from_legs(cls, bar_diameter_mm, spacing_mm, yield_stress, legs_x, legs_y, **kwargs):
        """Define las áreas contando ramas: legs_x ramas paralelas a x y legs_y paralelas a y."""
        if legs_x < 2 or legs_y < 2:
            raise ValueError("Un estribo cerrado tiene al menos 2 ramas en cada dirección.")
        ab = bar_area(bar_diameter_mm)
        return cls(bar_diameter_mm, spacing_mm, yield_stress, legs_x * ab, legs_y * ab, **kwargs)

    @property
    def bar_area_mm2(self) -> float:
        return bar_area(self.bar_diameter_mm)


@dataclass(frozen=True)
class RectangularSection:
    """Sección rectangular con capas de barras longitudinales y estribos.

    width_mm, depth_mm:   b (dirección x) y h (dirección y, peralte de flexión).
    clear_cover_mm:       recubrimiento libre hasta la cara exterior del estribo.
    layers:               capas de barras (BarLayer); se ordenan de arriba hacia abajo.
                          La primera y la última son las capas de las caras superior e inferior.
    transverse:           acero transversal.
    """

    width_mm: float
    depth_mm: float
    clear_cover_mm: float
    layers: tuple
    transverse: TransverseReinforcement
    # --- se calculan en __post_init__ ---
    core_width_mm: float = field(init=False)
    core_depth_mm: float = field(init=False)

    def __post_init__(self):
        if min(self.width_mm, self.depth_mm) <= 0 or self.clear_cover_mm < 0:
            raise ValueError("Las dimensiones de la sección y el recubrimiento deben ser válidos.")
        layers = tuple(sorted(self.layers, key=lambda layer: layer.depth_mm))
        if len(layers) < 2:
            raise ValueError("Se requieren al menos 2 capas de barras (superior e inferior).")
        object.__setattr__(self, "layers", layers)

        hoop_centre = self.hoop_centreline_cover_mm
        object.__setattr__(self, "core_width_mm", self.width_mm - 2.0 * hoop_centre)
        object.__setattr__(self, "core_depth_mm", self.depth_mm - 2.0 * hoop_centre)
        if self.core_width_mm <= 0 or self.core_depth_mm <= 0:
            raise ValueError("El recubrimiento es tan grande que no queda núcleo de concreto.")

        for i, layer in enumerate(layers, start=1):
            offset = self.clear_cover_mm + self.transverse.bar_diameter_mm + layer.diameter_mm / 2.0
            if layer.depth_mm < offset - 1e-9 or layer.depth_mm > self.depth_mm - offset + 1e-9:
                raise ValueError(
                    f"La capa {i} (a {layer.depth_mm:g} mm de la cara comprimida) queda fuera del estribo: "
                    f"debe estar entre {offset:g} y {self.depth_mm - offset:g} mm."
                )
            if self.layer_clear_spacing(layer) <= 0:
                raise ValueError(f"Las barras de la capa {i} se traslapan: reduzca su número o diámetro.")
        for i, (upper, lower) in enumerate(zip(layers[:-1], layers[1:]), start=1):
            if lower.depth_mm - upper.depth_mm - (upper.diameter_mm + lower.diameter_mm) / 2.0 <= 0:
                raise ValueError(f"Las capas {i} y {i + 1} se traslapan: sepárelas más.")

        t = self.transverse
        if t.supported_bars_top_bottom is not None and not 2 <= t.supported_bars_top_bottom <= min(layers[0].count, layers[-1].count):
            raise ValueError(
                f"Las barras apoyadas en la capa superior/inferior deben estar entre 2 y {min(layers[0].count, layers[-1].count)}."
            )
        if t.supported_bars_sides is not None and not 2 <= t.supported_bars_sides <= len(layers):
            raise ValueError(f"Las barras apoyadas en cada cara lateral deben estar entre 2 y {len(layers)}.")

    @classmethod
    def with_perimeter_bars(cls, width_mm, depth_mm, clear_cover_mm, bars_top_bottom, bars_sides, bar_diameter_mm, transverse):
        """Armado perimetral simétrico: capas superior e inferior con `bars_top_bottom` barras y
        `bars_sides - 2` capas intermedias de 2 barras, equiespaciadas."""
        if bars_top_bottom < 2 or bars_sides < 2:
            raise ValueError("Se requieren al menos 2 barras por cara (las de esquina).")
        offset = clear_cover_mm + transverse.bar_diameter_mm + bar_diameter_mm / 2.0
        depths = np.linspace(offset, depth_mm - offset, bars_sides)
        counts = [2] * bars_sides
        counts[0] = counts[-1] = bars_top_bottom
        layers = tuple(BarLayer(float(d), int(n), bar_diameter_mm) for d, n in zip(depths, counts))
        return cls(width_mm, depth_mm, clear_cover_mm, layers, transverse)

    # ----- geometría derivada -----
    @property
    def hoop_centreline_cover_mm(self) -> float:
        """Distancia de la cara de concreto al eje del estribo."""
        return self.clear_cover_mm + self.transverse.bar_diameter_mm / 2.0

    def layer_x_extent(self, layer: BarLayer) -> float:
        """Coordenada x del centro de la barra más externa de la capa (positiva)."""
        return self.width_mm / 2.0 - (self.clear_cover_mm + self.transverse.bar_diameter_mm + layer.diameter_mm / 2.0)

    def layer_clear_spacing(self, layer: BarLayer) -> float:
        """Espaciamiento libre entre barras contiguas de una capa."""
        return 2.0 * self.layer_x_extent(layer) / (layer.count - 1) - layer.diameter_mm

    def layer_x_positions(self, layer: BarLayer) -> np.ndarray:
        """Coordenadas x (desde el centro) de las barras de la capa."""
        extent = self.layer_x_extent(layer)
        return np.linspace(-extent, extent, layer.count)

    @property
    def gross_area_mm2(self) -> float:
        return self.width_mm * self.depth_mm

    @property
    def core_area_mm2(self) -> float:
        """Área del núcleo medida al eje del estribo, b'' x d''."""
        return self.core_width_mm * self.core_depth_mm

    @property
    def number_of_bars(self) -> int:
        return sum(layer.count for layer in self.layers)

    @property
    def steel_area_mm2(self) -> float:
        return sum(layer.area_mm2 for layer in self.layers)

    @property
    def longitudinal_ratio_core(self) -> float:
        """rho_cc: acero longitudinal / área del núcleo."""
        return self.steel_area_mm2 / self.core_area_mm2

    def bar_layers(self):
        """Capas de barras: arreglos (y, número de barras, área por capa).

        Las capas se ordenan de abajo (y < 0) hacia arriba, como las láminas.
        """
        ordered = self.layers[::-1]
        y = np.array([self.depth_mm / 2.0 - layer.depth_mm for layer in ordered])
        counts = np.array([layer.count for layer in ordered], dtype=int)
        areas = np.array([layer.area_mm2 for layer in ordered])
        return y, counts, areas

    def bar_layer_diameters(self) -> np.ndarray:
        """Diámetros de las capas, en el mismo orden que `bar_layers()`."""
        return np.array([layer.diameter_mm for layer in self.layers[::-1]])

    def sum_squared_clear_spacings(self) -> float:
        """Suma de (w')² de las barras apoyadas por estribos/ganchos (Ec. 3.16).

        w' es la distancia libre entre barras apoyadas consecutivas del perímetro.
        Capas superior e inferior: barras equiespaciadas. Caras laterales: si todas
        las capas están apoyadas, w' es la separación libre real entre capas
        consecutivas; si se indica un número de barras apoyadas, se supone una
        distribución uniforme.
        """
        top, bottom = self.layers[0], self.layers[-1]
        t = self.transverse
        total = 0.0
        for layer in (top, bottom):
            supported = t.supported_bars_top_bottom or layer.count
            w = 2.0 * self.layer_x_extent(layer) / (supported - 1) - layer.diameter_mm
            total += (supported - 1) * w**2
        if t.supported_bars_sides is None:
            for upper, lower in zip(self.layers[:-1], self.layers[1:]):
                w = lower.depth_mm - upper.depth_mm - (upper.diameter_mm + lower.diameter_mm) / 2.0
                total += 2.0 * w**2  # una vez por cada cara lateral
        else:
            mean_diameter = float(np.mean([layer.diameter_mm for layer in self.layers]))
            w = (bottom.depth_mm - top.depth_mm) / (t.supported_bars_sides - 1) - mean_diameter
            total += 2.0 * (t.supported_bars_sides - 1) * w**2
        return total


@dataclass(frozen=True)
class SectionMesh:
    """Láminas de concreto y capas de acero listas para integrar.

    Arreglos por lámina: y (centro, desde mitad de peralte), thickness,
    core_area y cover_area [mm²]. El área del núcleo ya descuenta el acero
    longitudinal para no contarlo dos veces (igual que COLUMN).
    Arreglos por capa de acero (de abajo hacia arriba): bar_y y bar_area.
    """

    y: np.ndarray
    thickness: np.ndarray
    core_area: np.ndarray
    cover_area: np.ndarray
    bar_y: np.ndarray
    bar_area: np.ndarray


def build_mesh(section: RectangularSection, number_of_strips: int = 60) -> SectionMesh:
    """Discretiza la sección en láminas respetando el borde del núcleo (eje del estribo).

    Se reparten láminas en tres zonas (recubrimiento inferior, núcleo y
    recubrimiento superior) proporcionalmente a su espesor, con al menos 2
    láminas en cada recubrimiento.
    """
    if number_of_strips < 10:
        raise ValueError("Use al menos 10 láminas.")
    h = section.depth_mm
    hoop = section.hoop_centreline_cover_mm
    n_cover = max(2, round(number_of_strips * hoop / h))
    n_core = number_of_strips - 2 * n_cover
    if n_core < 4:
        raise ValueError("Pocas láminas para el núcleo: aumente el número de láminas.")

    edges = np.concatenate(
        [
            np.linspace(-h / 2.0, -h / 2.0 + hoop, n_cover + 1),
            np.linspace(-h / 2.0 + hoop, h / 2.0 - hoop, n_core + 1)[1:],
            np.linspace(h / 2.0 - hoop, h / 2.0, n_cover + 1)[1:],
        ]
    )
    y = 0.5 * (edges[:-1] + edges[1:])
    thickness = np.diff(edges)

    in_core = np.abs(y) < h / 2.0 - hoop
    core_area = np.where(in_core, section.core_width_mm * thickness, 0.0)
    cover_area = section.width_mm * thickness - core_area

    bar_y, _, layer_area = section.bar_layers()
    # Se descuenta el área de las barras del concreto del núcleo para no contarla dos veces
    # (COLUMN la resta de la lámina que contiene la capa). Aquí se reparte entre las láminas
    # que cubre el diámetro de la barra, en proporción al traslape; con láminas más gruesas
    # que la barra coincide con COLUMN y con mallas finas evita láminas de área negativa.
    for y_bar, a_bar, diameter in zip(bar_y, layer_area, section.bar_layer_diameters()):
        half_bar = diameter / 2.0
        overlap = np.clip(np.minimum(edges[1:], y_bar + half_bar) - np.maximum(edges[:-1], y_bar - half_bar), 0.0, None)
        core_area -= a_bar * overlap / overlap.sum()
    if np.any(core_area[in_core] <= 0.0):
        raise ValueError("El acero longitudinal ocupa toda el área de una lámina: reduzca el número de láminas.")
    return SectionMesh(y, thickness, core_area, cover_area, bar_y, layer_area)


def equivalent_octagonal_area_mm2(bar_diameter_mm: float) -> float:
    """Área transversal por dirección de estribos cuadrados + octagonales superpuestos.

    Soesianawati (1986), Fig. 2.2 y Ec. 5.18: Ash = (2 + sqrt 2) Asb.
    """
    return (2.0 + sqrt(2.0)) * bar_area(bar_diameter_mm)
