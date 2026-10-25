"""Geometría de una sección rectangular de concreto armado y su discretización en láminas.

Sigue la idealización de la sección 5.2 (Soesianawati, 1986) y 7.4.2 (Mander, 1984):
la sección se divide en láminas ("laminae") paralelas al eje neutro; cada
lámina contiene un área de concreto de recubrimiento y un área de concreto del
núcleo confinado; las barras longitudinales se agrupan en capas ("levels").

Convenciones:
    - Unidades: mm y mm².
    - El eje x es el ancho b; el eje y es el peralte h. La flexión es respecto al
      eje x, de modo que las láminas son horizontales.
    - y se mide desde la mitad del peralte, positivo hacia la cara comprimida.
      Como la sección y su armadura son simétricas, la mitad del peralte
      coincide con el centroide plástico (eje de referencia de COLUMN).
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
class TransverseReinforcement:
    """Acero transversal (estribos y ganchos) de un conjunto de estribos.

    bar_diameter_mm: diámetro de la barra del estribo.
    spacing_mm:      espaciamiento centro a centro de los conjuntos de estribos, s.
    yield_stress:    fyh [MPa].
    area_x_mm2:      área total de acero transversal paralelo al eje x que cruza el
                     núcleo, por conjunto de estribos (incluye las ramas del estribo
                     perimetral y de los ganchos suplementarios). Define rho_x.
    area_y_mm2:      ídem paralelo al eje y. Define rho_y.
    supported_bars_top_bottom: barras de la cara superior (e inferior) apoyadas en una
                     esquina de estribo o gancho, contando las de esquina. None = todas.
    supported_bars_sides: ídem para cada cara lateral. None = todas.
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
    """Sección rectangular con barras longitudinales distribuidas en el perímetro.

    width_mm, depth_mm:   b (dirección x) y h (dirección y, peralte de flexión).
    clear_cover_mm:       recubrimiento libre hasta la cara exterior del estribo.
    bars_top_bottom:      barras en la cara superior (e inferior), contando las de esquina (>= 2).
    bars_sides:           barras en cada cara lateral, contando las de esquina (>= 2).
    bar_diameter_mm:      diámetro de las barras longitudinales.
    transverse:           acero transversal.
    """

    width_mm: float
    depth_mm: float
    clear_cover_mm: float
    bars_top_bottom: int
    bars_sides: int
    bar_diameter_mm: float
    transverse: TransverseReinforcement
    # --- se calculan en __post_init__ ---
    core_width_mm: float = field(init=False)
    core_depth_mm: float = field(init=False)

    def __post_init__(self):
        if min(self.width_mm, self.depth_mm, self.bar_diameter_mm) <= 0 or self.clear_cover_mm < 0:
            raise ValueError("Las dimensiones de la sección, el diámetro de barra y el recubrimiento deben ser válidos.")
        if self.bars_top_bottom < 2 or self.bars_sides < 2:
            raise ValueError("Se requieren al menos 2 barras por cara (las de esquina).")
        hoop_centre = self.hoop_centreline_cover_mm
        object.__setattr__(self, "core_width_mm", self.width_mm - 2.0 * hoop_centre)
        object.__setattr__(self, "core_depth_mm", self.depth_mm - 2.0 * hoop_centre)
        if self.core_width_mm <= 0 or self.core_depth_mm <= 0:
            raise ValueError("El recubrimiento es tan grande que no queda núcleo de concreto.")
        if self.bar_centre_offset_mm >= self.depth_mm / 2.0 or self.bar_centre_offset_mm >= self.width_mm / 2.0:
            raise ValueError("Las barras longitudinales no caben dentro de la sección.")
        for count, span, name in (
            (self.bars_top_bottom, 2.0 * self.bar_x_extent_mm, "superior/inferior"),
            (self.bars_sides, 2.0 * self.bar_y_extent_mm, "lateral"),
        ):
            if span / (count - 1) - self.bar_diameter_mm <= 0:
                raise ValueError(f"Las barras de la cara {name} se traslapan: aumente la sección o reduzca el número de barras.")
        for supported, count, name in (
            (self.transverse.supported_bars_top_bottom, self.bars_top_bottom, "superior/inferior"),
            (self.transverse.supported_bars_sides, self.bars_sides, "lateral"),
        ):
            if supported is not None and not 2 <= supported <= count:
                raise ValueError(f"Las barras apoyadas en la cara {name} deben estar entre 2 y {count}.")

    # ----- geometría derivada -----
    @property
    def hoop_centreline_cover_mm(self) -> float:
        """Distancia de la cara de concreto al eje del estribo."""
        return self.clear_cover_mm + self.transverse.bar_diameter_mm / 2.0

    @property
    def bar_centre_offset_mm(self) -> float:
        """Distancia de la cara de concreto al centro de la barra longitudinal."""
        return self.clear_cover_mm + self.transverse.bar_diameter_mm + self.bar_diameter_mm / 2.0

    @property
    def bar_y_extent_mm(self) -> float:
        """Coordenada y del centro de la capa superior de barras (positiva)."""
        return self.depth_mm / 2.0 - self.bar_centre_offset_mm

    @property
    def bar_x_extent_mm(self) -> float:
        return self.width_mm / 2.0 - self.bar_centre_offset_mm

    @property
    def gross_area_mm2(self) -> float:
        return self.width_mm * self.depth_mm

    @property
    def core_area_mm2(self) -> float:
        """Área del núcleo medida al eje del estribo, b'' x d''."""
        return self.core_width_mm * self.core_depth_mm

    @property
    def number_of_bars(self) -> int:
        return 2 * self.bars_top_bottom + 2 * (self.bars_sides - 2)

    @property
    def steel_area_mm2(self) -> float:
        return self.number_of_bars * bar_area(self.bar_diameter_mm)

    @property
    def longitudinal_ratio_core(self) -> float:
        """rho_cc: acero longitudinal / área del núcleo."""
        return self.steel_area_mm2 / self.core_area_mm2

    def bar_layers(self):
        """Capas de barras: arreglos (y, número de barras, área por capa).

        Las capas se ordenan de abajo (y < 0) hacia arriba. La capa superior e
        inferior tienen bars_top_bottom barras; las intermedias, 2 barras.
        """
        n_layers = self.bars_sides
        y = np.linspace(-self.bar_y_extent_mm, self.bar_y_extent_mm, n_layers)
        counts = np.full(n_layers, 2, dtype=int)
        counts[0] = counts[-1] = self.bars_top_bottom
        return y, counts, counts * bar_area(self.bar_diameter_mm)

    def sum_squared_clear_spacings(self) -> float:
        """Suma de (w')² de las barras apoyadas por estribos/ganchos (Ec. 3.16).

        w' es la distancia libre entre barras apoyadas consecutivas de una misma
        cara; se supone que las barras apoyadas están equiespaciadas.
        """
        tb = self.transverse.supported_bars_top_bottom or self.bars_top_bottom
        sd = self.transverse.supported_bars_sides or self.bars_sides
        w_tb = 2.0 * self.bar_x_extent_mm / (tb - 1) - self.bar_diameter_mm
        w_sd = 2.0 * self.bar_y_extent_mm / (sd - 1) - self.bar_diameter_mm
        return 2.0 * (tb - 1) * w_tb**2 + 2.0 * (sd - 1) * w_sd**2


@dataclass(frozen=True)
class SectionMesh:
    """Láminas de concreto y capas de acero listas para integrar.

    Arreglos por lámina: y (centro, desde mitad de peralte), thickness,
    core_area y cover_area [mm²]. El área del núcleo ya descuenta el acero
    longitudinal de su lámina para no contarlo dos veces (igual que COLUMN).
    Arreglos por capa de acero: bar_y y bar_area.
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
    half_bar = section.bar_diameter_mm / 2.0
    for y_bar, a_bar in zip(bar_y, layer_area):
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
