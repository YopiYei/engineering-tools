"""Corte de la sección con las zonas confinadas y no confinadas de Mander (1988), Fig. 3.

Sin Streamlit. Longitudes en mm. Sobre el corte acotado de `column_plots.section_figure` se
dibujan las zonas de concreto del núcleo que el arqueo no alcanza a confinar: entre dos barras
apoyadas consecutivas, el concreto queda sin confinar dentro de una parábola de segundo grado con
tangente inicial a 45°, de ancho w' (distancia libre entre barras) y profundidad w'/4, cuya área es w'²/6.
"""

import numpy as np
import plotly.graph_objects as go

from calculations.structural.column_section import RectangularSection
from visualization.column_plots import section_figure

INEFFECTIVE_ZONE_COLOR = "rgba(255, 127, 14, 0.75)"
ARCH_POINTS = 25


def _arch_polygon(start: float, end: float, face: float, inward: float, along_x: bool):
    """Parábola de arqueo entre dos barras de una cara del estribo.

    start, end: posición de los bordes de las barras a lo largo de la cara.
    face:       coordenada de la cara (eje del estribo) en la dirección perpendicular.
    inward:     +1 o -1, sentido hacia el interior del núcleo.
    along_x:    True si la cara es horizontal (superior o inferior).
    """
    clear_spacing = end - start
    along = np.linspace(start, end, ARCH_POINTS)
    depth = (along - start) * (1.0 - (along - start) / clear_spacing)  # 45° en los extremos, w'/4 al centro
    across = face + inward * depth
    return (along, across) if along_x else (across, along)


def _add_ineffective_zone(fig: go.Figure, xs, ys, show_legend: bool):
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=INEFFECTIVE_ZONE_COLOR,
        line=dict(color="rgb(200, 90, 0)", width=1), name="Núcleo no confinado (arqueo entre barras)",
        legendgroup="arching", showlegend=show_legend, hoverinfo="skip",
    ))


def confinement_section_figure(section: RectangularSection, legs_x: int = 2, legs_y: int = 2) -> go.Figure:
    """Corte acotado con las parábolas de concreto no confinado entre las barras longitudinales."""
    fig = section_figure(section, legs_x=legs_x, legs_y=legs_y)
    width, depth = section.width_mm, section.depth_mm
    hoop_centreline = section.hoop_centreline_cover_mm
    top_face_y, bottom_face_y = depth - hoop_centreline, hoop_centreline
    left_face_x, right_face_x = hoop_centreline, width - hoop_centreline

    show_legend = True

    # Caras superior e inferior: entre barras contiguas de la misma capa
    for layer, face_y, inward in ((section.layers[0], top_face_y, -1.0), (section.layers[-1], bottom_face_y, +1.0)):
        bar_x = width / 2.0 + section.layer_x_positions(layer)
        bar_radius = layer.diameter_mm / 2.0
        for left_bar_x, right_bar_x in zip(bar_x[:-1], bar_x[1:]):
            xs, ys = _arch_polygon(left_bar_x + bar_radius, right_bar_x - bar_radius, face_y, inward, along_x=True)
            _add_ineffective_zone(fig, xs, ys, show_legend)
            show_legend = False

    # Caras laterales: entre capas consecutivas de barras (una vez por cada cara)
    for upper_layer, lower_layer in zip(section.layers[:-1], section.layers[1:]):
        upper_bar_bottom = depth - upper_layer.depth_mm - upper_layer.diameter_mm / 2.0
        lower_bar_top = depth - lower_layer.depth_mm + lower_layer.diameter_mm / 2.0
        for face_x, inward in ((left_face_x, +1.0), (right_face_x, -1.0)):
            xs, ys = _arch_polygon(lower_bar_top, upper_bar_bottom, face_x, inward, along_x=False)
            _add_ineffective_zone(fig, xs, ys, show_legend)
            show_legend = False

    return fig
