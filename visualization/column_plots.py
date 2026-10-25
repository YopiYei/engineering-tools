"""Gráficos de Plotly del análisis momento-curvatura tipo COLUMN.

Cada función recibe un resultado de `column_analysis` y devuelve una figura; no usa Streamlit.
Unidades de los gráficos: curvatura en µrad/mm (como en la tesis de Mander, 1 µrad/mm = 1e-3 1/m),
momento en kN·m, deformaciones adimensionales y longitudes en mm.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from calculations.structural.column_analysis import ColumnModel, MomentCurvatureResult

PHI_UNIT = 1e6  # 1/mm -> µrad/mm
MOMENT_UNIT = 1e6  # N*mm -> kN·m

# Marcador de cada estado límite (color y símbolo); los terminales usan una X.
LIMIT_STYLE = {
    "cover_peak": ("#8c564b", "circle"),
    "first_yield": ("#d62728", "diamond"),
    "code_strain": ("#7f7f7f", "square"),
    "max_moment": ("#2ca02c", "star"),
    "core_peak": ("#9467bd", "circle"),
    "steel_hardening": ("#ff7f0e", "triangle-up"),
    "cover_spalling": ("#bcbd22", "triangle-down"),
    "core_ultimate": ("#000000", "x"),
    "hoop_fracture": ("#000000", "x"),
    "steel_ultimate": ("#000000", "x"),
    "ductility_limit": ("#000000", "x"),
    "moment_drop": ("#000000", "x"),
}


def _layout(fig, title, xaxis, yaxis, height=480):
    fig.update_layout(template="plotly_white", title=title, xaxis_title=xaxis, yaxis_title=yaxis, height=height,
                      hovermode="closest", legend=dict(orientation="h", y=-0.25))
    return fig


def moment_curvature_figure(result: MomentCurvatureResult, normalized: bool = False) -> go.Figure:
    """Curva M-phi con los estados límite alcanzados. `normalized` usa phi/phi_y y M/Mi."""
    phi = result.curvature * PHI_UNIT
    moment = result.moment / MOMENT_UNIT
    x_scale = 1.0
    m_scale = 1.0
    if normalized:
        if not result.yield_curvature:
            raise ValueError("La curva normalizada requiere phi_y, que no está definida en este análisis.")
        x_scale = 1.0 / (result.yield_curvature * PHI_UNIT)
        m_scale = 1.0 / (result.ideal_moment / MOMENT_UNIT)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi * x_scale, y=moment * m_scale, mode="lines", name="M-φ monotónica",
                             line=dict(color="#1f77b4", width=3)))
    for record in result.limit_states:
        if not record.reached or record.curvature is None:
            continue
        color, symbol = LIMIT_STYLE.get(record.key, ("#333333", "circle"))
        fig.add_trace(go.Scatter(
            x=[record.curvature * PHI_UNIT * x_scale], y=[record.moment / MOMENT_UNIT * m_scale], mode="markers", name=record.label,
            marker=dict(color=color, symbol=symbol, size=12, line=dict(width=1, color="white")),
            hovertemplate=f"<b>{record.label}</b><br>{record.definition}<br>φ=%{{x:.4g}}, M=%{{y:.4g}}<extra></extra>"))
    # Referencias: resistencia ideal Mi y curvatura de fluencia phi_y
    mi = result.ideal_moment / MOMENT_UNIT * m_scale
    fig.add_hline(y=mi, line=dict(color="gray", dash="dot"), annotation_text="Mi (método de la norma)", annotation_position="bottom right")
    if result.yield_curvature:
        fig.add_vline(x=result.yield_curvature * PHI_UNIT * x_scale, line=dict(color="gray", dash="dot"),
                      annotation_text="φy", annotation_position="top")
    if normalized:
        return _layout(fig, "Diagrama momento–curvatura normalizado", "Ductilidad de curvatura, φ/φy", "M / Mi")
    return _layout(fig, "Diagrama momento–curvatura", "Curvatura, φ (µrad/mm = 10⁻³ 1/m)", "Momento, M (kN·m)")


def material_curves_figure(model: ColumnModel, concrete_strain_max: float | None = None) -> go.Figure:
    """Curvas esfuerzo-deformación: concreto del recubrimiento y del núcleo (izquierda) y acero (derecha)."""
    if concrete_strain_max is None:
        concrete_strain_max = max(4.0 * model.core.peak_strain, 1.2 * model.cover.spall_strain, 0.015)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Concreto (compresión positiva)", "Acero longitudinal"))
    e = np.linspace(0.0, concrete_strain_max, 400)
    cover = model.cover.stress_and_tangent(e)[0]
    core = model.core.stress_and_tangent(e)[0]
    fig.add_trace(go.Scatter(x=e, y=cover, name="Recubrimiento (no confinado)", line=dict(color="#8c564b", width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=e, y=core, name="Núcleo confinado", line=dict(color="#1f77b4", width=3)), 1, 1)
    for strain, stress, text, color in (
        (model.cover.peak_strain, model.cover.peak_stress, "f'co, εco", "#8c564b"),
        (model.core.peak_strain, model.core.peak_stress, "f'cc, εcc", "#1f77b4"),
        (model.cover.spall_strain, 0.0, "εspall", "#8c564b"),
    ):
        fig.add_trace(go.Scatter(x=[strain], y=[stress], mode="markers+text", text=[text], textposition="top center",
                                 marker=dict(color=color, size=9), showlegend=False), 1, 1)
    steel = model.steel
    s = np.linspace(0.0, steel.ultimate_strain, 500)
    fig.add_trace(go.Scatter(x=s, y=steel.stress_and_tangent(s)[0], name="Acero (tracción = compresión)",
                             line=dict(color="#d62728", width=3)), 1, 2)
    for strain, stress, text in ((steel.yield_strain, steel.yield_stress, "fy, εy"),
                                 (steel.hardening_strain, steel.yield_stress, "εsh"),
                                 (steel.ultimate_strain, steel.ultimate_stress, "fsu, εsu")):
        fig.add_trace(go.Scatter(x=[strain], y=[stress], mode="markers+text", text=[text], textposition="bottom right",
                                 marker=dict(color="#d62728", size=9), showlegend=False), 1, 2)
    fig.update_xaxes(title_text="Deformación, ε", row=1, col=1)
    fig.update_xaxes(title_text="Deformación, ε", row=1, col=2)
    fig.update_yaxes(title_text="Esfuerzo, f (MPa)", row=1, col=1)
    fig.update_yaxes(title_text="Esfuerzo, f (MPa)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=460, legend=dict(orientation="h", y=-0.25))
    return fig


def strain_evolution_figure(result: MomentCurvatureResult) -> go.Figure:
    """Deformaciones de las fibras de referencia en función de la curvatura."""
    table = result.to_dataframe()
    phi = table["curvature_urad_per_mm"]
    model = result.model
    fig = go.Figure()
    for column, name, color in (
        ("strain_extreme_cover", "Concreto, fibra extrema (recubrimiento)", "#8c564b"),
        ("strain_core_edge", "Concreto, borde del núcleo", "#1f77b4"),
        ("strain_bar_compression", "Barra comprimida", "#2ca02c"),
        ("strain_bar_tension", "Barra traccionada", "#d62728"),
    ):
        fig.add_trace(go.Scatter(x=phi, y=table[column], mode="lines", name=name, line=dict(color=color, width=2.5)))
    for value, text in ((model.concrete.strain_at_peak, "εco"), (model.core.peak_strain, "εcc"),
                        (model.steel.yield_strain, "εy"), (-model.steel.yield_strain, "-εy")):
        fig.add_hline(y=value, line=dict(color="lightgray", dash="dot"), annotation_text=text, annotation_position="right")
    fig.add_hline(y=0.0, line=dict(color="black", width=1))
    return _layout(fig, "Evolución de las deformaciones (compresión positiva)", "Curvatura, φ (µrad/mm)", "Deformación, ε")


def neutral_axis_figure(result: MomentCurvatureResult) -> go.Figure:
    """Profundidad del eje neutro medida desde la cara comprimida."""
    table = result.to_dataframe().iloc[1:]
    h = result.model.section.depth_mm
    fig = go.Figure(go.Scatter(x=table["curvature_urad_per_mm"], y=table["neutral_axis_depth_mm"], mode="lines",
                               name="Eje neutro, c", line=dict(color="#1f77b4", width=3)))
    fig.add_hline(y=h, line=dict(color="gray", dash="dot"), annotation_text="h (sección totalmente comprimida si c > h)",
                  annotation_position="top left")
    fig.add_hline(y=h / 2.0, line=dict(color="lightgray", dash="dot"), annotation_text="h/2", annotation_position="bottom left")
    return _layout(fig, "Evolución del eje neutro", "Curvatura, φ (µrad/mm)", "Profundidad del eje neutro, c (mm)")


def convergence_figure(result: MomentCurvatureResult) -> go.Figure:
    """Iteraciones de Newton y error de equilibrio axial de cada incremento."""
    table = result.to_dataframe()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Iteraciones por incremento", "Error de equilibrio |N − P| (kN)"))
    colors = np.where(table["converged"], "#1f77b4", "#d62728")
    fig.add_trace(go.Bar(x=table["step"], y=table["iterations"], marker_color=colors, showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=table["step"], y=np.maximum(table["axial_residual_kN"].abs(), 1e-12), mode="lines+markers",
                             showlegend=False, line=dict(color="#2ca02c")), 1, 2)
    tolerance = result.settings.equilibrium_tolerance * result.model.concrete.unconfined_strength * result.model.section.gross_area_mm2 / 1e3
    fig.add_hline(y=tolerance, line=dict(color="red", dash="dash"), annotation_text="tolerancia", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="Incremento", row=1, col=1)
    fig.update_xaxes(title_text="Incremento", row=1, col=2)
    fig.update_layout(template="plotly_white", height=400)
    return fig


def strain_profile_figure(result: MomentCurvatureResult, index: int) -> go.Figure:
    """Perfil de deformaciones sobre el peralte en el incremento `index`, con barras y borde del núcleo."""
    state = result.states[index]
    section, mesh = result.model.section, result.model.mesh
    h = section.depth_mm
    y = np.array([-h / 2.0, h / 2.0])
    fig = go.Figure(go.Scatter(x=state.strain_at(y), y=y, mode="lines", name="Perfil de deformaciones", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=state.eps_bars, y=mesh.bar_y, mode="markers", name="Capas de barras",
                             marker=dict(color="#d62728", size=10, symbol="diamond")))
    for edge, text in ((section.core_depth_mm / 2.0, "borde del núcleo"), (-section.core_depth_mm / 2.0, "borde del núcleo")):
        fig.add_hline(y=edge, line=dict(color="lightgray", dash="dot"), annotation_text=text, annotation_position="right")
    fig.add_vline(x=0.0, line=dict(color="black", width=1))
    return _layout(fig, f"Perfil de deformaciones, φ = {state.phi * PHI_UNIT:.2f} µrad/mm", "Deformación, ε (compresión positiva)",
                   "Posición y desde la mitad del peralte (mm)", height=460)


# =========================================================================================
# Corte de la sección (concreto, núcleo, barras longitudinales, estribos y cotas)
# =========================================================================================

CONCRETE_COLOR = "#d9d9d9"
CORE_COLOR = "#b9d3ee"
STIRRUP_COLOR = "#1f3a93"
BAR_COLOR = "#a01c1c"
DIMENSION_COLOR = "#9aa0a6"  # gris medio: legible en tema claro y oscuro


def _dimension(fig, start, end, text, text_shift=(0.0, 0.0), text_angle=0):
    """Cota: flecha de doble punta entre dos puntos y su texto (coordenadas de datos)."""
    fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowside="end+start",
                       arrowcolor=DIMENSION_COLOR, text="")
    middle = ((start[0] + end[0]) / 2.0 + text_shift[0], (start[1] + end[1]) / 2.0 + text_shift[1])
    fig.add_annotation(x=middle[0], y=middle[1], text=text, showarrow=False, textangle=text_angle,
                       font=dict(size=12, color=DIMENSION_COLOR))


def section_figure(section, octagonal: bool = False, legs_x: int = 2, legs_y: int = 2) -> go.Figure:
    """Corte acotado de la sección: concreto, núcleo confinado, estribos y barras longitudinales.

    Se dibuja a escala con la cara comprimida arriba. `octagonal` agrega el estribo octagonal del
    arreglo cuadrado + octagonal; `legs_x` y `legs_y` agregan ganchos suplementarios (ramas
    adicionales a las 2 del estribo perimetral) cuando el acero transversal se define por ramas.
    """
    b, h = section.width_mm, section.depth_mm
    ds = section.transverse.bar_diameter_mm
    tc = section.hoop_centreline_cover_mm  # eje del estribo
    unit = max(b, h) * 0.075  # separación base de las cotas
    # Espesor de las líneas de estribo en píxeles, proporcional al diámetro real (unos 430 px para h)
    stirrup_width = max(2.0, ds * 430.0 / h)

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=h, fillcolor=CONCRETE_COLOR, line=dict(color="#555555", width=2), layer="below")
    fig.add_shape(type="rect", x0=tc, y0=tc, x1=b - tc, y1=h - tc, fillcolor=CORE_COLOR, line=dict(width=0), layer="below")

    def stirrup(xs, ys):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=STIRRUP_COLOR, width=stirrup_width),
                                 showlegend=False, hoverinfo="skip"))

    # Estribo perimetral al eje del estribo
    stirrup([tc, b - tc, b - tc, tc, tc], [tc, tc, h - tc, h - tc, tc])
    core_w, core_d = b - 2 * tc, h - 2 * tc
    if octagonal:  # octágono inscrito en el núcleo
        cut = min(core_w, core_d) / (2.0 + 2.0**0.5)
        x0, x1, y0, y1 = tc, b - tc, tc, h - tc
        stirrup([x0 + cut, x1 - cut, x1, x1, x1 - cut, x0 + cut, x0, x0, x0 + cut],
                [y0, y0, y0 + cut, y1 - cut, y1, y1, y1 - cut, y0 + cut, y0])
    for k in range(1, int(legs_x) - 1):  # ganchos paralelos al eje x
        y = tc + core_d * k / (int(legs_x) - 1)
        stirrup([tc, b - tc], [y, y])
    for k in range(1, int(legs_y) - 1):  # ganchos paralelos al eje y
        x = tc + core_w * k / (int(legs_y) - 1)
        stirrup([x, x], [tc, h - tc])

    # Barras longitudinales (a escala) y su información al pasar el cursor
    hover_x, hover_y, hover_text = [], [], []
    for number, layer in enumerate(section.layers, start=1):
        y_bar = h - layer.depth_mm
        for x_rel in section.layer_x_positions(layer):
            x_bar, r = b / 2.0 + x_rel, layer.diameter_mm / 2.0
            fig.add_shape(type="circle", x0=x_bar - r, y0=y_bar - r, x1=x_bar + r, y1=y_bar + r,
                          fillcolor=BAR_COLOR, line=dict(color="#4a0d0d", width=1))
            hover_x.append(x_bar)
            hover_y.append(y_bar)
            hover_text.append(f"Capa {number}: {layer.count} Ø{layer.diameter_mm:g} mm<br>"
                              f"a {layer.depth_mm:g} mm de la cara comprimida")
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=10, opacity=0), hovertext=hover_text,
                             hoverinfo="text", showlegend=False))

    # Cotas generales: ancho y peralte, y dimensiones del núcleo (eje del estribo)
    _dimension(fig, (0, -unit), (b, -unit), f"b = {b:g} mm", text_shift=(0, -0.55 * unit))
    _dimension(fig, (tc, -2.1 * unit), (b - tc, -2.1 * unit), f"b″ = {core_w:g} mm (núcleo, al eje del estribo)",
               text_shift=(0, -0.55 * unit))
    right = b + 3.4 * unit
    _dimension(fig, (right, 0), (right, h), f"h = {h:g} mm", text_shift=(0.5 * unit, 0), text_angle=-90)
    _dimension(fig, (right + 1.7 * unit, tc), (right + 1.7 * unit, h - tc), f"d″ = {core_d:g} mm",
               text_shift=(0.5 * unit, 0), text_angle=-90)

    # Recubrimiento libre (hasta el estribo), arriba a la izquierda
    cover = section.clear_cover_mm
    if cover > 0:
        fig.add_annotation(x=cover / 2.0, y=h, ax=0.0, ay=h + 1.2 * unit, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor=DIMENSION_COLOR,
                           text=f"recubrimiento libre r = {cover:g} mm", xanchor="left", yshift=6,
                           font=dict(size=12, color=DIMENSION_COLOR))

    # Capas: profundidad desde la cara comprimida (izquierda) y número y diámetro (derecha)
    for layer in section.layers:
        y_bar = h - layer.depth_mm
        x_edge = b / 2.0 + section.layer_x_positions(layer)[0] - layer.diameter_mm / 2.0
        fig.add_shape(type="line", x0=-0.15 * unit, x1=x_edge, y0=y_bar, y1=y_bar, line=dict(color="#888888", width=1, dash="dot"))
        fig.add_annotation(x=-0.25 * unit, y=y_bar, text=f"{layer.depth_mm:.1f}", showarrow=False, xanchor="right",
                           font=dict(size=11, color=DIMENSION_COLOR))
        fig.add_annotation(x=b + 0.3 * unit, y=y_bar, text=f"<b>{layer.count} Ø{layer.diameter_mm:g}</b>", showarrow=False,
                           xanchor="left", font=dict(size=12, color=BAR_COLOR))
    fig.add_annotation(x=-0.25 * unit, y=h + 0.2 * unit, text="profundidad (mm)", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=11, color=DIMENSION_COLOR))
    fig.add_annotation(x=b, y=h + 1.2 * unit, text="▼ cara comprimida", showarrow=False, xanchor="right",
                       font=dict(size=12, color=DIMENSION_COLOR))

    # Leyenda con muestras de cada elemento
    for name, color, symbol in (("Concreto no confinado (recubrimiento)", CONCRETE_COLOR, "square"),
                                ("Núcleo confinado", CORE_COLOR, "square"),
                                (f"Estribos Ø{ds:g} @ {section.transverse.spacing_mm:g} mm", STIRRUP_COLOR, "line-ew"),
                                ("Barras longitudinales", BAR_COLOR, "circle")):
        marker = dict(symbol=symbol, size=14, color=color, line=dict(color=color, width=3 if symbol == "line-ew" else 1))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=marker, name=name))

    steel_ratio = section.steel_area_mm2 / section.gross_area_mm2
    fig.update_xaxes(visible=False, range=[-3.6 * unit, right + 3.2 * unit], constrain="domain")
    fig.update_yaxes(visible=False, range=[-3.4 * unit, h + 2.4 * unit], scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", height=620, hovermode="closest",
                      title=f"Corte de la sección: {section.number_of_bars} barras, "
                            f"As = {section.steel_area_mm2:,.0f} mm² (ρ = {steel_ratio:.2%})",
                      legend=dict(orientation="h", y=-0.02, yanchor="top"), margin=dict(l=10, r=10, t=60, b=10))
    return fig
