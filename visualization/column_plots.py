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
