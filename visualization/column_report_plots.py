"""Figuras estáticas (PNG) para el informe del análisis COLUMN.

Son equivalentes a los gráficos interactivos de `column_plots.py`, pero con matplotlib para poder
incrustarlas en Word y PDF sin depender de un navegador. Se usa `Figure` directamente (sin pyplot)
para no abrir ventanas ni compartir estado global.
"""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from calculations.structural.column_analysis import ColumnModel, MomentCurvatureResult

PHI_UNIT = 1e6  # 1/mm -> µrad/mm

MARKERS = {
    "cover_peak": ("#8c564b", "o"), "first_yield": ("#d62728", "D"), "code_strain": ("#7f7f7f", "s"),
    "max_moment": ("#2ca02c", "*"), "core_peak": ("#9467bd", "o"), "steel_hardening": ("#ff7f0e", "^"),
    "cover_spalling": ("#bcbd22", "v"), "core_ultimate": ("black", "X"), "hoop_fracture": ("black", "X"),
    "steel_ultimate": ("black", "X"), "ductility_limit": ("black", "X"), "moment_drop": ("black", "X"),
}
TERMINATION_COLORS = {"hoop_fracture": "#d62728", "steel_ultimate": "#1f77b4", "core_ultimate": "#2ca02c",
                      "ductility_limit": "#9467bd", "moment_drop": "#ff7f0e", "no_convergence": "#7f7f7f"}


@dataclass(frozen=True)
class PngImage:
    data: bytes
    width_px: int
    height_px: int

    @property
    def aspect(self) -> float:
        """Alto / ancho."""
        return self.height_px / self.width_px


def _to_png(fig: Figure, dpi: int = 170) -> PngImage:
    FigureCanvasAgg(fig)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    from PIL import Image  # pillow ya es dependencia de matplotlib

    buffer.seek(0)
    with Image.open(buffer) as image:
        size = image.size
    return PngImage(buffer.getvalue(), size[0], size[1])


def _dimension(ax, start, end, text, offset=(0.0, 0.0), rotation=0):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.9))
    ax.text((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1], text, ha="center", va="center",
            fontsize=8, color="#333333", rotation=rotation)


def section_png(section, octagonal=False, legs_x=2, legs_y=2) -> PngImage:
    """Corte acotado de la sección (concreto, núcleo, estribos y barras)."""
    b, h = section.width_mm, section.depth_mm
    ds = section.transverse.bar_diameter_mm
    tc = section.hoop_centreline_cover_mm
    unit = max(b, h) * 0.075
    fig = Figure(figsize=(6.2, 6.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    x_min, x_max = -3.4 * unit, b + 8.4 * unit
    y_min, y_max = -3.4 * unit, h + 2.4 * unit
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), b, h, facecolor="#d9d9d9", edgecolor="#555555", lw=1.5))
    ax.add_patch(Rectangle((tc, tc), b - 2 * tc, h - 2 * tc, facecolor="#b9d3ee", edgecolor="none"))
    # Ancho de línea del estribo en puntos, a escala con el dibujo (72 puntos por pulgada)
    points_per_mm = fig.get_figheight() * 0.96 * 72 / (y_max - y_min)
    lw = max(1.2, ds * points_per_mm)

    def stirrup(xs, ys):
        ax.plot(xs, ys, color="#1f3a93", lw=lw, solid_capstyle="butt", solid_joinstyle="miter")

    core_w, core_d = b - 2 * tc, h - 2 * tc
    stirrup([tc, b - tc, b - tc, tc, tc], [tc, tc, h - tc, h - tc, tc])
    if octagonal:
        cut = min(core_w, core_d) / (2.0 + 2.0**0.5)
        x0, x1, y0, y1 = tc, b - tc, tc, h - tc
        stirrup([x0 + cut, x1 - cut, x1, x1, x1 - cut, x0 + cut, x0, x0, x0 + cut],
                [y0, y0, y0 + cut, y1 - cut, y1, y1, y1 - cut, y0 + cut, y0])
    for k in range(1, int(legs_x) - 1):
        y = tc + core_d * k / (int(legs_x) - 1)
        stirrup([tc, b - tc], [y, y])
    for k in range(1, int(legs_y) - 1):
        x = tc + core_w * k / (int(legs_y) - 1)
        stirrup([x, x], [tc, h - tc])

    for layer in section.layers:
        y_bar = h - layer.depth_mm
        for x_rel in section.layer_x_positions(layer):
            ax.add_patch(Circle((b / 2 + x_rel, y_bar), layer.diameter_mm / 2, facecolor="#a01c1c", edgecolor="#4a0d0d", lw=0.8))
        ax.plot([-0.15 * unit, b / 2 + section.layer_x_positions(layer)[0] - layer.diameter_mm / 2], [y_bar, y_bar],
                color="#888888", lw=0.7, ls=":")
        ax.text(-0.25 * unit, y_bar, f"{layer.depth_mm:.1f}", ha="right", va="center", fontsize=7.5, color="#333333")
        ax.text(b + 0.3 * unit, y_bar, f"{layer.count} Ø{layer.diameter_mm:g}", ha="left", va="center", fontsize=8.5,
                color="#a01c1c", fontweight="bold")
    ax.text(-0.25 * unit, h + 0.35 * unit, "profundidad (mm)", ha="right", va="bottom", fontsize=7.5, color="#333333")
    ax.text(b, h + 1.2 * unit, "▼ cara comprimida", ha="right", va="center", fontsize=8, color="#333333")
    if section.clear_cover_mm > 0:
        ax.annotate(f"recubrimiento libre r = {section.clear_cover_mm:g} mm", xy=(section.clear_cover_mm / 2, h),
                    xytext=(0, h + 1.2 * unit), fontsize=8, color="#333333", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))

    _dimension(ax, (0, -unit), (b, -unit), f"b = {b:g} mm", (0, -0.6 * unit))
    _dimension(ax, (tc, -2.1 * unit), (b - tc, -2.1 * unit), f"b″ = {core_w:g} mm (núcleo, al eje del estribo)", (0, -0.6 * unit))
    right = b + 4.4 * unit
    _dimension(ax, (right, 0), (right, h), f"h = {h:g} mm", (0.7 * unit, 0), 90)
    _dimension(ax, (right + 1.9 * unit, tc), (right + 1.9 * unit, h - tc), f"d″ = {core_d:g} mm", (0.7 * unit, 0), 90)

    handles = [Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor="#555555"),
               Rectangle((0, 0), 1, 1, facecolor="#b9d3ee"),
               Rectangle((0, 0), 1, 1, facecolor="#1f3a93"),
               Circle((0, 0), 1, facecolor="#a01c1c")]
    ax.legend(handles, ["Concreto no confinado (recubrimiento)", "Núcleo confinado",
                        f"Estribos Ø{ds:g} @ {section.transverse.spacing_mm:g} mm", "Barras longitudinales"],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"Corte de la sección: {section.number_of_bars} barras, As = {section.steel_area_mm2:,.0f} mm² "
                 f"(ρ = {section.steel_area_mm2 / section.gross_area_mm2:.2%})", fontsize=10)
    return _to_png(fig)


def moment_curvature_png(result: MomentCurvatureResult) -> PngImage:
    """Curva M–φ con los estados límite alcanzados."""
    fig = Figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(result.curvature * PHI_UNIT, result.moment / 1e6, color="#1f77b4", lw=2.2, label="M-φ monotónica")
    for record in result.limit_states:
        if not record.reached or record.curvature is None:
            continue
        color, marker = MARKERS.get(record.key, ("#333333", "o"))
        ax.plot(record.curvature * PHI_UNIT, record.moment / 1e6, marker=marker, color=color, ms=9 if marker != "*" else 13,
                ls="none", markeredgecolor="white", label=record.label)
    ax.axhline(result.ideal_moment / 1e6, color="gray", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], result.ideal_moment / 1e6, " Mi", va="bottom", ha="right", fontsize=8, color="gray")
    if result.yield_curvature:
        ax.axvline(result.yield_curvature * PHI_UNIT, color="gray", ls=":", lw=1)
        ax.text(result.yield_curvature * PHI_UNIT, ax.get_ylim()[1], " φy", va="top", fontsize=8, color="gray")
    ax.set_xlabel("Curvatura, φ (µrad/mm = 10⁻³ 1/m)")
    ax.set_ylabel("Momento, M (kN·m)")
    ax.set_title("Diagrama momento–curvatura")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.5, frameon=False)
    return _to_png(fig)


def materials_png(model: ColumnModel) -> PngImage:
    """Curvas esfuerzo–deformación del concreto (recubrimiento y núcleo) y del acero."""
    fig = Figure(figsize=(7.4, 3.6))
    ax_c, ax_s = fig.add_subplot(121), fig.add_subplot(122)
    e = np.linspace(0, max(4 * model.core.peak_strain, 1.2 * model.cover.spall_strain, 0.015), 400)
    ax_c.plot(e, model.cover.stress_and_tangent(e)[0], color="#8c564b", lw=2, label="Recubrimiento")
    ax_c.plot(e, model.core.stress_and_tangent(e)[0], color="#1f77b4", lw=2, label="Núcleo confinado")
    ax_c.set(xlabel="Deformación, ε", ylabel="Esfuerzo (MPa)", title="Concreto (compresión positiva)")
    ax_c.legend(fontsize=8)
    steel = model.steel
    s = np.linspace(0, steel.ultimate_strain, 500)
    ax_s.plot(s, steel.stress_and_tangent(s)[0], color="#d62728", lw=2)
    ax_s.set(xlabel="Deformación, ε", ylabel="Esfuerzo (MPa)", title="Acero longitudinal")
    for ax in (ax_c, ax_s):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _to_png(fig)


def interaction_png(model: ColumnModel, curve, points, current_axial_kn=None) -> PngImage:
    """Izquierda: diagrama N–M (ideal y Mmax del análisis). Derecha: ductilidad vs carga axial."""
    fig = Figure(figsize=(8.2, 4.4))
    ax_m, ax_d = fig.add_subplot(121), fig.add_subplot(122)
    ax_m.plot(curve.moment_knm, curve.axial_kn, color="#1f77b4", lw=2, label="Ideal (método de la norma)")
    for name, point, marker in (("Tracción pura", curve.tension_point, "v"), ("Punto balanceado", curve.balanced_point, "D"),
                                ("Compresión pura", curve.compression_point, "^")):
        ax_m.plot(point[1], point[0], marker, color="#d62728", ms=7)
        ax_m.annotate(name, (point[1], point[0]), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ok = [p for p in points if p.max_moment is not None]
    if ok:
        ax_m.plot([p.max_moment / 1e6 for p in ok], [p.axial_load_n / 1e3 for p in ok], "o-", color="#2ca02c", ms=4, lw=1.2,
                  label="Mmax del análisis")
    if current_axial_kn is not None:
        ax_m.axhline(current_axial_kn, color="gray", ls=":", lw=1)
    ax_m.set(xlabel="Momento, M (kN·m)", ylabel="Carga axial, P (kN)", title="Diagrama de interacción P–M")
    ax_m.legend(fontsize=7.5)
    gross = model.concrete.unconfined_strength * model.section.gross_area_mm2
    shown = set()
    for p in points:
        if p.ductility is None:
            continue
        color = TERMINATION_COLORS.get(p.termination_key, "#333333")
        ax_d.plot(p.axial_load_n / gross, p.ductility, "o", color=color, ms=6,
                  label=None if p.termination_label in shown else p.termination_label)
        shown.add(p.termination_label)
    ductile = [p for p in points if p.ductility is not None]
    if ductile:
        ax_d.plot([p.axial_load_n / gross for p in ductile], [p.ductility for p in ductile], "-", color="#bbbbbb", lw=1, zorder=0)
    ax_d.set(xlabel="P / (f'co·Ag)", ylabel="Ductilidad de curvatura, φu/φy", title="Ductilidad vs. carga axial")
    if shown:
        ax_d.legend(fontsize=7.5, title="Termina por", title_fontsize=7.5)
    for ax in (ax_m, ax_d):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _to_png(fig)


def variants_png(outcomes) -> PngImage:
    """Curvas M–φ de las variantes: absolutas (izquierda) y normalizadas φ/φy, M/Mi (derecha)."""
    fig = Figure(figsize=(8.2, 4.2))
    ax_a, ax_n = fig.add_subplot(121), fig.add_subplot(122)
    for outcome in outcomes:
        if outcome.result is None:
            continue
        r = outcome.result
        ax_a.plot(r.curvature * PHI_UNIT, r.moment / 1e6, lw=1.8, label=outcome.variant.name)
        if r.yield_curvature:
            ax_n.plot(r.curvature / r.yield_curvature, r.moment / r.ideal_moment, lw=1.8, label=outcome.variant.name)
    ax_a.set(xlabel="Curvatura, φ (µrad/mm)", ylabel="Momento, M (kN·m)", title="Curvas M–φ")
    ax_n.set(xlabel="φ/φy", ylabel="M / Mi", title="Normalizadas")
    for ax in (ax_a, ax_n):
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax_a.legend(fontsize=8)
    fig.tight_layout()
    return _to_png(fig)
