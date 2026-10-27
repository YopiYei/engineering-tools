import numpy as np
import plotly.graph_objects as go
import streamlit as st

from calculations.structural.column_section import RectangularSection, TransverseReinforcement
from calculations.structural.mander_stress_strain import (
    ELASTIC_MODULUS_MODELS,
    SPALLING_STRAIN,
    UNCONFINED_PEAK_STRAIN,
    build_curve,
    confined_strength_from_lateral_stresses,
    confined_strength_from_section,
    cover_stress,
    with_ultimate_strain,
)
from styles.main import apply_global_styles, panel_intro
from visualization.mander_plots import confinement_section_figure


apply_global_styles()

CONFINEMENT_MODES = {
    "direct": "f'cc conocido",
    "lateral": "Esfuerzos laterales f'lx, f'ly",
    "section": "Sección rectangular con estribos",
}

st.title("Mander Stress-Strain Curve")

panel_intro(
    what="La curva esfuerzo–deformación de Mander, Priestley y Park (1988) para concreto no confinado y confinado, con una sola ecuación (Popovics).",
    purpose="Permite ver cuánto aumentan la resistencia y la ductilidad del concreto por el confinamiento transversal, y obtener la curva para el análisis de secciones.",
    results="Curvas no confinada, confinada y del recubrimiento; f'cc, εcc, Ec, Esec, r y, si se pide, la deformación última εcu (primer estribo roto).",
)

with st.expander("Referencia y ecuaciones", expanded=False):
    st.markdown(
        "Mander, J. B., Priestley, M. J. N., & Park, R. (1988). *Theoretical stress-strain model "
        "for confined concrete*. J. Struct. Eng., ASCE, 114(8), 1804-1826."
    )
    st.latex(r"f_c = \frac{f'_{cc}\,x\,r}{r - 1 + x^{r}} \qquad x = \frac{\varepsilon_c}{\varepsilon_{cc}}")
    st.latex(r"\varepsilon_{cc} = \varepsilon_{co}\left[1 + 5\left(\frac{f'_{cc}}{f'_{co}} - 1\right)\right]")
    st.latex(r"r = \frac{E_c}{E_c - E_{sec}} \qquad E_{sec} = \frac{f'_{cc}}{\varepsilon_{cc}}")
    st.latex(r"E_c = 5000\sqrt{f'_{co}}\ \text{[MPa]} \qquad \varepsilon_{co} = 0.002")
    st.markdown(
        r"""
        - **No confinado:** $f'_{cc}=f'_{co}$ y $\varepsilon_{cc}=\varepsilon_{co}$.
        - **Recubrimiento:** misma curva hasta $2\varepsilon_{co}$; luego recta hasta cero en $\varepsilon_{sp}$.
        - **$f'_{cc}$:** se obtiene de los esfuerzos laterales efectivos $f'_{lx}$, $f'_{ly}$ con la
          superficie de falla de cinco parámetros (Fig. 4 del artículo, aquí resuelta numéricamente),
          donde $f'_l = k_e\,\rho\,f_{yh}$.
        - **$\varepsilon_{cu}$ (primer estribo roto):** balance de energía, Ec. 64:
          $110\,\rho_{st} = U_{cc} + U_{sc} - 0.017\sqrt{f'_{co}}$ (MJ/m³).
        """
    )

with st.expander("Ejemplo de uso: qué datos necesito y cómo usar el panel", expanded=False):
    st.markdown(
        r"""
        ### Datos que necesita

        | Dato | Símbolo | Dónde sale | Valor típico |
        |---|---|---|---|
        | Resistencia del concreto sin confinar | $f'_{co}$ | Planos o especificaciones | 21–45 MPa |
        | Deformación en $f'_{co}$ | $\varepsilon_{co}$ | Artículo | 0.002 |
        | Deformación de descascaramiento | $\varepsilon_{sp}$ | Artículo | 0.006 |
        | Estribos: diámetro, espaciamiento, $f_{yh}$, ramas en x e y | $d_h$, $s$, $f_{yh}$ | Planos | Ø10 @ 100 mm, 420 MPa |
        | Sección: $b$, $h$, recubrimiento libre, barras longitudinales | | Planos | 400 × 600, r = 40 mm, 4Ø25 por cara |

        ### Pasos

        1. **Unconfined concrete:** ingrese $f'_{co}$ y deje $\varepsilon_{co}=0.002$ y $\varepsilon_{sp}=0.006$ si no tiene otros datos.
        2. **Confinement:** elija cómo conoce el confinamiento.
           - *f'cc conocido*: ya lo calculó o lo midió en ensayos; se grafica la curva directamente.
           - *Esfuerzos laterales*: tiene $f'_{lx}$ y $f'_{ly}$ (en MPa) y el programa obtiene $f'_{cc}$ de la
             superficie de falla de Mander (lo que el artículo resuelve en la Fig. 4).
           - *Sección rectangular con estribos*: ingrese la geometría y el armado; el programa calcula
             $k_e$, $f'_{lx}$, $f'_{ly}$ y $f'_{cc}$, y dibuja el corte con las zonas confinadas y no confinadas.
        3. Marque *Calcular deformación última* si quiere $\varepsilon_{cu}$ (primer estribo roto).
        4. Lea las métricas y la curva: la confinada (azul) sube y se extiende más que la no confinada (roja).

        ### Ejemplos con resultados de referencia

        **A. Ejemplo del artículo (modo *Esfuerzos laterales*):** $f'_{co}=30$ MPa, $f'_{lx}=2.7$, $f'_{ly}=5.1$ MPa
        → $f'_{cc}\approx 49.5$ MPa ($1.65\,f'_{co}$), $\varepsilon_{cc}\approx 0.0085$.

        **B. Columna 400 × 600 mm (modo *Sección rectangular*, valores por defecto):** $f'_{co}=30$ MPa,
        r = 40 mm, 4Ø25 por cara superior/inferior y 4 por cara lateral, estribo Ø10 @ 100 mm con 2 ramas en x y 3 en y, $f_{yh}=420$ MPa
        → $k_e\approx 0.70$, $f'_{cc}\approx 38.7$ MPa ($1.29\,f'_{co}$), $\varepsilon_{cc}\approx 0.0049$,
        y $\varepsilon_{cu}\approx 0.027$ si marca el cálculo de la deformación última.
        Pruebe reducir el espaciamiento a 75 mm y observe cómo sube $f'_{cc}$ y $\varepsilon_{cu}$.

        > Las zonas naranjas del corte son las parábolas de concreto no confinado entre barras
        > (arqueo, Fig. 3 del artículo); mientras más cerca estén las barras apoyadas en estribos o ganchos,
        > menor es esa zona y mayor es $k_e$.
        """
    )

# ---------------------------------------------------------
# 1. Unconfined concrete
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("1. Unconfined concrete")
    col_fco, col_eco, col_ec, col_esp = st.columns(4)
    with col_fco:
        unconfined_strength = st.number_input("f'co (MPa)", min_value=10.0, max_value=90.0, value=30.0, step=1.0)
    with col_eco:
        unconfined_peak_strain = st.number_input("εco", min_value=0.001, max_value=0.004, value=UNCONFINED_PEAK_STRAIN, step=0.0005, format="%.4f")
    with col_ec:
        elastic_modulus_model = st.selectbox(
            "Ec",
            options=list(ELASTIC_MODULUS_MODELS),
            format_func=lambda key: ELASTIC_MODULUS_MODELS[key][0],
        )
    with col_esp:
        spalling_strain = st.number_input("εsp (recubrimiento)", min_value=0.004, max_value=0.012, value=SPALLING_STRAIN, step=0.001, format="%.4f")

# ---------------------------------------------------------
# 2. Confinement
# ---------------------------------------------------------

confined_strength = None
transverse_volumetric_ratio = None
longitudinal_volumetric_ratio = 0.0
confinement_summary = {}

with st.container(border=True):
    st.subheader("2. Confinement")
    confinement_mode = st.radio("Dato de confinamiento", list(CONFINEMENT_MODES), format_func=CONFINEMENT_MODES.get, horizontal=True)

    try:
        if confinement_mode == "direct":
            confined_strength = st.number_input(
                "f'cc (MPa)", min_value=unconfined_strength, max_value=5.0 * unconfined_strength,
                value=min(1.5 * unconfined_strength, 5.0 * unconfined_strength), step=1.0,
            )

        elif confinement_mode == "lateral":
            col_lx, col_ly = st.columns(2)
            with col_lx:
                lateral_stress_x = st.number_input("f'lx (MPa)", min_value=0.0, value=2.7, step=0.1)
            with col_ly:
                lateral_stress_y = st.number_input("f'ly (MPa)", min_value=0.0, value=5.1, step=0.1)
            confined_strength = confined_strength_from_lateral_stresses(unconfined_strength, lateral_stress_x, lateral_stress_y)
            confinement_summary = {"f'lx (MPa)": lateral_stress_x, "f'ly (MPa)": lateral_stress_y}

        else:
            col_b, col_h, col_cover = st.columns(3)
            with col_b:
                section_width = st.number_input("Ancho b (mm)", min_value=100.0, value=400.0, step=10.0)
            with col_h:
                section_depth = st.number_input("Peralte h (mm)", min_value=100.0, value=600.0, step=10.0)
            with col_cover:
                clear_cover = st.number_input("Recubrimiento libre (mm)", min_value=10.0, value=40.0, step=5.0)

            col_nb, col_ns, col_db = st.columns(3)
            with col_nb:
                bars_top_bottom = st.number_input("Barras en cara sup. e inf.", min_value=2, value=4, step=1)
            with col_ns:
                bars_sides = st.number_input("Barras en cada cara lateral (con esquinas)", min_value=2, value=4, step=1)
            with col_db:
                longitudinal_diameter = st.number_input("Ø longitudinal (mm)", min_value=6.0, value=25.0, step=1.0)

            col_dh, col_s, col_fyh = st.columns(3)
            with col_dh:
                hoop_diameter = st.number_input("Ø estribo (mm)", min_value=6.0, value=10.0, step=1.0)
            with col_s:
                hoop_spacing = st.number_input("Espaciamiento s (mm)", min_value=20.0, value=100.0, step=5.0)
            with col_fyh:
                hoop_yield_stress = st.number_input("fyh (MPa)", min_value=100.0, value=420.0, step=10.0)

            col_lx, col_ly = st.columns(2)
            with col_lx:
                legs_x = st.number_input("Ramas paralelas a x", min_value=2, value=2, step=1)
            with col_ly:
                legs_y = st.number_input("Ramas paralelas a y", min_value=2, value=3, step=1)

            transverse = TransverseReinforcement.from_legs(hoop_diameter, hoop_spacing, hoop_yield_stress, legs_x, legs_y)
            section = RectangularSection.with_perimeter_bars(
                section_width, section_depth, clear_cover, int(bars_top_bottom), int(bars_sides), longitudinal_diameter, transverse
            )
            confinement = confined_strength_from_section(section, unconfined_strength, unconfined_peak_strain)
            confined_strength = confinement.confined_strength
            longitudinal_volumetric_ratio = section.longitudinal_ratio_core
            # rho_st: volumen de estribos / volumen del núcleo
            transverse_volumetric_ratio = confinement.rho_x + confinement.rho_y
            st.plotly_chart(confinement_section_figure(section, legs_x=int(legs_x), legs_y=int(legs_y)), width="stretch")
            confinement_summary = {
                "ke": confinement.effectiveness,
                "ρx": confinement.rho_x,
                "ρy": confinement.rho_y,
                "ρst": transverse_volumetric_ratio,
                "ρcc": longitudinal_volumetric_ratio,
                "f'lx (MPa)": confinement.lateral_stress_x,
                "f'ly (MPa)": confinement.lateral_stress_y,
            }

        compute_ultimate = st.checkbox("Calcular deformación última εcu (primer estribo roto, Ec. 64)", value=False)
        if compute_ultimate:
            if transverse_volumetric_ratio is None:
                transverse_volumetric_ratio = st.number_input("ρst (volumen de estribos / volumen del núcleo)", min_value=0.0005, value=0.015, step=0.001, format="%.4f")
            if confinement_mode != "section":
                longitudinal_volumetric_ratio = st.number_input("ρcc (acero longitudinal / área del núcleo, 0 = se desprecia)", min_value=0.0, value=0.0, step=0.005, format="%.4f")
            longitudinal_yield_stress = st.number_input("fy longitudinal (MPa)", min_value=0.0, value=420.0, step=10.0) if longitudinal_volumetric_ratio > 0 else 0.0
    except ValueError as error:
        st.error(str(error))
        st.stop()

# ---------------------------------------------------------
# 3. Curves
# ---------------------------------------------------------

try:
    unconfined_curve = build_curve(
        unconfined_strength,
        unconfined_peak_strain=unconfined_peak_strain,
        elastic_modulus_model=elastic_modulus_model,
    )
    confined_curve = build_curve(
        unconfined_strength,
        confined_strength=confined_strength,
        unconfined_peak_strain=unconfined_peak_strain,
        elastic_modulus_model=elastic_modulus_model,
    )
    if compute_ultimate:
        confined_curve = with_ultimate_strain(
            confined_curve,
            transverse_volumetric_ratio=transverse_volumetric_ratio,
            longitudinal_volumetric_ratio=longitudinal_volumetric_ratio,
            longitudinal_yield_stress=longitudinal_yield_stress,
        )
except ValueError as error:
    st.error(str(error))
    st.stop()

strain_axis = np.linspace(0.0, max(unconfined_curve.strain[-1], confined_curve.strain[-1]), 500)
cover_curve_stress = cover_stress(
    strain_axis, unconfined_strength, unconfined_peak_strain, unconfined_curve.power_r, spalling_strain
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=confined_curve.strain * 1000, y=confined_curve.stress, mode="lines", name="Confinado",
    line=dict(color="rgb(31, 119, 180)", width=3), fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.08)",
))
fig.add_trace(go.Scatter(
    x=strain_axis * 1000, y=cover_curve_stress, mode="lines", name="No confinado / recubrimiento",
    line=dict(color="rgb(214, 39, 40)", width=3),
))
fig.add_trace(go.Scatter(
    x=[unconfined_curve.unconfined_peak_strain * 1000, confined_curve.confined_peak_strain * 1000],
    y=[unconfined_strength, confined_curve.confined_strength],
    mode="markers+text", text=["f'co", "f'cc"], textposition="top center",
    marker=dict(size=9, color="black"), showlegend=False,
))
if confined_curve.ultimate_strain is not None:
    ultimate_stress = float(np.interp(confined_curve.ultimate_strain, confined_curve.strain, confined_curve.stress))
    fig.add_trace(go.Scatter(
        x=[confined_curve.ultimate_strain * 1000], y=[ultimate_stress],
        mode="markers+text", text=["εcu"], textposition="top right",
        marker=dict(size=10, symbol="x", color="rgb(44, 160, 44)"), name="Primer estribo roto",
    ))
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig.update_layout(
    title=f"Mander (1988): f'co = {unconfined_strength:g} MPa, f'cc = {confined_curve.confined_strength:.1f} MPa",
    xaxis_title="Strain (‰ = mm/m)",
    yaxis_title="Stress (MPa)",
    hovermode="x unified",
    template="plotly_white",
    height=520,
    legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------
# 4. Results
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("3. Results")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("f'cc (MPa)", f"{confined_curve.confined_strength:.2f}")
    col_b.metric("f'cc / f'co", f"{confined_curve.confined_strength / unconfined_strength:.3f}")
    col_c.metric("εcc", f"{confined_curve.confined_peak_strain:.5f}")
    col_d.metric("εcu", f"{confined_curve.ultimate_strain:.4f}" if confined_curve.ultimate_strain is not None else "—")

    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Ec (MPa)", f"{confined_curve.elastic_modulus:,.0f}")
    col_f.metric("Esec (MPa)", f"{confined_curve.secant_modulus:,.0f}")
    col_g.metric("r", f"{confined_curve.power_r:.3f}")

    if confinement_summary:
        st.caption("Confinamiento: " + " · ".join(f"{name} = {value:.4g}" for name, value in confinement_summary.items()))
