import numpy as np
import plotly.graph_objects as go
import streamlit as st

from calculations.structural.column_axial_response import (
    generate_column_axial_response,
    rectangular_column_area,
)

from styles.main import apply_global_styles, panel_intro


apply_global_styles()


# ---------------------------------------------------------
# Model labels (presentation layer only)
# ---------------------------------------------------------

COMPRESSION_MODEL_LABELS = {
    "collins_mitchell": "Collins & Mitchell (1991)",
    "de_nicolo": "De Nicolo, Pani y Pozzo (1994)",
    "kent_park": "Kent & Park (1971) — no confinado",
    "desayi_krishnan": "Desayi & Krishnan (1964)",
}

TENSION_MODEL_LABELS = {
    "collins_mitchell": "Collins & Mitchell (lineal)",
    "restrepo_rodriguez": "Restrepo-Rodríguez (2019)",
    "fields_bischoff": "Fields & Bischoff (2004)",
}

ACTION_LABELS = {
    "compression": "Compresión (acortamiento → aplastamiento)",
    "tension": "Tracción (elongación → fisuración)",
}


st.title("Column Axial Force-Displacement Response")

panel_intro(
    what="La curva fuerza–desplazamiento de una columna prismática de concreto simple bajo carga axial centrada, a partir de la curva esfuerzo–deformación del material.",
    purpose="Muestra cómo la ley constitutiva del concreto gobierna la rigidez, la resistencia y la capacidad de deformación axial de un elemento (F = σ·A, δ = ε·L).",
    results="Curva F–δ en compresión o tracción, área, fuerza pico, desplazamiento en el pico y último, y rigidez inicial.",
)

st.markdown(
    """
    Convierte la curva esfuerzo-deformación del concreto en una curva
    **fuerza-desplazamiento** para una columna prismática de concreto
    simple sometida a una carga axial centrada (compresión pura o
    tracción pura, sin flexión), asumiendo esfuerzo y deformación
    uniformes en toda la sección:
    """
)

st.latex(r"\delta = \varepsilon \cdot L \qquad\qquad F = \sigma \cdot A")

with st.expander("📚 Supuestos y justificación de los modelos", expanded=False):
    st.markdown(
        """
        - Columna de concreto simple (sin refuerzo), sección rectangular
          constante, bajo carga axial centrada. Esfuerzo y deformación se
          asumen uniformes en toda la sección y constantes a lo largo de
          la altura.
        - **Compresión (2.1 — aplastamiento):** se recomienda por defecto
          el modelo de **Kent & Park (1971) no confinado**, porque su
          rama descendente representa explícitamente la pérdida súbita de
          capacidad después del pico, propia del colapso frágil por
          aplastamiento del concreto simple sin confinamiento transversal.
        - **Tracción (2.2 — fisuración):** se recomienda el modelo de
          **Restrepo-Rodríguez (2019)**, que incorpora una rama de
          ablandamiento explícita después del agrietamiento hasta
          ε_cto = 2500 microdeformaciones.
        - Ambos selectores de modelo pueden cambiarse para comparar otras
          combinaciones.
        """
    )


# ---------------------------------------------------------
# 1. Column geometry and material
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("1. Column Geometry and Material")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fc = st.number_input(
            "f'c (MPa)",
            min_value=0.1,
            value=28.0,
            step=1.0,
        )

    with col2:
        width_cm = st.number_input(
            "Width, b (cm)",
            min_value=0.1,
            value=50.0,
            step=1.0,
        )

    with col3:
        depth_cm = st.number_input(
            "Depth, d (cm)",
            min_value=0.1,
            value=50.0,
            step=1.0,
        )

    with col4:
        height_m = st.number_input(
            "Height, L (m)",
            min_value=0.01,
            value=1.0,
            step=0.1,
        )

    width_mm = width_cm * 10.0
    depth_mm = depth_cm * 10.0
    height_mm = height_m * 1000.0

    area_mm2 = rectangular_column_area(width_mm, depth_mm)

    st.caption(
        f"Área de la sección, A = {area_mm2:,.0f} mm² "
        f"({area_mm2 / 1e6:.4f} m²)"
    )


# ---------------------------------------------------------
# 2. Loading action and model
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("2. Loading Action and Model")

    action = st.radio(
        "Select loading action",
        options=["compression", "tension"],
        format_func=lambda x: ACTION_LABELS[x],
        horizontal=True,
    )

    if action == "compression":
        model_options = list(COMPRESSION_MODEL_LABELS.keys())
        model = st.selectbox(
            "Compression model",
            options=model_options,
            index=model_options.index("kent_park"),
            format_func=lambda k: COMPRESSION_MODEL_LABELS[k],
        )
    else:
        model_options = list(TENSION_MODEL_LABELS.keys())
        model = st.selectbox(
            "Tension model",
            options=model_options,
            index=model_options.index("restrepo_rodriguez"),
            format_func=lambda k: TENSION_MODEL_LABELS[k],
        )


# ---------------------------------------------------------
# 3. Generate curve
# ---------------------------------------------------------

if st.button(
    "Generate Force-Displacement Curve",
    type="primary",
    use_container_width=True,
):
    try:
        if fc <= 0:
            raise ValueError("f'c must be greater than zero.")

        response = generate_column_axial_response(
            fc_prime=fc,
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            action=action,
            model=model,
        )

        displacement = response["displacement_mm"]
        force_kn = response["force_n"] / 1000.0

        is_compression = action == "compression"
        color = "rgb(31, 119, 180)" if is_compression else "rgb(214, 39, 40)"
        fill_color = (
            "rgba(31, 119, 180, 0.1)" if is_compression else "rgba(214, 39, 40, 0.1)"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=displacement,
                y=force_kn,
                mode="lines",
                name="F-δ",
                line=dict(color=color, width=3),
                fill="tozeroy",
                fillcolor=fill_color,
            )
        )

        peak_idx = int(np.argmax(force_kn))
        fig.add_annotation(
            x=displacement[peak_idx],
            y=force_kn[peak_idx],
            text=(
                f"F pico = {force_kn[peak_idx]:.1f} kN<br>"
                f"δ = {displacement[peak_idx]:.3f} mm"
            ),
            showarrow=True,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color,
            borderwidth=1,
            font=dict(size=10, color="#1a1a1a"),
        )

        action_label = "Compresión" if is_compression else "Tracción"
        model_label = (
            COMPRESSION_MODEL_LABELS if is_compression else TENSION_MODEL_LABELS
        )[model]

        fig.update_layout(
            title=(
                f"Curva Fuerza-Desplazamiento — {action_label} "
                f"(f'c = {fc:.0f} MPa, b×d = {width_cm:.0f}×{depth_cm:.0f} cm, "
                f"L = {height_m:.2f} m)<br><sub>Modelo: {model_label}</sub>"
            ),
            xaxis_title="Desplazamiento, δ (mm)",
            yaxis_title="Fuerza axial, F (kN)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resultados")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Área, A", f"{response['area_mm2']:,.0f} mm²")
        col2.metric(
            "F pico" if is_compression else "F agrietamiento",
            f"{response['force_peak_n'] / 1000:,.1f} kN",
        )
        col3.metric("δ en pico", f"{response['displacement_at_peak_mm']:.3f} mm")
        col4.metric(
            "δ última (dominio calculado)",
            f"{response['displacement_ultimate_mm']:.3f} mm",
        )
        col5.metric(
            "Rigidez inicial, K₀",
            f"{response['initial_stiffness_n_per_mm'] / 1000:,.1f} kN/mm",
        )

    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")
