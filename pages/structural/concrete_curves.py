import streamlit as st
import plotly.graph_objects as go
import numpy as np

from calculations.structural.concrete_properties import (
    generate_curves,
    concrete_collins_mitchell,
    concrete_de_nicolo_pani_pozzo,
    concrete_kent_park,
    concrete_desayi_krishnan,
)

from styles.main import apply_global_styles, panel_intro
from utils.shared_state import publish_concrete


apply_global_styles()


st.title("Concrete Stress-Strain Curves")

panel_intro(
    what="Curvas esfuerzo–deformación del concreto en compresión (Collins y Mitchell, De Nicolo et al., Kent y Park, Desayi y Krishnan) y en tracción (Collins y Mitchell, Restrepo y Rodríguez, Fields y Bischoff).",
    purpose="Permite comparar leyes constitutivas y elegir la adecuada para el análisis no lineal de elementos y secciones de concreto.",
    results="Curvas individuales, comparaciones y combinaciones de modelos para distintos valores de f'c, con sus parámetros característicos.",
)


# ---------------------------------------------------------
# Theory Section (Expandable)
# ---------------------------------------------------------

with st.expander("📚 Theory and Models", expanded=False):

    st.markdown("## Modelos de Compresión")

    with st.container(border=True):
        st.markdown("### 1. Collins & Mitchell (1991)")
        st.latex(r"\varepsilon'_c = \frac{f'_c}{E_c} \frac{n}{n-1}")
        st.latex(r"n = 0.8 + \frac{f'_c}{17}")
        st.latex(r"E_c = 4700\sqrt{f'_c}")
        st.write("Modelo recomendado para concreto de peso normal.")

    with st.container(border=True):
        st.markdown("### 2. De Nicolo, Pani y Pozzo (1994)")
        st.latex(r"\varepsilon'_c = 0.00076 + \sqrt{(0.626f'_c - 4.33) \times 10^{-7}}")
        st.write("Válido para 10 MPa ≤ fc ≤ 100 MPa.")

    with st.container(border=True):
        st.markdown("### 3. Kent & Park (1971)")
        st.latex(r"f_c = f'_c \left[2\frac{\varepsilon}{\varepsilon_0} - \left(\frac{\varepsilon}{\varepsilon_0}\right)^2\right]")
        st.write("Rama descendente con parámetro Z.")

    with st.container(border=True):
        st.markdown("### 4. Desayi & Krishnan (1964)")
        st.latex(r"f_c = E_c \frac{\varepsilon}{1 + \left(\frac{\varepsilon}{\varepsilon_0}\right)^2}")
        st.write("Modelo parabólico para concreto no confinado.")

    st.divider()

    st.markdown("## Modelos de Tracción")

    with st.container(border=True):
        st.markdown("### 1. Collins & Mitchell (Simplificado)")
        st.latex(r"f_t = 0.1\sqrt{f'_c}")

    with st.container(border=True):
        st.markdown("### 2. Restrepo-Rodríguez (2019)")
        st.write("Rama elástica + rama de ablandamiento con β = 0.9, εcto = 2500 μ")

    with st.container(border=True):
        st.markdown("### 3. Fields & Bischoff (2004)")
        st.latex(r"\frac{f_c}{f_{cr}} = e^{-0.8(\varepsilon_c - \varepsilon_{cr}) \times 10^3}")


# ---------------------------------------------------------
# View Mode Selection
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("View Mode")
    view_mode = st.radio(
        "Select view mode",
        options=["single", "comparison", "combinations"],
        format_func=lambda x: {
            "single": "Single fc' value",
            "comparison": "Compare multiple fc' values (20, 28, 35, 40, 45, 56 MPa)",
            "combinations": "Compare model combinations",
        }[x],
        horizontal=True,
    )


# ---------------------------------------------------------
# Input Parameters
# ---------------------------------------------------------

if view_mode == "single":
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        with st.container(border=True):
            st.subheader("Compression")
            compression_model = st.selectbox(
                "Model",
                options=[
                    "collins_mitchell",
                    "de_nicolo",
                    "kent_park",
                    "desayi_krishnan",
                ],
                format_func=lambda x: {
                    "collins_mitchell": "Collins & Mitchell",
                    "de_nicolo": "De Nicolo, Pani y Pozzo",
                    "kent_park": "Kent & Park",
                    "desayi_krishnan": "Desayi & Krishnan",
                }[x],
                key="comp_model_single",
            )

    with col2:
        with st.container(border=True):
            st.subheader("Tension")
            tension_model = st.selectbox(
                "Model",
                options=[
                    "collins_mitchell",
                    "restrepo_rodriguez",
                    "fields_bischoff",
                ],
                format_func=lambda x: {
                    "collins_mitchell": "Collins & Mitchell",
                    "restrepo_rodriguez": "Restrepo-Rodríguez",
                    "fields_bischoff": "Fields & Bischoff",
                }[x],
                key="tens_model_single",
            )

    with col3:
        with st.container(border=True):
            st.subheader("Parameters")
            fc = st.number_input(
                "fc' (MPa)",
                min_value=10.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                key="fc_single",
            )

            show_formulas = st.checkbox("Show formulas", value=False, key="formulas_single")

    # El panel COLUMN toma este f'c y estos modelos como datos del concreto (solo en la vista de un f'c)
    publish_concrete(fc, compression_model, tension_model)

elif view_mode == "comparison":
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        with st.container(border=True):
            st.subheader("Compression")
            compression_model = st.selectbox(
                "Model",
                options=[
                    "collins_mitchell",
                    "de_nicolo",
                    "kent_park",
                    "desayi_krishnan",
                ],
                format_func=lambda x: {
                    "collins_mitchell": "Collins & Mitchell",
                    "de_nicolo": "De Nicolo, Pani y Pozzo",
                    "kent_park": "Kent & Park",
                    "desayi_krishnan": "Desayi & Krishnan",
                }[x],
                key="comp_model_comp",
            )

    with col2:
        with st.container(border=True):
            st.subheader("Tension")
            tension_model = st.selectbox(
                "Model",
                options=[
                    "collins_mitchell",
                    "restrepo_rodriguez",
                    "fields_bischoff",
                ],
                format_func=lambda x: {
                    "collins_mitchell": "Collins & Mitchell",
                    "restrepo_rodriguez": "Restrepo-Rodríguez",
                    "fields_bischoff": "Fields & Bischoff",
                }[x],
                key="tens_model_comp",
            )

    with col3:
        with st.container(border=True):
            st.subheader("Visualization")
            view_type = st.radio(
                "View type",
                options=["split", "overlapped"],
                format_func=lambda x: {
                    "split": "Split",
                    "overlapped": "Overlapped",
                }[x],
                key="view_type_comp",
                horizontal=False,
            )

    # FC values selection
    with st.container(border=True):
        st.subheader("Select f'c values (MPa)")
        all_fc_values = [20, 28, 35, 40, 45, 56]

        cols_fc = st.columns(6)
        selected_fc_values = []

        for i, fc_val in enumerate(all_fc_values):
            with cols_fc[i]:
                if st.checkbox(
                    f"{fc_val}",
                    value=True,
                    key=f"fc_check_{fc_val}",
                ):
                    selected_fc_values.append(fc_val)

    show_formulas = st.checkbox("Show formulas", value=False, key="formulas_comp")
    fc = None

else:  # combinations mode
    st.write("Compare different combinations of compression and tension models for fc' = 30 MPa")
    show_formulas = False
    fc = None
    compression_model = None
    tension_model = None


# ---------------------------------------------------------
# Generate Curves
# ---------------------------------------------------------

if view_mode == "single":
    generate_button = st.button(
        "Generate Curves",
        type="primary",
        use_container_width=True,
    )
else:
    generate_button = True


if generate_button:
    try:

        if view_mode == "single":
            # Single curve mode
            curves = generate_curves(
                fc_prime=fc,
                compression_model=compression_model,
                tension_model=tension_model,
            )

            strain_comp = curves["strain_comp"]
            stress_comp = curves["stress_comp"]
            strain_tens = curves["strain_tens"]
            stress_tens = curves["stress_tens"]

            # Create figure
            fig = go.Figure()

            # Compression curve
            fig.add_trace(
                go.Scatter(
                    x=strain_comp * 1000,
                    y=stress_comp,
                    mode="lines",
                    name="Compression",
                    line=dict(
                        color="rgb(31, 119, 180)",
                        width=3,
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(31, 119, 180, 0.1)",
                )
            )

            # Tension curve
            fig.add_trace(
                go.Scatter(
                    x=-strain_tens * 1000,
                    y=stress_tens,
                    mode="lines",
                    name="Tension",
                    line=dict(
                        color="rgb(214, 39, 40)",
                        width=3,
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(214, 39, 40, 0.1)",
                )
            )

            # Mark peaks
            max_comp_idx = np.argmax(stress_comp)
            max_tens_idx = np.argmax(stress_tens)

            fig.add_annotation(
                x=strain_comp[max_comp_idx] * 1000,
                y=stress_comp[max_comp_idx],
                text=f"Peak<br>σ={stress_comp[max_comp_idx]:.1f} MPa",
                showarrow=True,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgb(31, 119, 180)",
                borderwidth=1,
                font=dict(size=10, color="#1a1a1a"),
            )

            fig.add_annotation(
                x=-strain_tens[max_tens_idx] * 1000,
                y=stress_tens[max_tens_idx],
                text=f"ft={stress_tens[max_tens_idx]:.3f} MPa",
                showarrow=True,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgb(214, 39, 40)",
                borderwidth=1,
                font=dict(size=10, color="#1a1a1a"),
            )

            # Add zero lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

            # Update layout
            comp_name = {
                "collins_mitchell": "Collins & Mitchell",
                "de_nicolo": "De Nicolo, Pani y Pozzo",
                "kent_park": "Kent & Park",
                "desayi_krishnan": "Desayi & Krishnan",
            }[compression_model]

            tens_name = {
                "collins_mitchell": "Collins & Mitchell",
                "restrepo_rodriguez": "Restrepo-Rodríguez",
                "fields_bischoff": "Fields & Bischoff",
            }[tension_model]

            fig.update_layout(
                title=f"Stress-Strain Diagram (f'c = {fc} MPa)<br><sub>Compression: {comp_name} | Tension: {tens_name}</sub>",
                xaxis_title="Strain (‰ = mm/m)",
                yaxis_title="Stress (MPa)",
                hovermode="x unified",
                template="plotly_white",
                height=500,
                showlegend=True,
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        elif view_mode == "comparison":
            # Comparison mode
            if not selected_fc_values:
                st.warning("Please select at least one f'c value")
            else:
                # Color mapping
                color_map = {
                    20: "rgb(31, 119, 180)",
                    28: "rgb(255, 127, 14)",
                    35: "rgb(44, 160, 44)",
                    40: "rgb(214, 39, 40)",
                    45: "rgb(148, 103, 189)",
                    56: "rgb(140, 86, 75)",
                }

                comp_name = {
                    "collins_mitchell": "Collins & Mitchell",
                    "de_nicolo": "De Nicolo, Pani y Pozzo",
                    "kent_park": "Kent & Park",
                    "desayi_krishnan": "Desayi & Krishnan",
                }[compression_model]

                if view_type == "split":
                    # Split view: One graph for compression, one for tension
                    col1, col2 = st.columns(2)

                    # Compression graph
                    with col1:
                        fig_comp = go.Figure()

                        for fc_val in selected_fc_values:
                            curves = generate_curves(
                                fc_prime=fc_val,
                                compression_model=compression_model,
                                tension_model=tension_model,
                            )

                            strain_comp = curves["strain_comp"]
                            stress_comp = curves["stress_comp"]

                            max_idx = np.argmax(stress_comp)
                            max_stress = stress_comp[max_idx]
                            max_strain = strain_comp[max_idx]

                            fig_comp.add_trace(
                                go.Scatter(
                                    x=strain_comp * 1000,
                                    y=stress_comp,
                                    mode="lines",
                                    name=f"f'c = {fc_val} MPa",
                                    line=dict(color=color_map[fc_val], width=3),
                                )
                            )

                            fig_comp.add_annotation(
                                x=max_strain * 1000,
                                y=max_stress,
                                text=f"{fc_val}MPa<br>σ={max_stress:.1f}",
                                showarrow=True,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor=color_map[fc_val],
                                borderwidth=1,
                                font=dict(size=8, color="#1a1a1a"),
                                ax=20,
                                ay=-20,
                            )

                        fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_comp.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

                        fig_comp.update_layout(
                            title=f"Compression Curves - {comp_name}",
                            xaxis_title="Strain (‰ = mm/m)",
                            yaxis_title="Stress (MPa)",
                            hovermode="x unified",
                            template="plotly_white",
                            height=550,
                            showlegend=True,
                        )

                        st.plotly_chart(fig_comp, use_container_width=True)

                    # Tension graph
                    with col2:
                        fig_tens = go.Figure()

                        for fc_val in selected_fc_values:
                            curves = generate_curves(
                                fc_prime=fc_val,
                                compression_model=compression_model,
                                tension_model=tension_model,
                            )

                            strain_tens = curves["strain_tens"]
                            stress_tens = curves["stress_tens"]

                            max_idx = np.argmax(stress_tens)
                            max_stress = stress_tens[max_idx]
                            max_strain = strain_tens[max_idx]

                            fig_tens.add_trace(
                                go.Scatter(
                                    x=-strain_tens * 1000,
                                    y=stress_tens,
                                    mode="lines",
                                    name=f"f't = {max_stress:.3f} MPa",
                                    line=dict(color=color_map[fc_val], width=3),
                                )
                            )

                            fig_tens.add_annotation(
                                x=-max_strain * 1000,
                                y=max_stress,
                                text=f"{fc_val}MPa<br>ft={max_stress:.3f}",
                                showarrow=True,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor=color_map[fc_val],
                                borderwidth=1,
                                font=dict(size=8, color="#1a1a1a"),
                                ax=-20,
                                ay=-20,
                            )

                        fig_tens.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig_tens.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

                        fig_tens.update_layout(
                            title="Tension Curves",
                            xaxis_title="Strain (‰ = mm/m)",
                            yaxis_title="Stress (MPa)",
                            hovermode="x unified",
                            template="plotly_white",
                            height=550,
                            showlegend=True,
                        )

                        st.plotly_chart(fig_tens, use_container_width=True)

                else:  # overlapped view
                    # Individual graphs for each fc
                    for fc_val in selected_fc_values:
                        curves = generate_curves(
                            fc_prime=fc_val,
                            compression_model=compression_model,
                            tension_model=tension_model,
                        )

                        strain_comp = curves["strain_comp"]
                        stress_comp = curves["stress_comp"]
                        strain_tens = curves["strain_tens"]
                        stress_tens = curves["stress_tens"]

                        fig = go.Figure()

                        max_comp_idx = np.argmax(stress_comp)
                        max_tens_idx = np.argmax(stress_tens)

                        # Compression
                        fig.add_trace(
                            go.Scatter(
                                x=strain_comp * 1000,
                                y=stress_comp,
                                mode="lines",
                                name="Compression",
                                line=dict(color=color_map[fc_val], width=3),
                                fill="tozeroy",
                                fillcolor=f"rgba({color_map[fc_val][4:-1]}, 0.15)",
                            )
                        )

                        fig.add_annotation(
                            x=strain_comp[max_comp_idx] * 1000,
                            y=stress_comp[max_comp_idx],
                            text=f"Peak Compression<br>σ={stress_comp[max_comp_idx]:.1f} MPa<br>ε={strain_comp[max_comp_idx]*1000:.3f}‰",
                            showarrow=True,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor=color_map[fc_val],
                            borderwidth=2,
                            font=dict(size=10, color=color_map[fc_val]),
                        )

                        # Tension
                        fig.add_trace(
                            go.Scatter(
                                x=-strain_tens * 1000,
                                y=stress_tens,
                                mode="lines",
                                name="Tension",
                                line=dict(color="rgb(214, 39, 40)", width=3),
                                fill="tozeroy",
                                fillcolor="rgba(214, 39, 40, 0.15)",
                            )
                        )

                        fig.add_annotation(
                            x=-strain_tens[max_tens_idx] * 1000,
                            y=stress_tens[max_tens_idx],
                            text=f"Peak Tension<br>ft={stress_tens[max_tens_idx]:.3f} MPa",
                            showarrow=True,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgb(214, 39, 40)",
                            borderwidth=2,
                            font=dict(size=10, color="rgb(214, 39, 40)"),
                        )

                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

                        fig.update_layout(
                            title=f"f'c = {fc_val} MPa - Compression & Tension (Overlapped)",
                            xaxis_title="Strain (‰ = mm/m)",
                            yaxis_title="Stress (MPa)",
                            hovermode="x unified",
                            template="plotly_white",
                            height=500,
                            showlegend=True,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Final overlay graph
                    st.markdown("---")
                    st.subheader("All Curves Overlapped")

                    fig_final = go.Figure()

                    for fc_val in selected_fc_values:
                        curves = generate_curves(
                            fc_prime=fc_val,
                            compression_model=compression_model,
                            tension_model=tension_model,
                        )

                        strain_comp = curves["strain_comp"]
                        stress_comp = curves["stress_comp"]
                        strain_tens = curves["strain_tens"]
                        stress_tens = curves["stress_tens"]

                        fig_final.add_trace(
                            go.Scatter(
                                x=strain_comp * 1000,
                                y=stress_comp,
                                mode="lines",
                                name=f"Comp - f'c={fc_val}",
                                line=dict(color=color_map[fc_val], width=2.5),
                            )
                        )

                        fig_final.add_trace(
                            go.Scatter(
                                x=-strain_tens * 1000,
                                y=stress_tens,
                                mode="lines",
                                name=f"Tens - f'c={fc_val}",
                                line=dict(color=color_map[fc_val], width=2.5, dash="dash"),
                            )
                        )

                    fig_final.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_final.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

                    fig_final.update_layout(
                        title=f"Complete Overlay - All Curves ({comp_name})",
                        xaxis_title="Strain (‰ = mm/m)",
                        yaxis_title="Stress (MPa)",
                        hovermode="x unified",
                        template="plotly_white",
                        height=650,
                        showlegend=True,
                        legend=dict(
                            x=0.99,
                            y=0.01,
                            xanchor="right",
                            yanchor="bottom",
                            font=dict(size=8),
                        ),
                    )

                    st.plotly_chart(fig_final, use_container_width=True)

        elif view_mode == "combinations":
            # Show five combination scenarios
            fc_base = 30.0

            combinations = [
                ("collins_mitchell", "collins_mitchell", "1.1: Collins & Mitchell (Comp + Tens)"),
                ("kent_park", "restrepo_rodriguez", "1.2: Kent & Park + Restrepo-Rodríguez"),
                ("desayi_krishnan", "restrepo_rodriguez", "1.3: Desayi & Krishnan + Restrepo-Rodríguez"),
                ("kent_park", "fields_bischoff", "1.4: Kent & Park + Fields & Bischoff"),
                ("desayi_krishnan", "fields_bischoff", "1.5: Desayi & Krishnan + Fields & Bischoff"),
            ]

            for comp_model, tens_model, title in combinations:
                with st.container(border=True):
                    st.markdown(f"### {title}")

                    curves = generate_curves(
                        fc_prime=fc_base,
                        compression_model=comp_model,
                        tension_model=tens_model,
                    )

                    strain_comp = curves["strain_comp"]
                    stress_comp = curves["stress_comp"]
                    strain_tens = curves["strain_tens"]
                    stress_tens = curves["stress_tens"]

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=strain_comp * 1000,
                            y=stress_comp,
                            mode="lines",
                            name="Compression",
                            line=dict(color="rgb(31, 119, 180)", width=2.5),
                            fill="tozeroy",
                            fillcolor="rgba(31, 119, 180, 0.1)",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=-strain_tens * 1000,
                            y=stress_tens,
                            mode="lines",
                            name="Tension",
                            line=dict(color="rgb(214, 39, 40)", width=2.5),
                            fill="tozeroy",
                            fillcolor="rgba(214, 39, 40, 0.1)",
                        )
                    )

                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

                    fig.update_layout(
                        title=f"f'c = {fc_base} MPa",
                        xaxis_title="Strain (‰)",
                        yaxis_title="Stress (MPa)",
                        hovermode="x unified",
                        template="plotly_white",
                        height=350,
                        showlegend=True,
                        margin=dict(l=50, r=50, t=50, b=50),
                    )

                    st.plotly_chart(fig, use_container_width=True)

    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")
