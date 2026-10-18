import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculations.structural.moment_curvature import (
    PARK_BEAMS,
    PARK_ES,
    PARK_FC,
    PARK_FY,
    PSI_TO_MPA,
    curvature_ductility,
    material_functions,
    moment_curvature,
    park_beam_curve,
)
from styles.main import apply_global_styles


apply_global_styles()

CONCRETE_LABELS = {
    "park": "Park & Paulay (parábola + descenso lineal, hasta 0.004)",
    "mander": "Mander no confinado (εsp = 0.006)",
}
STEEL_LABELS = {
    "elastoplastic": "Elastoplástico",
    "mander": "Mander (meseta + endurecimiento)",
}


st.title("Moment-Curvature Analysis")

st.markdown(
    r"""
    Diagrama momento-curvatura de secciones rectangulares en flexión pura
    ($P=0$), por análisis de fibras.
    """
)

st.latex(r"\varphi = \frac{\varepsilon_{cm}}{kd} \qquad \sum F = 0 \qquad M = \sum F\,z")

with st.expander("Referencia y supuestos", expanded=False):
    st.markdown(
        r"""
        - Park, R., & Paulay, T. *Reinforced Concrete Structures*, Fig. 6.6
          (vigas 1 a 7: $h=1.1d$, $d'=0.1d$).
        - Mander, Priestley & Park (1988), concreto no confinado y acero con
          endurecimiento ($\varepsilon_{sh}=0.01$, $\varepsilon_{su}=0.12$,
          $f_{su}=1.5f_y$, $E_{sh}=E_s/60$).
        - Se desprecia la tracción del concreto. $\mu_\varphi$ aproximada:
          $\varphi_y$ por secante al 75 % de $M_{max}$.
        """
    )

# ---------------------------------------------------------
# 1. Section
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("1. Section")
    source = st.radio("Source", ["Park & Paulay Fig. 6.6", "Custom section"], horizontal=True)

    if source == "Park & Paulay Fig. 6.6":
        beams = st.multiselect("Beams", options=list(PARK_BEAMS), default=list(PARK_BEAMS))
        table = pd.DataFrame(
            [{"Viga": b, "ρ": PARK_BEAMS[b][0], "ρ'": PARK_BEAMS[b][1]} for b in beams]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        fc, fy, es = PARK_FC, PARK_FY, PARK_ES
        st.caption(
            f"f″c = {fc:.1f} MPa (4000 psi), fy = {fy:.0f} MPa (40 ksi), Es = {es:,.0f} MPa."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        b_mm = c1.number_input("Width, b (mm)", min_value=50.0, value=300.0, step=10.0)
        h_mm = c2.number_input("Height, h (mm)", min_value=100.0, value=550.0, step=10.0)
        d_mm = c3.number_input("Depth, d (mm)", min_value=50.0, value=500.0, step=10.0)
        dp_mm = c4.number_input("Cover to compression steel, d' (mm)", min_value=10.0, value=50.0, step=5.0)
        c5, c6, c7, c8 = st.columns(4)
        as_mm2 = c5.number_input("As (mm²)", min_value=1.0, value=1900.0, step=50.0)
        asp_mm2 = c6.number_input("A's (mm²)", min_value=0.0, value=0.0, step=50.0)
        fc = c7.number_input("f'c (MPa)", min_value=5.0, value=28.0, step=1.0)
        fy = c8.number_input("fy (MPa)", min_value=100.0, value=420.0, step=10.0)
        es = st.number_input("Es (MPa)", min_value=100000.0, value=200000.0, step=1000.0)

# ---------------------------------------------------------
# 2. Models
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("2. Material Models")
    m1, m2 = st.columns(2)
    concrete = m1.selectbox("Concrete", list(CONCRETE_LABELS), format_func=CONCRETE_LABELS.get, index=1)
    steel = m2.selectbox("Steel", list(STEEL_LABELS), format_func=STEEL_LABELS.get, index=1)
    compare = st.checkbox("Overlay Park & Paulay models (elastoplastic steel)", value=True)

# ---------------------------------------------------------
# 3. Curve
# ---------------------------------------------------------

if st.button("Generate Moment-Curvature Diagram", type="primary", use_container_width=True):
    try:
        fig = go.Figure()
        rows = []
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

        if source == "Park & Paulay Fig. 6.6":
            if not beams:
                raise ValueError("Select at least one beam.")
            for i, beam in enumerate(beams):
                color = palette[(beam - 1) % len(palette)]
                curve = park_beam_curve(beam, concrete, steel)
                x, y = curve["phi_d"], curve["m_norm"] / PSI_TO_MPA
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Viga {beam}", line=dict(color=color, width=3)))
                phi_y, mu = curvature_ductility(curve["phi"], curve["moment"])
                rows.append({"Viga": beam, "M/bd² máx (psi)": round(float(y.max())),
                             "M/bd² máx (MPa)": round(float(y.max()) * PSI_TO_MPA, 2),
                             "φd final": round(float(x[-1]), 4), "μφ aprox.": round(mu, 1)})
                if compare:
                    ref = park_beam_curve(beam, "park", "elastoplastic")
                    fig.add_trace(go.Scatter(x=ref["phi_d"], y=ref["m_norm"] / PSI_TO_MPA, mode="lines",
                                             name=f"Viga {beam} (Park)", line=dict(color=color, width=1.5, dash="dot"),
                                             showlegend=False))
            fig.update_layout(xaxis_title="Curvatura adimensional, φd", yaxis_title="M/bd² (lb/pulg²)")
        else:
            if d_mm >= h_mm or dp_mm >= d_mm:
                raise ValueError("Require d' < d < h.")
            layers = [(d_mm, as_mm2)] + ([(dp_mm, asp_mm2)] if asp_mm2 > 0 else [])
            concrete_fn, steel_fn, strain_max = material_functions(concrete, steel, fc, fy, es)
            curve = moment_curvature(b_mm, h_mm, layers, concrete_fn, steel_fn, strain_max)
            x, y = curve["phi"] * 1000.0, curve["moment"] / 1e6
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Selected models", line=dict(color="#1f77b4", width=3)))
            phi_y, mu = curvature_ductility(curve["phi"], curve["moment"])
            rows.append({"M máx (kN·m)": round(float(y.max()), 1), "φ final (1/m)": round(float(x[-1]), 3),
                         "φy (1/m)": round(phi_y * 1000.0, 4), "μφ aprox.": round(mu, 1)})
            if compare:
                cf, sf, smax = material_functions("park", "elastoplastic", fc, fy, es)
                ref = moment_curvature(b_mm, h_mm, layers, cf, sf, smax)
                fig.add_trace(go.Scatter(x=ref["phi"] * 1000.0, y=ref["moment"] / 1e6, mode="lines",
                                         name="Park & Paulay", line=dict(color="#7f7f7f", width=2, dash="dot")))
            fig.update_layout(xaxis_title="Curvatura, φ (1/m)", yaxis_title="Momento, M (kN·m)")

        fig.update_layout(template="plotly_white", height=520, hovermode="x unified",
                          title="Diagrama momento-curvatura")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resultados")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Línea punteada: modelos de Park & Paulay. μφ = φu/φy con φu al final del cálculo.")

    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")
