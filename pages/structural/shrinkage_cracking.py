import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculations.structural.shrinkage_cracking import (
    EPS_SH_DEFAULT,
    PHI_DEFAULT,
    crack_width_vs_rho,
    gilbert_crack_width,
    kgf_cm2_to_mpa,
)
from styles.main import apply_global_styles


apply_global_styles()

st.title("Shrinkage Cracking in Restrained Slabs")

st.markdown(
    r"""
    Ancho medio de fisura de una losa totalmente restringida por contracción
    (Gilbert, 1992). Franja de 1 m de ancho.
    """
)

with st.expander("Referencia y supuestos", expanded=False):
    st.markdown(
        r"""
        - Gilbert, R. I. (1992). *Shrinkage cracking in fully restrained
          concrete members.* ACI Structural Journal, 89(2), 141-149.
        - $E_c = 4700\sqrt{f'_c}$ y $f_t = 0.4\sqrt{f'_c}$ (MPa).
        - Cuantía sobre el área bruta: $\rho = A_s/A_c$, con $A_c = 1000\,e$.
        - Si el acero fluye en la grieta queda una sola grieta, con
          $N = f_y A_s$ (Ec. 32-34 del artículo).
        """
    )

# ---------------------------------------------------------
# 1. Data
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("1. Data")

    c1, c2, c3, c4 = st.columns(4)
    fc_kgf = c1.number_input("f'c (kgf/cm²)", min_value=100.0, value=210.0, step=10.0)
    fy_kgf = c2.number_input("fy (kgf/cm²)", min_value=2000.0, value=4200.0, step=100.0)
    length_m = c3.number_input("Length, L (m)", min_value=0.5, value=4.0, step=0.5)
    thickness_cm = c4.number_input("Thickness, e (cm)", min_value=3.0, value=10.0, step=1.0)

    c5, c6, c7, c8 = st.columns(4)
    db_mm = c5.number_input("Bar diameter, db (mm)", min_value=4.0, value=8.0, step=1.0)
    rho_mode = c6.selectbox(
        "Reinforcement ratio",
        ["14/fy (no expuesto)", "21/fy (severamente expuesto)", "Personalizada"],
        index=1,
    )
    if rho_mode == "Personalizada":
        rho_pct = c7.number_input("ρ (%)", min_value=0.05, value=0.40, step=0.05, format="%.3f")
        rho = rho_pct / 100.0
    elif rho_mode.startswith("14"):
        rho = 14.0 / fy_kgf
    else:
        rho = 21.0 / fy_kgf
    c7.caption(f"ρ = {rho:.5f} ({rho * 100:.3f} %)")
    phi_creep = c8.number_input("Creep coefficient, φ*", min_value=0.0, value=PHI_DEFAULT, step=0.1)
    eps_sh_micro = st.number_input(
        "Final shrinkage strain, ε*sh (×10⁻⁶)",
        value=EPS_SH_DEFAULT * 1e6,
        step=50.0,
    )

fc = kgf_cm2_to_mpa(fc_kgf)
fy = kgf_cm2_to_mpa(fy_kgf)
common = dict(
    length_mm=length_m * 1000.0,
    thickness_mm=thickness_cm * 10.0,
    bar_diameter_mm=db_mm,
    fy_mpa=fy,
    creep_coefficient=phi_creep,
    shrinkage_strain=eps_sh_micro * 1e-6,
)

# ---------------------------------------------------------
# 2. Result
# ---------------------------------------------------------

try:
    result = gilbert_crack_width(rho=rho, fc_mpa=fc, **common)
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.container(border=True):
    st.subheader("2. Result")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crack width, w", f"{result['w_mm']:.2f} mm")
    m2.metric("Number of cracks", f"{result['n_cracks']}")
    m3.metric("Spacing, s", f"{result['spacing_mm']:,.0f} mm")
    m4.metric("N final", f"{result['n_inf_n'] / 1000:.1f} kN/m")

    if result["yielded"]:
        st.warning(
            f"El acero fluye en la grieta: la prueba elástica exige "
            f"{result['sigma_s2_trial_mpa']:.0f} MPa > fy = {fy:.0f} MPa. "
            "Queda una sola grieta y w no es admisible."
        )
    elif result["w_mm"] <= 0.3:
        st.success(f"w = {result['w_mm']:.2f} mm ≤ 0.3 mm (exposición severa).")
    else:
        st.info(f"w = {result['w_mm']:.2f} mm > 0.3 mm; aceptable solo en ambiente protegido.")

    with st.expander("Cálculo paso a paso", expanded=False):
        rows = [
            {
                "Paso": label,
                "Valor": value if isinstance(value, str) else (None if value is None else float(value)),
                "Unidad": unit,
            }
            for label, value, unit in result["steps"]
        ]
        df = pd.DataFrame(rows)
        df["Valor"] = df["Valor"].map(
            lambda v: "" if v is None else (v if isinstance(v, str) else f"{v:,.4g}")
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# 3. Width vs rho
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("3. Crack width vs reinforcement ratio")

    fc_options = [175, 210, 280, 350]
    selected = st.multiselect(
        "Concrete strengths (kgf/cm²)",
        options=sorted(set(fc_options + [round(fc_kgf)])),
        default=sorted(set(fc_options + [round(fc_kgf)])),
    )
    w_max = st.slider("Max w on chart (mm)", 0.5, 8.0, 1.6, 0.1)

    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#666666"]
    fig = go.Figure()
    for color, f in zip(palette, selected):
        rho_arr, w_arr = crack_width_vs_rho(
            fc_mpa=kgf_cm2_to_mpa(f),
            **{k: v for k, v in common.items()},
        )
        fig.add_trace(
            go.Scatter(x=rho_arr * 100, y=w_arr, mode="lines", name=f"f'c = {f}", line=dict(color=color, width=2))
        )
    for ratio, label, dash in ((14.0 / fy_kgf, "14/fy", "dash"), (21.0 / fy_kgf, "21/fy", "dot")):
        fig.add_vline(x=ratio * 100, line=dict(color="black", dash=dash, width=1),
                      annotation_text=label, annotation_position="top")
    fig.add_hline(y=0.3, line=dict(color="gray", width=1), annotation_text="0.3 mm")
    fig.add_hline(y=0.5, line=dict(color="gray", width=1, dash="dashdot"), annotation_text="0.5 mm")
    fig.update_layout(
        xaxis_title="Reinforcement ratio, ρ (%)",
        yaxis_title="Crack width, w (mm)",
        yaxis=dict(range=[0, w_max]),
        template="plotly_white",
        height=480,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Tramos fuera de escala: el acero fluye en la grieta (w de varios mm). "
        "El quiebre de las curvas se debe a que el número de grietas es entero."
    )
