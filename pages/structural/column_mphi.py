import uuid
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from calculations.structural.column_analysis import (
    AnalysisSettings,
    ConcreteProperties,
    MomentCurvatureAnalysis,
    build_column_model,
    ideal_flexural_strength,
)
from calculations.structural.column_confinement import ConfinementResult, confine_rectangular_core
from calculations.structural.column_interaction import (
    analysis_interaction,
    code_interaction_curve,
    default_axial_range,
    interaction_dataframe,
)
from calculations.structural.column_examples import (
    HD16_STEEL,
    SECTION_SIZE_MM,
    SOESIANAWATI_UNITS,
    unit_model,
    unit_section,
)
from calculations.structural.column_materials import ReinforcingSteel
from calculations.structural.column_response import ColumnData, compute_column_response, yield_penetration_length
from calculations.structural.column_section import (
    BarLayer,
    RectangularSection,
    TransverseReinforcement,
    equivalent_octagonal_area_mm2,
)
from calculations.structural.column_variants import Variant, run_variants, variants_dataframe
from calculations.structural.concrete_link import concrete_from_panel
from styles.main import apply_global_styles
from utils.column_report import build_column_report, interpretation_lines, write_docx, write_pdf
from utils.shared_state import read_concrete, read_rectangle
from visualization.column_plots import (
    PHI_UNIT,
    convergence_figure,
    displacement_components_figure,
    ductility_figure,
    force_displacement_figure,
    interaction_figure,
    material_curves_figure,
    moment_curvature_figure,
    neutral_axis_figure,
    section_figure,
    strain_evolution_figure,
    strain_profile_figure,
    variants_figure,
)


apply_global_styles()

RESULT_KEY = "column_mphi_result"
INTERACTION_KEY = "column_mphi_interaction"
VARIANTS_KEY = "column_mphi_variants"
REPORT_KEY = "column_mphi_report"
COLUMN_KEY = "column_mphi_column"
OWN_DATA = "Datos propios (parten de los de la Unidad 1)"
EXAMPLES = {f"Ejemplo — Soesianawati (1986), {unit.name}": number for number, unit in SOESIANAWATI_UNITS.items()}
EXAMPLES[OWN_DATA] = None

COLUMN_LABELS = {
    "step": "Incremento",
    "curvature_urad_per_mm": "Curvatura φ (µrad/mm)",
    "moment_kNm": "Momento M (kN·m)",
    "lateral_force_kN": "Fuerza lateral V (kN)",
    "elastic_displacement_mm": "Δe flexión elástica (mm)",
    "shear_displacement_mm": "Δs cortante (mm)",
    "plastic_displacement_mm": "Δp plástico (mm)",
    "displacement_mm": "Δ total (mm)",
    "plastic_rotation_rad": "θp (rad)",
    "displacement_ductility": "μΔ = Δ/Δy",
    "drift_ratio": "Deriva Δ/L",
    "limit_state": "Estado límite alcanzado",
}
INTERACTION_LABELS = {
    "axial_kN": "P (kN)",
    "axial_ratio": "P/(f'co·Ag)",
    "Mi_kNm": "Mi (kN·m)",
    "Mmax_kNm": "Mmax (kN·m)",
    "curvature_at_Mmax_urad_per_mm": "φ en Mmax (µrad/mm)",
    "phi_y_urad_per_mm": "φy (µrad/mm)",
    "phi_u_urad_per_mm": "φu (µrad/mm)",
    "ductility": "μφ = φu/φy",
    "termination": "Termina por",
    "error": "Error",
}
VARIANT_LABELS = {
    "name": "Configuración",
    "spacing_mm": "s (mm)",
    "stirrup_diameter_mm": "Ø estribo (mm)",
    "stirrup_fy_mpa": "fyh (MPa)",
    "axial_kN": "P (kN)",
    "fco_mpa": "f'co (MPa)",
    "ke": "ke",
    "fcc_mpa": "f'cc (MPa)",
    "ecc": "εcc",
    "Mi_kNm": "Mi (kN·m)",
    "Mmax_kNm": "Mmax (kN·m)",
    "phi_y_urad_per_mm": "φy (µrad/mm)",
    "phi_u_urad_per_mm": "φu (µrad/mm)",
    "ductility": "μφ",
    "termination": "Termina por",
    "error": "Error",
}
TABLE_LABELS = {
    "step": "Incremento",
    "curvature_urad_per_mm": "Curvatura φ (µrad/mm)",
    "moment_kNm": "Momento M (kN·m)",
    "axial_strain_eps0": "ε₀ (eje de referencia)",
    "neutral_axis_depth_mm": "Eje neutro c (mm)",
    "strain_extreme_cover": "ε fibra extrema",
    "strain_core_edge": "ε borde del núcleo",
    "strain_bar_compression": "ε barra comprimida",
    "strain_bar_tension": "ε barra traccionada",
    "curvature_ductility": "μφ = φ/φy",
    "iterations": "Iteraciones (Newton)",
    "axial_residual_kN": "Error axial N−P (kN)",
    "converged": "Convergió",
    "limit_state": "Estado límite alcanzado",
}


def same_layers(bars, reference_layers):
    """Compara las capas ingresadas (profundidad, número, diámetro) con las del ejemplo."""
    expected = [(layer.depth_mm, layer.count, layer.diameter_mm) for layer in reference_layers]
    return len(bars) == len(expected) and bool(np.allclose(np.array(bars, dtype=float), np.array(expected, dtype=float), atol=1e-6))


def is_unmodified_example(reference, current):
    """True si los datos son exactamente los publicados de la unidad `reference` (la comparación solo vale así)."""
    return (
        (current["b"], current["h"], current["cover"]) == (SECTION_SIZE_MM, SECTION_SIZE_MM, 13.0)
        and abs(current["fco"] - reference.unconfined_strength_mpa) < 1e-9
        and abs(current["eco"] - reference.strain_at_peak) < 1e-12
        and abs(current["P"] - reference.axial_load_n / 1e3) < 1e-6
        and current["steel"] == (HD16_STEEL.yield_stress, HD16_STEEL.ultimate_stress, HD16_STEEL.elastic_modulus,
                                 HD16_STEEL.hardening_strain, HD16_STEEL.ultimate_strain, HD16_STEEL.hardening_modulus)
        and same_layers(current["bars"], unit_section(reference).layers)
        and current["stirrup"][:3] == (reference.stirrup_diameter_mm, reference.stirrup_spacing_mm, reference.stirrup_yield_mpa)
        and current["stirrup"][3].startswith("Cuadrado") and current["stirrup"][6:] == (None, None)
        and current["confinement"] == (None, None)
        and current["ft"] is None and current["ec"] is None and current["spall"] == 0.008
    )


st.title("COLUMN — Análisis Momento–Curvatura")

with st.container(border=True):
    st.markdown(
        "Esta herramienta permite analizar el comportamiento no lineal a flexión de secciones de concreto "
        "armado sometidas a una carga axial determinada. Mediante modelos constitutivos del concreto y del "
        "acero, compatibilidad de deformaciones y equilibrio de fuerzas, se obtiene el diagrama "
        "momento–curvatura, útil para evaluar la resistencia y ductilidad de la sección."
    )

# ---------------------------------------------------------
# 1. Descripción y fundamento teórico
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("1. Descripción y fundamento teórico")
    st.markdown(
        "Reconstrucción numérica, **monotónica**, del análisis de COLUMN (Mander, 1984) para secciones "
        "rectangulares. Se impone una curvatura φ y se busca la deformación axial ε₀ que equilibra la carga "
        "axial P; con ella se obtiene el momento M. Se repite con φ creciente hasta un estado límite."
    )
    st.latex(r"\varepsilon(y)=\varepsilon_0+\phi\,y \qquad N=\sum f_{c,i}A_{c,i}+\sum f_{s,j}A_{s,j}=P \qquad M=\sum f\,A\,y")
    st.latex(r"\Delta\varepsilon_0=\frac{P-N}{EA},\qquad EA=\sum E_t\,A \qquad(\text{Newton sobre }\varepsilon_0)")

    with st.expander("Modelos constitutivos, confinamiento y estados límite", expanded=False):
        st.markdown(
            r"""
            **Concreto** (Popovics, Mander 1984 Ec. 3.33): la misma ecuación para el núcleo confinado
            ($f'_{cc},\varepsilon_{cc}$) y el recubrimiento ($f'_{co},\varepsilon_{co}$):
            """
        )
        st.latex(r"f_c=\frac{f'_{cc}\,x\,r}{r-1+x^{r}},\quad x=\frac{\varepsilon}{\varepsilon_{cc}},\quad r=\frac{E_c}{E_c-E_{sec}},\quad E_{sec}=\frac{f'_{cc}}{\varepsilon_{cc}}")
        st.markdown(
            r"""
            - **Recubrimiento:** desciende linealmente desde $2\varepsilon_{co}$ hasta cero en $\varepsilon_{spall}$ y,
              una vez descascarado, no vuelve a resistir. **Núcleo:** confinado por los estribos.
            - **Tracción del concreto:** lineal hasta $f'_t$; luego fisurado (resistencia nula).
            - **Confinamiento:** $k_e$ (Ec. 3.17), $f'_{l}=k_e\rho\,f_{yh}$ (Ec. 3.18) y $f'_{cc}$ por el
              procedimiento de 6 pasos de la superficie de resistencia (§3.4.5.1);
              $\varepsilon_{cc}=\varepsilon_{co}\,[1+5(f'_{cc}/f'_{co}-1)]$ (Ec. 3.32).
            - **Acero** (Ec. 2.1–2.10): elástico, meseta de fluencia y endurecimiento
              $f_s=f_{su}+(f_y-f_{su})\left|\frac{\varepsilon_{su}-\varepsilon_s}{\varepsilon_{su}-\varepsilon_{sh}}\right|^{P}$,
              igual en tracción y compresión.
            - **Curvatura de fluencia** (Ec. 7.60): $\phi_y=\phi'_y\,M_i/M_y$, con $\phi'_y$ y $M_y$ en la primera
              fluencia del acero y $M_i$ la resistencia ideal del método de la norma ($\varepsilon_c=0.003$).
            - **Fractura del estribo** (balance de energía, Ec. 7.61–7.66): el análisis termina cuando el trabajo
              sobre el núcleo y las barras comprimidas supera
              $SE=105\,A_b L_s/s+\sum W_c\,A_{cc}\,0.017\sqrt{f'_c}$ (MJ/m).
            """
        )

    with st.expander("Referencias (documento, sección, página y ecuación)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    ["Mander (1984), Research Report 84-2", "§7.4.2 Moment-Curvature Analysis", "280–290 (PDF 299–309)", "7.41–7.66: algoritmo, EA, φy, energía"],
                    ["Mander (1984)", "Cap. 2 §2.2 Skeleton branch", "11–14 (PDF 30–33)", "2.1–2.10: acero"],
                    ["Mander (1984)", "§3.3.3–3.3.4, §3.4.5, §3.6.2", "55–57, 62–66, 69–71", "3.16–3.21, 3.29–3.34: confinamiento y concreto"],
                    ["Mander (1984)", "§3.8.2 Energy balance theory", "86–91 (PDF 105–110)", "3.65–3.76: energía de la fractura de estribos"],
                    ["Mander (1984)", "Apéndice A: programa COLUMN", "A1–A41 (PDF 462–502)", "Guía de usuario y listado FORTRAN"],
                    ["Soesianawati (1986), Research Report 86-10", "§5.5.2 Cyclic Moment-Curvature Analysis", "50–51 (PDF 64–65)", "Describe COLUMN y remite a §7.4.2 de Mander"],
                    ["Soesianawati (1986)", "Tabla 5.1, Tabla 8.3", "45, 140", "Datos de validación"],
                ],
                columns=["Documento", "Sección", "Página impresa", "Ecuaciones / contenido"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Las páginas son las impresas; el PDF digital de Soesianawati lleva 14 más y el de Mander, 19 más.")

    with st.expander("Alcance y diferencias respecto al programa COLUMN original", expanded=False):
        st.markdown(
            """
            - **Implementado:** carga monotónica, sección rectangular simétrica, flexión respecto a un eje, carga
              axial constante, concreto confinado de Mander, acero con endurecimiento, tracción del concreto,
              descascaramiento, balance de energía para la fractura del estribo, curvatura de fluencia y Mi.
            - **No implementado:** ramas cíclicas de carga-descarga, efecto de la velocidad de deformación,
              penetración de fluencia, deformación por cortante, secciones circulares y huecas, y pandeo de barras.
            - La función esfuerzo-deformación se evalúa sobre la curva monotónica ("skeleton"); no se modela la
              descarga elástica de fibras (en COLUMN sí se hace en ciclos).
            - **No es una traducción literal** del FORTRAN de COLUMN (que sí figura en el Apéndice A de la tesis):
              se siguió el procedimiento descrito en §7.4.2 y los modelos de los Cap. 2 y 3.
              La validación contra datos publicados está en `tests/test_column_mphi.py`.
            """
        )

# ---------------------------------------------------------
# 2. Geometría y materiales
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("2. Geometría y materiales")
    choice = st.selectbox(
        "Datos de partida",
        list(EXAMPLES),
        help="Los ejemplos usan datos publicados (Soesianawati, 1986: Tablas 2.1, 3.1 y Fig. 5.8). Puede editarlos.",
    )
    example_number = EXAMPLES[choice]
    unit = SOESIANAWATI_UNITS[example_number or 1]
    PREFIX = f"col{list(EXAMPLES).index(choice)}"
    if example_number:
        st.caption(
            "Datos **ilustrativos** provenientes de una referencia verificable: columnas de ensayo de Soesianawati (1986), "
            "sección de 400 × 400 mm, 12 barras HD16 y estribos cuadrados + octagonales superpuestos."
        )

    def number(label, value, key, **kwargs):
        """number_input con clave propia de cada ejemplo, para que editar uno no afecte a los demás."""
        return st.number_input(label, value=value, key=f"{PREFIX}_{key}", **kwargs)

    # Datos publicados por otros paneles (Section Properties y Concrete Stress-Strain)
    shared_rectangle = read_rectangle()
    shared_concrete = read_concrete()
    panel_concrete = None
    if shared_concrete is not None:
        try:
            panel_concrete = concrete_from_panel(shared_concrete["fc"], shared_concrete["compression_model"], shared_concrete["tension_model"])
        except ValueError as error:
            st.warning(f"No se pudo usar el concreto de «Concrete Stress-Strain»: {error}")

    st.markdown("**Geometría**")
    geometry_source = st.radio("Origen de b y h", ["Manual", "Tomar de «Section Properties»"], horizontal=True, key=f"{PREFIX}_geomsrc",
                               help="Section Properties publica el rectángulo (Custom Dimensions → Rectangle) con su unidad.")
    from_panel_geometry = geometry_source.startswith("Tomar") and shared_rectangle is not None
    if geometry_source.startswith("Tomar") and shared_rectangle is None:
        st.info("«Section Properties» aún no tiene un rectángulo: ábralo, elija Custom Dimensions → Rectangle e ingrese b y h. "
                "Mientras tanto se usan los valores manuales.")
    default_width, default_depth = SECTION_SIZE_MM, SECTION_SIZE_MM
    geometry_tag = "m"
    if from_panel_geometry:
        default_width, default_depth = max(100.0, shared_rectangle["width_mm"]), max(100.0, shared_rectangle["height_mm"])
        geometry_tag = f"p{default_width:g}x{default_depth:g}"
        st.caption(f"Tomado de Section Properties: b = {shared_rectangle['width']:g} {shared_rectangle['unit']}, "
                   f"h = {shared_rectangle['height']:g} {shared_rectangle['unit']} (= {default_width:g} × {default_depth:g} mm). Puede editarlos.")
    g1, g2, g3 = st.columns(3)
    width_mm = number("Ancho, b (mm)", default_width, f"b_{geometry_tag}", min_value=100.0, step=10.0)
    depth_mm = number("Peralte, h (mm) — flexión respecto al eje x", default_depth, f"h_{geometry_tag}", min_value=100.0, step=10.0)
    cover_mm = number("Recubrimiento libre hasta el estribo (mm)", 13.0, "cover", min_value=0.0, step=1.0)

    st.markdown("**Concreto**")
    concrete_source = st.radio("Origen de los datos del concreto", ["Manual", "Tomar de «Concrete Stress-Strain»"], horizontal=True,
                               key=f"{PREFIX}_concsrc",
                               help="Toma f'c, la deformación del pico, Ec y f't de las curvas del modelo elegido en ese panel (vista «Single fc' value»).")
    from_panel_concrete = concrete_source.startswith("Tomar") and panel_concrete is not None
    if concrete_source.startswith("Tomar") and panel_concrete is None:
        st.info("«Concrete Stress-Strain» aún no publicó datos: ábralo en la vista «Single fc' value» y elija f'c y los modelos. "
                "Mientras tanto se usan los valores manuales.")
    concrete_tag = "m"
    default_fco, default_eco = float(unit.unconfined_strength_mpa), float(unit.strain_at_peak)
    if from_panel_concrete:
        default_fco, default_eco = panel_concrete.unconfined_strength, panel_concrete.strain_at_peak
        concrete_tag = f"p{default_fco:g}_{shared_concrete['compression_model']}_{shared_concrete['tension_model']}"
        st.caption(f"Tomado de Concrete Stress-Strain — {panel_concrete.description}. εco = {panel_concrete.strain_at_peak:.5f}, "
                   f"Ec = {panel_concrete.elastic_modulus:,.0f} MPa, f't = {panel_concrete.tensile_strength:.2f} MPa. Puede editarlos.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fco = number("f'co, resistencia in situ (MPa)", default_fco, f"fco_{concrete_tag}", min_value=5.0, step=1.0, format="%.2f")
    with c2:
        eps_co = number("εco (deformación en f'co)", default_eco, f"eco_{concrete_tag}", min_value=0.001, step=0.0001, format="%.5f")
    with c3:
        eps_spall = number("εspall (descascaramiento)", 0.008, "spall", min_value=0.004, step=0.001, format="%.4f",
                           help="Dato del usuario en COLUMN. La tesis de Mander usa 0.008 en sus estudios paramétricos (§7.4.3).")
    with c4:
        auto_ft = st.checkbox("f't = 0.6√f'co (supuesto)", value=not from_panel_concrete, key=f"{PREFIX}_autoft_{concrete_tag}",
                              help="No es un dato de COLUMN: es un valor típico. Desmarque para ingresar f't (0 desprecia la tracción).")
        ft = None if auto_ft else number("f't (MPa)", panel_concrete.tensile_strength if from_panel_concrete else 0.0, f"ft_{concrete_tag}",
                                         min_value=0.0, step=0.1)
    auto_ec = st.checkbox("Ec = 5000√f'co (Mander §3.6.2)", value=not from_panel_concrete, key=f"{PREFIX}_autoec_{concrete_tag}")
    ec = None if auto_ec else number("Ec (MPa)", panel_concrete.elastic_modulus if from_panel_concrete else 30000.0, f"ec_{concrete_tag}",
                                     min_value=1000.0, step=500.0)

    st.markdown("**Acero longitudinal** (curva monotónica, Ec. 2.1–2.10)")
    s1, s2, s3 = st.columns(3)
    fy = number("fy (MPa)", HD16_STEEL.yield_stress, "fy", min_value=100.0, step=10.0)
    fsu = number("fsu (MPa)", HD16_STEEL.ultimate_stress, "fsu", min_value=101.0, step=10.0)
    es = number("Es (MPa)", HD16_STEEL.elastic_modulus, "es", min_value=10000.0, step=1000.0)
    s4, s5, s6 = st.columns(3)
    esh = number("εsh (inicio del endurecimiento)", HD16_STEEL.hardening_strain, "esh", min_value=0.001, step=0.001, format="%.4f")
    esu = number("εsu (deformación en fsu)", HD16_STEEL.ultimate_strain, "esu", min_value=0.01, step=0.005, format="%.4f")
    esh_mod = number("Esh (módulo de endurecimiento, MPa)", HD16_STEEL.hardening_modulus, "Esh", min_value=100.0, step=100.0)

# ---------------------------------------------------------
# 3. Refuerzo longitudinal y transversal
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("3. Refuerzo longitudinal y transversal")
    st.markdown("**Barras longitudinales** — una fila por capa, de la cara comprimida (arriba) a la cara inferior")
    default_stirrup = TransverseReinforcement.from_legs(unit.stirrup_diameter_mm, unit.stirrup_spacing_mm, unit.stirrup_yield_mpa, 2, 2)
    default_layers = unit_section(unit).layers
    for bars_per_face, bar_size in ((4, 16.0), (3, 16.0), (2, 16.0), (2, 12.0), (2, 10.0)):
        try:  # armado perimetral que quepa en la geometría actual (con 400 × 400 es el publicado: 4 + 4 barras Ø16)
            default_layers = RectangularSection.with_perimeter_bars(
                width_mm, depth_mm, cover_mm, bars_per_face, bars_per_face, bar_size, default_stirrup).layers
            break
        except ValueError:
            continue
    layers_table = st.data_editor(
        pd.DataFrame(
            {
                "Profundidad (mm)": [layer.depth_mm for layer in default_layers],
                "N° de barras": [layer.count for layer in default_layers],
                "Diámetro (mm)": [layer.diameter_mm for layer in default_layers],
            }
        ),
        key=f"{PREFIX}_layers_{geometry_tag}",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Profundidad (mm)": st.column_config.NumberColumn(
                "Profundidad (mm)", help="Distancia de la cara comprimida al centro de las barras de la capa.",
                min_value=1.0, step=1.0, format="%.1f", required=True),
            "N° de barras": st.column_config.NumberColumn(
                "N° de barras", help="Barras de la capa, repartidas a todo el ancho (mínimo 2).",
                min_value=2, step=1, format="%d", required=True),
            "Diámetro (mm)": st.column_config.NumberColumn(
                "Diámetro (mm)", min_value=6.0, step=1.0, format="%.1f", required=True),
        },
    )
    st.caption(
        "La primera y la última fila son las capas de las caras superior e inferior; las filas intermedias son las barras "
        "de las caras laterales. Si cambia el recubrimiento o el diámetro del estribo, ajuste las profundidades."
    )
    try:
        rows = layers_table.dropna()
        layers = tuple(
            BarLayer(float(row["Profundidad (mm)"]), int(row["N° de barras"]) if float(row["N° de barras"]).is_integer() else float(row["N° de barras"]),
                     float(row["Diámetro (mm)"]))
            for _, row in rows.iterrows()
        )
        if len(layers) < 2:
            raise ValueError("Ingrese al menos 2 capas de barras (la superior y la inferior).")
    except ValueError as error:
        st.error(str(error))
        st.stop()
    layers_sorted = sorted(layers, key=lambda layer: layer.depth_mm)
    bars_signature = tuple((layer.depth_mm, int(layer.count), layer.diameter_mm) for layer in layers_sorted)

    st.markdown("**Estribos**")
    t1, t2, t3 = st.columns(3)
    stirrup_diameter = number("Diámetro del estribo (mm)", unit.stirrup_diameter_mm, "ds", min_value=4.0, step=1.0)
    stirrup_spacing = number("Espaciamiento, s (mm)", unit.stirrup_spacing_mm, "s", min_value=20.0, step=1.0)
    stirrup_fy = number("fyh del estribo (MPa)", unit.stirrup_yield_mpa, "fyh", min_value=100.0, step=10.0)
    arrangement = st.radio(
        "Configuración de estribos",
        ["Cuadrado + octagonal superpuestos (Soesianawati, Fig. 2.2)", "Estribo cerrado y ganchos (por número de ramas)"],
        key=f"{PREFIX}_arr",
        horizontal=True,
    )
    legs_x = legs_y = 2
    if arrangement.startswith("Estribo"):
        l1, l2 = st.columns(2)
        with l1:
            legs_x = number("Ramas paralelas al eje x (incluye las 2 del estribo perimetral)", 2, "lx", min_value=2, step=1)
        with l2:
            legs_y = number("Ramas paralelas al eje y (incluye las 2 del estribo perimetral)", 2, "ly", min_value=2, step=1)
    all_supported = st.checkbox(
        "Todas las barras longitudinales están apoyadas en esquinas de estribos o ganchos",
        value=True,
        key=f"{PREFIX}_allsup",
        help="Define los espaciamientos libres w' que reducen el confinamiento efectivo (Ec. 3.16-3.17).",
    )
    supported_tb = supported_sides = None
    if not all_supported:
        p1, p2 = st.columns(2)
        with p1:
            supported_tb = number("Barras apoyadas en la cara superior (e inferior)", 2, "sup_tb", min_value=2, max_value=int(min(layers_sorted[0].count, layers_sorted[-1].count)), step=1)
        with p2:
            supported_sides = number("Barras apoyadas en cada cara lateral", 2, "sup_sd", min_value=2, max_value=len(layers_sorted), step=1)

    confinement_mode = st.radio(
        "Resistencia del núcleo confinado",
        ["Calcular con el método de Mander (Cap. 3)", "Ingresar f'cc y εcc (como en COLUMN)"],
        key=f"{PREFIX}_cmode",
        horizontal=True,
    )
    manual_fcc = manual_ecc = None
    if confinement_mode.startswith("Ingresar"):
        m1, m2 = st.columns(2)
        with m1:
            manual_fcc = number("f'cc (MPa)", float(unit.published_fcc_mpa), "fcc", min_value=5.0, step=1.0, format="%.2f")
        with m2:
            manual_ecc = number("εcc", unit.published_ecc, "ecc", min_value=0.001, step=0.0005, format="%.5f")

    # Objetos de cálculo: se validan aquí para mostrar los errores junto a los datos
    try:
        if arrangement.startswith("Cuadrado"):
            area = equivalent_octagonal_area_mm2(stirrup_diameter)
            transverse = TransverseReinforcement(
                stirrup_diameter, stirrup_spacing, stirrup_fy, area, area, supported_tb, supported_sides
            )
        else:
            transverse = TransverseReinforcement.from_legs(
                stirrup_diameter, stirrup_spacing, stirrup_fy, int(legs_x), int(legs_y),
                supported_bars_top_bottom=supported_tb, supported_bars_sides=supported_sides,
            )
        section = RectangularSection(width_mm, depth_mm, cover_mm, layers, transverse)
        concrete = ConcreteProperties(fco, eps_co, eps_spall, ft, ec)
        steel = ReinforcingSteel(fy, fsu, es, esh, esu, esh_mod)
        computed = confine_rectangular_core(section, fco, eps_co)
        confinement = computed
        if manual_fcc is not None:
            confinement = replace(computed, confined_strength=manual_fcc, confined_strain=manual_ecc, strength_ratio=manual_fcc / fco)
        base_model = build_column_model(section, concrete, steel, confinement)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Núcleo b″ × d″ (mm)", f"{section.core_width_mm:.0f} × {section.core_depth_mm:.0f}")
    d2.metric("ρcc (acero long./núcleo)", f"{section.longitudinal_ratio_core:.4f}")
    d3.metric("ke", f"{confinement.effectiveness:.3f}")
    d4.metric("f'l (x / y) (MPa)", f"{confinement.lateral_stress_x:.2f} / {confinement.lateral_stress_y:.2f}")
    d5.metric("f'cc (MPa)", f"{confinement.confined_strength:.2f}", f"×{confinement.strength_ratio:.3f}", delta_color="off")
    d6.metric("εcc", f"{confinement.confined_strain:.5f}")

    st.markdown("**Corte de la sección** (a escala, con el refuerzo ingresado)")
    st.plotly_chart(
        section_figure(section, octagonal=arrangement.startswith("Cuadrado"), legs_x=int(legs_x), legs_y=int(legs_y)),
        use_container_width=True,
    )
    st.caption(
        "Gris: concreto no confinado (recubrimiento). Azul claro: núcleo confinado, limitado por el eje del estribo. "
        "Cotas en mm; las profundidades de las capas se miden desde la cara comprimida."
    )

# ---------------------------------------------------------
# 4. Parámetros del análisis
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("4. Parámetros del análisis")
    a1, a2 = st.columns(2)
    with a1:
        axial_kn = number("Carga axial, P (kN) — compresión positiva", float(unit.axial_load_n / 1e3), "P", step=50.0, format="%.1f")
        st.caption(f"P / (f'co·Ag) = {axial_kn * 1e3 / (fco * section.gross_area_mm2):.3f}")
    with a2:
        strips = number("Número de láminas de concreto", 60, "strips", min_value=20, max_value=400, step=10,
                        help="COLUMN admite hasta 75. Con 60 el momento máximo varía menos de 0.1 % respecto a 480.")
    schedule = st.radio("Incrementos de curvatura", ["COLUMN (0.2, 0.5 y 1.0 φy)", "Uniforme"], key=f"{PREFIX}_sched", horizontal=True)
    uniform_increment = None
    if schedule == "Uniforme":
        uniform_increment = number("Δφ (µrad/mm)", 2.0, "dphi", min_value=0.01, step=0.5) / PHI_UNIT
    increment_factor = number("Factor de refinamiento de incrementos", 1.0, "ifactor", min_value=0.05, max_value=5.0, step=0.25,
                              help="Multiplica todos los incrementos; valores menores dan más puntos en la curva.")

    st.markdown("**Criterios de terminación** (el análisis se detiene en el primero que ocurra; siempre rige εsu del acero)")
    k1, k2 = st.columns(2)
    with k1:
        use_hoop = st.checkbox("Fractura del estribo por balance de energía (Ec. 7.61–7.66)", value=True, key=f"{PREFIX}_hoop")
        work_core = work_steel = 1.0
        hoop_length = None
        if use_hoop:
            w1, w2 = st.columns(2)
            with w1:
                work_core = number("Wc (Ec. 7.62)", 1.0, "wc", min_value=0.05, max_value=1.0, step=0.05,
                                   help="Fracción del trabajo del núcleo que resiste el estribo crítico (1 = estribo perimetral único).")
            with w2:
                work_steel = number("Ws (Ec. 7.64)", 1.0, "ws", min_value=0.05, max_value=1.0, step=0.05)
            hoop_length = number("Ls, longitud del estribo crítico (mm)", 2.0 * (section.core_width_mm + section.core_depth_mm), "ls",
                                 min_value=100.0, step=50.0, help="Por defecto, el perímetro del estribo perimetral.")
    with k2:
        use_ecu = st.checkbox("Deformación última del núcleo, εcu (dato)", value=False, key=f"{PREFIX}_useecu")
        ecu = number("εcu (en el borde del núcleo)", 0.02, "ecu", min_value=0.004, step=0.002, format="%.4f") if use_ecu else None
        use_duct = st.checkbox("Ductilidad de curvatura máxima, φ/φy", value=False, key=f"{PREFIX}_useduct")
        max_duct = number("φ/φy máxima", 20.0, "duct", min_value=1.0, step=1.0) if use_duct else None
        use_drop = st.checkbox("Caída del momento respecto al máximo", value=False, key=f"{PREFIX}_usedrop")
        drop_ratio = number("M mínimo / Mmax", 0.8, "drop", min_value=0.1, max_value=0.99, step=0.05) if use_drop else None
        use_design = st.checkbox("Curvatura última de diseño de Mander (§7.4.3, Ec. 7.70)", value=False, key=f"{PREFIX}_usedesign",
                                 help="εmax = εspall + 0.04(K − 1) en el núcleo, o M = 0.8 Mi tras superar Mi. Es un criterio conservador.")

    with st.expander("Parámetros numéricos avanzados", expanded=False):
        tolerance = number("Tolerancia de |N − P| (fracción de f'co·Ag)", 1e-5, "tol", min_value=1e-9, max_value=1e-2,
                           step=1e-6, format="%.1e", help="COLUMN usa 5×10⁻⁴ (listado, tolerancia axial).")
        max_iterations = number("Iteraciones máximas de Newton por incremento", 15, "iter", min_value=5, max_value=100, step=1)

inputs = {
    "choice": choice, "b": width_mm, "h": depth_mm, "cover": cover_mm, "fco": fco, "eco": eps_co, "spall": eps_spall,
    "ft": ft, "ec": ec, "steel": (fy, fsu, es, esh, esu, esh_mod), "bars": bars_signature,
    "stirrup": (stirrup_diameter, stirrup_spacing, stirrup_fy, arrangement, legs_x, legs_y, supported_tb, supported_sides),
    "confinement": (manual_fcc, manual_ecc), "P": axial_kn, "strips": strips,
    "increments": (schedule, uniform_increment, increment_factor),
    "termination": (use_hoop, work_core, work_steel, hoop_length, ecu, max_duct, drop_ratio, use_design), "numerics": (tolerance, max_iterations),
}

# ---------------------------------------------------------
# 5. Ejecución del cálculo
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("5. Ejecución del cálculo")
    if st.button("Ejecutar análisis momento–curvatura", type="primary", use_container_width=True):
        try:
            model = build_column_model(section, concrete, steel, confinement, number_of_strips=int(strips))
            settings = AnalysisSettings(
                axial_load_n=axial_kn * 1e3,
                increment_schedule="column" if schedule.startswith("COLUMN") else "uniform",
                uniform_increment=uniform_increment,
                increment_factor=increment_factor,
                core_ultimate_strain=ecu,
                use_hoop_fracture=use_hoop,
                hoop_length_mm=hoop_length,
                work_fraction_core=work_core,
                work_fraction_steel=work_steel,
                max_curvature_ductility=max_duct,
                moment_drop_ratio=drop_ratio,
                use_design_ultimate=use_design,
                equilibrium_tolerance=tolerance,
                max_iterations=int(max_iterations),
            )
            with st.spinner("Calculando…"):
                st.session_state[RESULT_KEY] = {
                    "result": MomentCurvatureAnalysis(model, settings).run(),
                    "inputs": inputs,
                    "example": example_number,
                    "run_id": uuid.uuid4().hex,
                    "panel_curve": None if not from_panel_concrete else (
                        panel_concrete.compression_strain, panel_concrete.compression_stress, f"Panel: {panel_concrete.compression_label}"),
                    "arrangement": {
                        "octagonal": arrangement.startswith("Cuadrado"),
                        "legs_x": int(legs_x),
                        "legs_y": int(legs_y),
                        "description": arrangement,
                    },
                }
        except ValueError as error:
            st.session_state.pop(RESULT_KEY, None)
            st.error(str(error))
        except Exception as error:
            st.session_state.pop(RESULT_KEY, None)
            st.error(f"Ocurrió un error inesperado: {error}")

stored = st.session_state.get(RESULT_KEY)
if stored is None:
    st.info("Revise los datos y ejecute el análisis para ver los resultados.")
    st.stop()

result = stored["result"]
if stored["inputs"] != inputs:
    st.warning("Los datos cambiaron desde el último cálculo: los resultados mostrados corresponden al cálculo anterior. Vuelva a ejecutar.")

model = result.model
records = {record.key: record for record in result.limit_states}
final_state = result.states[-1]
maximum = result.maximum
table = result.to_dataframe()
unconverged = int((~table["converged"]).sum())

# ---------------------------------------------------------
# 6. Resultados numéricos y gráficos
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("6. Resultados numéricos y gráficos")
    if unconverged == 0:
        st.success(f"Los {len(result.states)} incrementos convergieron (máx. {int(table['iterations'].max())} iteraciones de Newton).")
    else:
        st.error(f"{unconverged} incrementos no convergieron.")
    for message in result.warnings:
        st.warning(message)

    ductility_max = result.ductility_of(maximum.phi)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Mmax (kN·m)", f"{maximum.moment / 1e6:,.1f}", f"{maximum.moment / result.ideal_moment:.2f} Mi", delta_color="off")
    q2.metric("φ en Mmax (µrad/mm)", f"{maximum.phi * PHI_UNIT:,.1f}", None if ductility_max is None else f"μφ = {ductility_max:.1f}", delta_color="off")
    q3.metric("Mi, resistencia ideal (kN·m)", f"{result.ideal_moment / 1e6:,.1f}", help="Método de la norma: εc = 0.003, bloque rectangular 0.85 f'c.")
    q4.metric("Eje neutro en Mmax, c (mm)", f"{table.loc[table['moment_kNm'].idxmax(), 'neutral_axis_depth_mm']:,.0f}")

    q5, q6, q7, q8 = st.columns(4)
    if result.yield_curvature_first is not None:
        q5.metric("φ'y, primera fluencia (µrad/mm)", f"{result.yield_curvature_first * PHI_UNIT:,.2f}")
        q6.metric("My, en la primera fluencia (kN·m)", f"{result.yield_moment_first / 1e6:,.1f}")
        q7.metric("φy = φ'y·Mi/My (µrad/mm)", f"{result.yield_curvature * PHI_UNIT:,.2f}", help="Ec. 7.60 de Mander (1984).")
    else:
        q5.metric("φ'y, primera fluencia (µrad/mm)", "no ocurre")
    ultimate_duct = result.ductility_of(final_state.phi)
    q8.metric("φ al terminar (µrad/mm)", f"{final_state.phi * PHI_UNIT:,.1f}",
              None if ultimate_duct is None else f"μφ = {ultimate_duct:.1f}", delta_color="off")
    st.caption(
        "La curvatura última y la ductilidad dependen del estado límite que termina el análisis "
        f"(aquí: **{result.termination_message}**); no son propiedades únicas de la sección."
    )

    st.markdown("**Estados límite y eventos** (cada uno con su criterio)")
    limit_rows = []
    for record in result.limit_states:
        duct = result.ductility_of(record.curvature)
        limit_rows.append({
            "Estado límite": record.label,
            "φ (µrad/mm)": None if not record.reached else round(record.curvature * PHI_UNIT, 3),
            "M (kN·m)": None if not record.reached else round(record.moment / 1e6, 2),
            "μφ": None if duct is None else round(duct, 2),
            "Alcanzado": "sí" if record.reached else "no",
            "Termina el análisis": "sí" if record.terminal else "",
        })
    st.dataframe(pd.DataFrame(limit_rows), hide_index=True, use_container_width=True)
    with st.expander("Criterio que define cada estado límite", expanded=False):
        st.markdown("\n".join(f"- **{record.label}:** {record.definition}" for record in result.limit_states))

    tab_mphi, tab_materials, tab_strains, tab_axis, tab_conv, tab_table = st.tabs(
        ["Momento–curvatura", "Materiales", "Deformaciones", "Eje neutro y perfil", "Convergencia", "Tabla y CSV"]
    )
    with tab_mphi:
        normalized = st.checkbox("Normalizar (M/Mi y φ/φy)", value=False, disabled=result.yield_curvature is None)
        st.plotly_chart(moment_curvature_figure(result, normalized and result.yield_curvature is not None), use_container_width=True)
    with tab_materials:
        st.plotly_chart(material_curves_figure(model, panel_curve=stored.get("panel_curve")), use_container_width=True)
        st.caption("Se muestran las curvas monotónicas usadas en el cálculo. El recubrimiento llega a cero en εspall; el núcleo no tiene límite en la curva.")
    with tab_strains:
        st.plotly_chart(strain_evolution_figure(result), use_container_width=True)
    with tab_axis:
        st.plotly_chart(neutral_axis_figure(result), use_container_width=True)
        step = st.slider("Incremento para el perfil de deformaciones", 0, len(result.states) - 1, len(result.states) - 1)
        st.plotly_chart(strain_profile_figure(result, step), use_container_width=True)
    with tab_conv:
        st.plotly_chart(convergence_figure(result), use_container_width=True)
        st.caption("Rojo: incremento sin convergencia. Newton se detiene en la tolerancia de equilibrio axial; el respaldo por búsqueda en intervalo se usa solo si Newton falla.")
    with tab_table:
        shown = table.rename(columns=TABLE_LABELS)
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.download_button("Descargar resultados (CSV)", shown.to_csv(index=False).encode("utf-8-sig"), "column_momento_curvatura.csv", "text/csv")

    # Comparación con valores publicados, solo si los datos son los del ejemplo sin modificar
    example = stored["example"]
    if example:
        reference = SOESIANAWATI_UNITS[example]
        pristine = stored["inputs"] == inputs and is_unmodified_example(reference, inputs)
        if pristine:
            st.markdown(f"**Comparación con valores publicados — {reference.name}** (Soesianawati, 1986)")
            mi_test = ideal_flexural_strength(model, reference.axial_load_n, concrete_strength=reference.cylinder_strength_test_mpa) / 1e6
            published_mmax = reference.overstrength_mander * reference.ideal_moment_knm
            comparison = pd.DataFrame(
                [
                    ["f'cc (MPa), Tabla 5.1", reference.published_fcc_mpa, model.confinement.confined_strength],
                    ["εcc, Tabla 5.1", reference.published_ecc, model.confinement.confined_strain],
                    ["Mi (kN·m), Tabla 8.3 (con f'c de ensayo)", reference.ideal_moment_knm, mi_test],
                    ["Mmax (kN·m) = (Mmax/Mi) × Mi, Tabla 8.3", published_mmax, maximum.moment / 1e6],
                ],
                columns=["Magnitud", "Publicado", "Calculado"],
            )
            comparison["Diferencia (%)"] = (comparison["Calculado"] / comparison["Publicado"] - 1.0) * 100.0
            st.dataframe(comparison.round(5), hide_index=True, use_container_width=True)
            st.caption("Mmax/Mi está publicado con 2 decimales (±0.5 %). Los valores de Mander de la tabla provienen del análisis monotónico con la fórmula cerrada de confinamiento.")

# ---------------------------------------------------------
# 7. Interpretación de resultados
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("7. Interpretación de resultados")
    st.markdown("\n".join("- " + line for line in interpretation_lines(result)))


# ---------------------------------------------------------
# 8. Diagrama de interacción P–M y ductilidad vs. carga axial
# ---------------------------------------------------------

run_id = stored["run_id"]
settings = result.settings

with st.container(border=True):
    st.subheader("8. Diagrama de interacción P–M y ductilidad vs. carga axial")
    st.markdown(
        "La carga axial cambia la resistencia y, sobre todo, la ductilidad. Aquí se repite el análisis completo para varias cargas "
        "axiales (con la misma sección y los mismos criterios de terminación) y se obtiene el momento máximo y la ductilidad de "
        "cada una. La **curva ideal** (método de la norma, εc = 0.003) da los puntos notables: tracción pura, punto balanceado y compresión pura."
    )
    _, high_default = default_axial_range(model, settings)
    capacity_kn = high_default / 0.8 / 1e3
    i1, i2, i3 = st.columns(3)
    n_points = i1.number_input("Número de cargas axiales", min_value=4, max_value=60, value=12, step=1, key="int_n")
    p_min = i2.number_input("P mínima (kN)", value=0.0, step=100.0, format="%.0f", key=f"int_pmin_{run_id}")
    p_max = i3.number_input("P máxima (kN)", value=float(round(0.8 * capacity_kn)), step=100.0, format="%.0f", key=f"int_pmax_{run_id}")
    st.caption(
        f"Capacidad axial del modelo ≈ {capacity_kn:,.0f} kN (compresión uniforme). Cada carga ejecuta un análisis completo "
        "(≈ 0.5 s); cerca de la capacidad axial el equilibrio puede no converger y esos puntos se marcan."
    )
    if st.button("Calcular diagrama de interacción", use_container_width=True, key="int_run"):
        if p_max <= p_min:
            st.error("La carga máxima debe ser mayor que la mínima.")
        else:
            progress_bar = st.progress(0.0, text="Analizando cargas axiales…")
            loads = np.linspace(p_min, p_max, int(n_points)) * 1e3
            points = analysis_interaction(
                model, settings, loads, progress=lambda i, n: progress_bar.progress(i / n, text=f"Carga axial {i} de {n}")
            )
            progress_bar.empty()
            st.session_state[INTERACTION_KEY] = {"run_id": run_id, "points": points, "curve": code_interaction_curve(model)}

    interaction = st.session_state.get(INTERACTION_KEY)
    if interaction is not None and interaction["run_id"] != run_id:
        interaction = None  # se calculó con otra configuración: ya no corresponde
    if interaction is None:
        st.info("Calcule el diagrama para ver los resultados.")
    else:
        curve, points = interaction["curve"], interaction["points"]
        b1, b2, b3 = st.columns(3)
        b1.metric("Tracción pura, P (kN)", f"{curve.tension_point[0]:,.0f}")
        b2.metric("Punto balanceado: P (kN) / M (kN·m)", f"{curve.balanced_point[0]:,.0f} / {curve.balanced_point[1]:,.0f}")
        b3.metric("Compresión pura, P (kN)", f"{curve.compression_point[0]:,.0f}")
        failed = [p for p in points if p.error]
        if failed:
            st.warning(f"{len(failed)} carga(s) no se pudieron analizar: {failed[0].error}")
        tab_pm, tab_duct, tab_table = st.tabs(["Interacción P–M", "Ductilidad vs. P", "Tabla y CSV"])
        with tab_pm:
            st.plotly_chart(interaction_figure(model, curve, points, settings.axial_load_n / 1e3), use_container_width=True)
            st.caption(
                "Azul: curva ideal. Verde: Mmax del análisis completo, que incluye el endurecimiento del acero y el confinamiento del núcleo, "
                "y por eso supera a la curva ideal. La línea punteada es la carga axial del análisis principal."
            )
        with tab_duct:
            st.plotly_chart(ductility_figure(model, points), use_container_width=True)
            st.caption(
                "La ductilidad disminuye al aumentar P. Los saltos aparecen cuando cambia el criterio que termina el análisis "
                "(deformación del acero ↔ fractura del estribo): compare solo puntos con el mismo criterio. Donde el acero no fluye no se define φy."
            )
        with tab_table:
            interaction_table = interaction_dataframe(points, model).rename(columns=INTERACTION_LABELS)
            st.dataframe(interaction_table, hide_index=True, use_container_width=True)
            st.download_button("Descargar interacción (CSV)", interaction_table.to_csv(index=False).encode("utf-8-sig"),
                               "column_interaccion.csv", "text/csv", key="int_csv")

# ---------------------------------------------------------
# 9. Comparación de configuraciones
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("9. Comparación de configuraciones")
    base_t = model.section.transverse
    st.markdown(
        "Cada fila es una configuración de la misma columna. **Una celda vacía usa el valor de la base** "
        f"(s = {base_t.spacing_mm:g} mm, Ø estribo = {base_t.bar_diameter_mm:g} mm, fyh = {base_t.yield_stress:g} MPa, "
        f"P = {settings.axial_load_n / 1e3:,.0f} kN, f'co = {model.concrete.unconfined_strength:.1f} MPa). "
        "Al cambiar el diámetro del estribo, el área transversal cambia con (Ø/Ø₀)² y las capas de barras se desplazan para conservar su distancia al estribo."
    )
    variants_table = st.data_editor(
        pd.DataFrame({
            "Nombre": ["Base", "s × 0.75", "s × 1.5"],
            "s (mm)": [np.nan, float(round(0.75 * base_t.spacing_mm)), float(round(1.5 * base_t.spacing_mm))],
            "Ø estribo (mm)": [np.nan] * 3,
            "fyh (MPa)": [np.nan] * 3,
            "P (kN)": [np.nan] * 3,
            "f'co (MPa)": [np.nan] * 3,
        }),
        key=f"var_table_{run_id}", num_rows="dynamic", hide_index=True, use_container_width=True,
        column_config={
            "Nombre": st.column_config.TextColumn("Nombre"),
            "s (mm)": st.column_config.NumberColumn("s (mm)", min_value=1.0, step=1.0, format="%.0f"),
            "Ø estribo (mm)": st.column_config.NumberColumn("Ø estribo (mm)", min_value=1.0, step=1.0, format="%.1f"),
            "fyh (MPa)": st.column_config.NumberColumn("fyh (MPa)", min_value=1.0, step=10.0, format="%.0f"),
            "P (kN)": st.column_config.NumberColumn("P (kN)", step=50.0, format="%.0f"),
            "f'co (MPa)": st.column_config.NumberColumn("f'co (MPa)", min_value=1.0, step=1.0, format="%.1f"),
        },
    )
    manual_confinement = stored["inputs"]["confinement"] != (None, None)
    if manual_confinement:
        st.info("Ingresó f'cc y εcc a mano: las variantes de estribos no los modifican (solo cambian P y el resto de los datos).")
    if st.button("Comparar configuraciones", use_container_width=True, key="var_run"):
        def optional(value):
            return None if pd.isna(value) else float(value)

        variants = [
            Variant(
                name=str(row["Nombre"]) if pd.notna(row["Nombre"]) and str(row["Nombre"]).strip() else f"Configuración {i}",
                stirrup_spacing_mm=optional(row["s (mm)"]), stirrup_diameter_mm=optional(row["Ø estribo (mm)"]),
                stirrup_yield_mpa=optional(row["fyh (MPa)"]),
                axial_load_n=None if pd.isna(row["P (kN)"]) else float(row["P (kN)"]) * 1e3,
                unconfined_strength=optional(row["f'co (MPa)"]),
            )
            for i, (_, row) in enumerate(variants_table.iterrows(), start=1)
        ]
        if not variants:
            st.error("Agregue al menos una configuración.")
        else:
            with st.spinner("Analizando configuraciones…"):
                st.session_state[VARIANTS_KEY] = {
                    "run_id": run_id,
                    "outcomes": run_variants(model, settings, variants, recompute_confinement=not manual_confinement),
                }

    variants_state = st.session_state.get(VARIANTS_KEY)
    if variants_state is not None and variants_state["run_id"] != run_id:
        variants_state = None
    if variants_state is None:
        st.info("Ejecute la comparación para ver los resultados.")
    else:
        outcomes = variants_state["outcomes"]
        for outcome in outcomes:
            if outcome.error:
                st.warning(f"«{outcome.variant.name}» no se pudo analizar: {outcome.error}")
        if any(outcome.result is not None for outcome in outcomes):
            normalized_variants = st.checkbox("Normalizar (M/Mi y φ/φy)", value=False, key="var_norm")
            st.plotly_chart(variants_figure(outcomes, normalized_variants), use_container_width=True)
            summary = variants_dataframe(outcomes).rename(columns=VARIANT_LABELS)
            st.dataframe(summary, hide_index=True, use_container_width=True)
            st.download_button("Descargar comparación (CSV)", summary.to_csv(index=False).encode("utf-8-sig"),
                               "column_comparacion.csv", "text/csv", key="var_csv")
            st.caption("Compare la ductilidad entre configuraciones que terminan por el mismo criterio; si cambia el criterio, cambia lo que mide μφ.")
            finished = [o.result.termination_key for o in outcomes if o.result is not None]
            if finished and all(key == "steel_ultimate" for key in finished):
                st.info(
                    "Todas las configuraciones terminan por la deformación última del acero (εsu), que no depende del confinamiento: "
                    "por eso los estribos casi no cambian la ductilidad. Para ver su efecto use una carga axial mayor "
                    "(con la que rige la fractura del estribo) o fije un límite de εcu."
                )

# ---------------------------------------------------------
# 10. De la sección a la columna
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("10. De la sección a la columna: rótula plástica y curva fuerza–desplazamiento")
    st.markdown(
        "El diagrama momento–curvatura es el de **una sección**. Aquí se lleva a una columna en voladizo equivalente (longitud L desde la "
        "sección crítica hasta el punto de inflexión, V = M/L) siguiendo Mander (1984, §7.4.3 y §7.4.4): el desplazamiento es "
        "Δ = Δe + Δs + Δp, con la flexión elástica, el cortante y la rotación plástica en una rótula de longitud equivalente "
        "Lp = Lpy + 0.06 L. Con eso se obtiene la ductilidad de desplazamiento, que es lo que se mide en un ensayo."
    )
    column_section = model.section
    default_length = 1750.0 if stored["example"] else float(round(4.0 * column_section.depth_mm))
    h1, h2 = st.columns(2)
    column_length = h1.number_input(
        "Altura de la columna, L (mm): de la sección crítica al punto de inflexión", min_value=100.0, value=default_length, step=50.0,
        format="%.0f", key=f"col_L_{run_id}",
        help="En los ensayos de Soesianawati (1986) la altura libre de cada mitad es de 1750 mm.")
    tension_diameter = column_section.layers[-1].diameter_mm
    auto_lpy = h2.checkbox(f"Lpy = 32√db = {yield_penetration_length(tension_diameter):.1f} mm (Ec. 7.71, db = {tension_diameter:g} mm)",
                           value=True, key=f"col_autolpy_{run_id}")
    manual_lpy = None if auto_lpy else h2.number_input("Lpy, penetración de fluencia (mm)", min_value=1.0, value=yield_penetration_length(tension_diameter),
                                                       step=5.0, format="%.1f", key=f"col_lpy_{run_id}")
    include_shear = st.checkbox("Incluir la deformación por cortante (Ec. 7.78–7.83)", value=False, key=f"col_shear_{run_id}",
                                help="Importa en columnas de poca esbeltez (L/h bajo). Usa el espaciamiento de estribos de la rótula y fuera de ella.")
    shear_options = {}
    if include_shear:
        with st.expander("Parámetros de cortante", expanded=True):
            w1, w2, w3 = st.columns(3)
            effective_depth = column_section.layers[-1].depth_mm
            shear_options = {
                "hinge_length_mm": w1.number_input("Lh, zona confinada de rótula (mm)", min_value=1.0, value=float(column_section.depth_mm),
                                                   step=10.0, format="%.0f", key=f"col_lh_{run_id}"),
                "outside_hoop_spacing_mm": w2.number_input("Espaciamiento de estribos fuera de la rótula (mm)", min_value=1.0,
                                                           value=2.0 * column_section.transverse.spacing_mm, step=5.0, format="%.0f", key=f"col_sout_{run_id}"),
                "form_factor": w3.number_input("Factor de forma f (1.2 en rectangulares)", min_value=0.5, value=1.2, step=0.1, key=f"col_f_{run_id}"),
                "shear_area_mm2": w1.number_input("Aq, área de alma b·d (mm²)", min_value=1.0, value=column_section.width_mm * effective_depth,
                                                  step=1000.0, format="%.0f", key=f"col_aq_{run_id}"),
                "hoop_shear_area_mm2": w2.number_input("Av, estribos que resisten el cortante (mm²)", min_value=1.0,
                                                       value=column_section.transverse.area_y_mm2, step=10.0, format="%.1f", key=f"col_av_{run_id}"),
            }
            w3.caption(f"d efectivo = {effective_depth:.0f} mm. Rigideces según la Ec. 7.81 con Es = 200 000 MPa, como en el listado de COLUMN.")

    reached = [record for record in result.limit_states if record.reached and record.curvature]
    ultimate_options = {"__end__": f"Final del análisis: {result.termination_message}", **{record.key: record.label for record in reached}}
    ultimate_choice = st.selectbox("Estado que define la curvatura última", list(ultimate_options), format_func=ultimate_options.get,
                                   key=f"col_ult_{run_id}",
                                   help="La ductilidad de desplazamiento depende de este criterio. «Curvatura última de diseño» (Mander §7.4.3) se activa en la sección 4.")
    ultimate_key = None if ultimate_choice == "__end__" else ultimate_choice

    column_response = None
    try:
        column_response = compute_column_response(
            result, ColumnData(length_mm=column_length, yield_penetration_mm=manual_lpy, include_shear=include_shear, **shear_options))
    except ValueError as error:
        st.info(str(error))

    if column_response is not None:
        ultimate = column_response.ultimate(ultimate_key)
        st.session_state[COLUMN_KEY] = {"run_id": run_id, "response": column_response, "ultimate_key": ultimate_key}
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Lp = Lpy + 0.06 L (mm)", f"{column_response.plastic_hinge_length_mm:,.0f}", f"Lp/h = {column_response.plastic_hinge_length_mm / column_section.depth_mm:.2f}", delta_color="off")
        a2.metric("EI efectiva (MN·m²)", f"{column_response.effective_stiffness / 1e12:,.1f}", help="Ec. 7.76: My·L²/(3·Δ de flexión a primera fluencia).")
        a3.metric("Δy, desplazamiento de fluencia (mm)", f"{column_response.yield_displacement:,.2f}",
                  f"V = {result.ideal_moment / column_response.column_length_mm / 1e3:,.0f} kN (Mi/L)", delta_color="off", help="Ec. 7.91: Δ'y·Mi/My.")
        a4.metric("Δ'y, primera fluencia (mm)", f"{column_response.first_yield_displacement:,.2f}")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Δu (mm)", f"{ultimate['displacement']:,.1f}", f"deriva {ultimate['displacement'] / column_response.column_length_mm:.1%}", delta_color="off")
        b2.metric("Ductilidad de desplazamiento μΔ = Δu/Δy", f"{ultimate['ductility']:.2f}")
        b3.metric("θp (Ec. 7.87) (rad)", f"{ultimate['plastic_rotation']:.4f}", help="Rotación plástica con la distribución parabólica de curvatura plástica.")
        b4.metric("θp de diseño = Lp(φu − φy) (rad)", f"{ultimate['design_plastic_rotation']:.4f}", help="Ec. 7.67.")
        st.caption(
            f"Estado último elegido: **{ultimate_options[ultimate_choice]}**, con φu = {ultimate['curvature'] * PHI_UNIT:.1f} µrad/mm "
            f"(μφ = {result.ductility_of(ultimate['curvature']):.1f}) y V = {ultimate['shear_force'] / 1e3:,.0f} kN."
        )
        if ultimate_choice == "__end__" and result.termination_key == "steel_ultimate":
            st.warning(
                "El estado último lo fija la deformación última del acero (εsu), que da desplazamientos muy grandes. En columnas reales con "
                "carga axial baja suele regir antes el pandeo de las barras o la fractura del estribo (en el ensayo de la Unidad 1 el desplazamiento "
                "máximo medido fue 91 mm). Compare con la «Curvatura última de diseño» de Mander (sección 4), que es conservadora."
            )
        tab_fd, tab_comp, tab_col_table = st.tabs(["Fuerza–desplazamiento", "Componentes del desplazamiento", "Tabla y CSV"])
        with tab_fd:
            st.plotly_chart(force_displacement_figure(column_response, ultimate_key), use_container_width=True)
            st.caption("La línea punteada es la idealización elastoplástica equivalente: llega a Mi/L en Δy.")
        with tab_comp:
            st.plotly_chart(displacement_components_figure(column_response), use_container_width=True)
        with tab_col_table:
            column_table = column_response.to_dataframe()
            column_table["limit_state"] = table["limit_state"]
            st.dataframe(column_table.rename(columns=COLUMN_LABELS), hide_index=True, use_container_width=True)
            st.download_button("Descargar respuesta de la columna (CSV)", column_table.rename(columns=COLUMN_LABELS).to_csv(index=False).encode("utf-8-sig"),
                               "column_fuerza_desplazamiento.csv", "text/csv", key="col_csv")
        st.caption(
            "Hipótesis: columna en voladizo con carga axial constante; la deformación por cortante es opcional; respuesta monotónica sin ciclos. "
            "La curvatura plástica se distribuye en una parábola más la penetración de la fluencia (Mander, §7.4.4)."
        )

# ---------------------------------------------------------
# 11. Informe (Word y PDF)
# ---------------------------------------------------------

with st.container(border=True):
    st.subheader("11. Informe (Word y PDF)")
    st.markdown(
        "Genera un informe con los datos, el corte de la sección, los resultados, los estados límite, la interpretación y, si los "
        "calculó, el diagrama de interacción y la comparación de configuraciones."
    )
    r1, r2 = st.columns(2)
    report_title = r1.text_input("Título del informe", value="Análisis momento–curvatura de una sección de concreto armado")
    report_author = r2.text_input("Autor", value="")
    column_state = st.session_state.get(COLUMN_KEY)
    if column_state is not None and column_state["run_id"] != run_id:
        column_state = None
    k0, k1, k2 = st.columns(3)
    include_column = k0.checkbox("Incluir la respuesta de la columna (sección 10)", value=column_state is not None, disabled=column_state is None)
    include_interaction = k1.checkbox("Incluir interacción P–M y ductilidad vs. P", value=interaction is not None, disabled=interaction is None)
    include_variants = k2.checkbox("Incluir comparación de configuraciones", value=variants_state is not None, disabled=variants_state is None)
    if st.button("Generar informe", type="primary", use_container_width=True, key="report_run"):
        try:
            with st.spinner("Generando el informe…"):
                document = build_column_report(
                    result,
                    arrangement=stored["arrangement"],
                    interaction=None if not (include_interaction and interaction) else {
                        "curve": interaction["curve"], "points": interaction["points"], "current_axial_kn": settings.axial_load_n / 1e3},
                    variants=variants_state["outcomes"] if include_variants and variants_state else None,
                    title=report_title,
                    author=report_author,
                    column=None if not (include_column and column_state) else {
                        "response": column_state["response"], "ultimate_key": column_state["ultimate_key"]},
                )
                st.session_state[REPORT_KEY] = {"run_id": run_id, "docx": write_docx(document), "pdf": write_pdf(document)}
        except RuntimeError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"No se pudo generar el informe: {error}")
    report = st.session_state.get(REPORT_KEY)
    if report is not None and report["run_id"] == run_id:
        d1, d2 = st.columns(2)
        d1.download_button("Descargar Word (.docx)", report["docx"], "informe_column.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True, key="report_docx")
        d2.download_button("Descargar PDF", report["pdf"], "informe_column.pdf", "application/pdf", use_container_width=True, key="report_pdf")
        st.caption("El informe corresponde al último cálculo ejecutado y a las secciones 8, 9 y 10 tal como estaban al generarlo.")
