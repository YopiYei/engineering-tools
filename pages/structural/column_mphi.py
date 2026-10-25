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
from calculations.structural.column_examples import (
    HD16_STEEL,
    SECTION_SIZE_MM,
    SOESIANAWATI_UNITS,
    unit_model,
    unit_section,
)
from calculations.structural.column_materials import ReinforcingSteel
from calculations.structural.column_section import (
    BarLayer,
    RectangularSection,
    TransverseReinforcement,
    equivalent_octagonal_area_mm2,
)
from styles.main import apply_global_styles
from visualization.column_plots import (
    PHI_UNIT,
    convergence_figure,
    material_curves_figure,
    moment_curvature_figure,
    neutral_axis_figure,
    section_figure,
    strain_evolution_figure,
    strain_profile_figure,
)


apply_global_styles()

RESULT_KEY = "column_mphi_result"
OWN_DATA = "Datos propios (parten de los de la Unidad 1)"
EXAMPLES = {f"Ejemplo — Soesianawati (1986), {unit.name}": number for number, unit in SOESIANAWATI_UNITS.items()}
EXAMPLES[OWN_DATA] = None

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

    g1, g2, g3 = st.columns(3)
    width_mm = number("Ancho, b (mm)", SECTION_SIZE_MM, "b", min_value=100.0, step=10.0)
    depth_mm = number("Peralte, h (mm) — flexión respecto al eje x", SECTION_SIZE_MM, "h", min_value=100.0, step=10.0)
    cover_mm = number("Recubrimiento libre hasta el estribo (mm)", 13.0, "cover", min_value=0.0, step=1.0)

    st.markdown("**Concreto**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fco = number("f'co, resistencia in situ (MPa)", float(unit.unconfined_strength_mpa), "fco", min_value=5.0, step=1.0, format="%.2f")
    with c2:
        eps_co = number("εco (deformación en f'co)", float(unit.strain_at_peak), "eco", min_value=0.001, step=0.0001, format="%.5f")
    with c3:
        eps_spall = number("εspall (descascaramiento)", 0.008, "spall", min_value=0.004, step=0.001, format="%.4f",
                           help="Dato del usuario en COLUMN. La tesis de Mander usa 0.008 en sus estudios paramétricos (§7.4.3).")
    with c4:
        auto_ft = st.checkbox("f't = 0.6√f'co (supuesto)", value=True, key=f"{PREFIX}_autoft",
                              help="No es un dato de COLUMN: es un valor típico. Desmarque para ingresar f't (0 desprecia la tracción).")
        ft = None if auto_ft else number("f't (MPa)", 0.0, "ft", min_value=0.0, step=0.1)
    auto_ec = st.checkbox("Ec = 5000√f'co (Mander §3.6.2)", value=True, key=f"{PREFIX}_autoec")
    ec = None if auto_ec else number("Ec (MPa)", 30000.0, "ec", min_value=1000.0, step=500.0)

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
    default_layers = unit_section(unit).layers
    layers_table = st.data_editor(
        pd.DataFrame(
            {
                "Profundidad (mm)": [layer.depth_mm for layer in default_layers],
                "N° de barras": [layer.count for layer in default_layers],
                "Diámetro (mm)": [layer.diameter_mm for layer in default_layers],
            }
        ),
        key=f"{PREFIX}_layers",
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
    "termination": (use_hoop, work_core, work_steel, hoop_length, ecu, max_duct, drop_ratio), "numerics": (tolerance, max_iterations),
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
                equilibrium_tolerance=tolerance,
                max_iterations=int(max_iterations),
            )
            with st.spinner("Calculando…"):
                st.session_state[RESULT_KEY] = {
                    "result": MomentCurvatureAnalysis(model, settings).run(),
                    "inputs": inputs,
                    "example": example_number,
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
        st.plotly_chart(material_curves_figure(model), use_container_width=True)
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
    lines = []
    ratio = maximum.moment / result.ideal_moment
    if ratio >= 1.0:
        lines.append(
            f"- La sección alcanza **Mmax = {maximum.moment / 1e6:,.1f} kN·m**, que es {ratio:.2f} veces la resistencia ideal Mi: el exceso "
            "proviene sobre todo del endurecimiento del acero y del confinamiento del núcleo, que el método de la norma no considera."
        )
    else:
        lines.append(
            f"- La sección alcanza **Mmax = {maximum.moment / 1e6:,.1f} kN·m**, que es solo {ratio:.2f} veces la resistencia ideal Mi: "
            "revise los datos, porque normalmente Mmax supera a Mi."
        )
    if records["first_yield"].reached and records["first_yield"].curvature <= maximum.phi:
        lines.append(
            f"- El acero traccionado fluye (φ'y = {result.yield_curvature_first * PHI_UNIT:.2f} µrad/mm) antes del momento máximo, "
            "de modo que la curva tiene un tramo de fluencia. Con φy = φ'y·Mi/My (Ec. 7.60) la ductilidad se mide como μφ = φ/φy."
        )
    elif records["first_yield"].reached:
        lines.append(
            "- El momento máximo ocurre antes de la primera fluencia del acero traccionado: la respuesta está controlada por la compresión del concreto."
        )
    else:
        lines.append("- El acero traccionado **no fluye** antes del estado límite que termina el análisis: la sección está controlada por compresión y no se define ductilidad de curvatura.")
    spalling = records.get("cover_spalling")
    if spalling is not None and spalling.reached:
        drop = 1.0 - spalling.moment / maximum.moment
        lines.append(
            f"- El recubrimiento se descascara a φ = {spalling.curvature * PHI_UNIT:.1f} µrad/mm (εspall = {eps_spall:g}); en ese punto el momento está "
            f"{drop:.0%} por debajo del máximo. Después la capacidad depende del núcleo confinado."
        )
    lines.append(
        f"- El análisis terminó por **{result.termination_message.lower()}** a φ = {final_state.phi * PHI_UNIT:.1f} µrad/mm"
        + (f" (μφ = {ultimate_duct:.1f})" if ultimate_duct is not None else "")
        + f", con M = {final_state.moment / 1e6:,.1f} kN·m ({final_state.moment / maximum.moment:.0%} de Mmax). "
        "Si cambia el criterio de terminación, la curvatura última cambia."
    )
    if result.termination_key == "hoop_fracture":
        lines.append("- La fractura del estribo se estima con un balance de energía (Mander, §3.8.2) cuyas constantes (105 MJ/m³ para el acero y 0.017√f'c para el concreto) son empíricas.")
    if result.termination_key == "steel_ultimate":
        lines.append("- Rige la deformación última del acero εsu; en una columna real el confinamiento o el pandeo de las barras suelen limitar antes la curvatura útil.")
    lines.append(
        "- **Limitaciones:** respuesta monotónica sobre la curva esqueleto (sin descarga de fibras), cortante y deslizamiento ignorados, "
        "reconstrucción de COLUMN y no su traducción literal."
    )
    st.markdown("\n".join(lines))
