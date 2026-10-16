import pandas as pd
import streamlit as st

from calculations.structural.section_classification import (
    COMPACT,
    NONCOMPACT,
    NONSLENDER,
    aisc_shape_compression,
    aisc_shape_elements,
    angle_compression,
    angle_elements,
    channel_compression,
    channel_elements,
    hss_rect_compression,
    hss_round_compression,
    i_shape_compression,
    tee_compression,
    classify_section,
    hss_rect_elements,
    hss_round_elements,
    i_shape_elements,
    tee_elements,
)

from visualization.sections import (
    draw_aisc_shape,
    draw_i_section,
)

from utils.aisc_data import (
    load_manifest,
    load_family,
)

from styles.main import apply_global_styles


apply_global_styles()


# ---------------------------------------------------------
# Verbosity levels
# ---------------------------------------------------------

VERBOSITY_RESULTS = 0
VERBOSITY_SUMMARY = 1
VERBOSITY_FULL = 2


STEEL_GRADES = {
    "A36 (Fy = 36 ksi)": 36.0,
    "A992 / A572 Gr.50 (Fy = 50 ksi)": 50.0,
    "A500 Gr.B (Fy = 46 ksi)": 46.0,
    "A500 Gr.C (Fy = 50 ksi)": 50.0,
    "Custom": None,
}

# Families / custom types for which a bending axis is meaningful.
AXIS_FAMILIES = {"w_m_s_hp", "c_mc", "hss_rect"}

CUSTOM_TYPES = [
    "I / W - welded built-up",
    "I / W - rolled",
    "Channel",
    "Tee",
    "Single angle",
    "Rectangular HSS",
    "Round HSS / pipe",
]

AXIS_CUSTOM_TYPES = {
    "I / W - welded built-up",
    "I / W - rolled",
    "Channel",
    "Rectangular HSS",
}


st.title("Section Classification")

st.caption(
    "AISC 360-16 Table B4.1 - local buckling of the elements of a section. "
    "Flexure (B4.1b): compact, noncompact or slender. "
    "Compression (B4.1a): nonslender or slender."
)


# ---------------------------------------------------------
# 1. Input mode
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("1. Input Mode")

    input_mode = st.radio(
        "Select input mode",
        options=["AISC Database", "Custom Dimensions"],
        horizontal=True,
    )

    load_type = st.radio(
        "Load type",
        options=["Flexure", "Compression"],
        horizontal=True,
    )

    compression = load_type == "Compression"
    table_name = "B4.1a" if compression else "B4.1b"


# ---------------------------------------------------------
# 2. Material
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("2. Material")

    col1, col2, col3 = st.columns(3)

    with col1:
        grade = st.selectbox("Steel grade", options=list(STEEL_GRADES.keys()), index=1)

    with col2:
        Fy = st.number_input(
            "Yield stress, Fy (ksi)",
            min_value=0.0,
            value=STEEL_GRADES[grade] or 50.0,
            key=f"fy_{grade}",
            help="The grade sets the starting value; you can overwrite it.",
        )

    with col3:
        E = st.number_input(
            "Elastic modulus, E (ksi)",
            min_value=0.0,
            value=29000.0,
        )


# ---------------------------------------------------------
# 3. Section
# ---------------------------------------------------------

family_key = None
shape_name = None
props = None
custom_type = None
inputs = {}
axis = "strong"

with st.container(border=True):

    st.subheader("3. Section")

    if input_mode == "AISC Database":

        manifest = load_manifest()
        family_labels = {item["label"]: item["key"] for item in manifest}

        col1, col2 = st.columns(2)

        with col1:
            family_label = st.selectbox(
                "Shape family", options=list(family_labels.keys())
            )

        family_key = family_labels[family_label]
        shapes = load_family(family_key)

        with col2:
            shape_name = st.selectbox("Shape", options=sorted(shapes.keys()))

        props = shapes[shape_name]
        has_axis = family_key in AXIS_FAMILIES and not compression

    else:

        custom_type = st.selectbox("Section type", options=CUSTOM_TYPES)
        has_axis = custom_type in AXIS_CUSTOM_TYPES and not compression

        def number(label, value, key):
            return st.number_input(label, min_value=0.0, value=value, key=key, format="%.4f")

        col1, col2 = st.columns(2)

        if custom_type.startswith("I / W"):
            with col1:
                inputs["bf"] = number("Flange width, bf", 12.0, "i_bf")
                inputs["tf"] = number("Flange thickness, tf", 0.5, "i_tf")
            with col2:
                inputs["h"] = number("Web clear height, h", 28.0, "i_h")
                inputs["tw"] = number("Web thickness, tw", 0.375, "i_tw")
            st.caption(
                "h = d - 2tf for welded sections, h = d - 2k for rolled sections."
            )

        elif custom_type == "Channel":
            with col1:
                inputs["bf"] = number("Flange width, bf", 3.0, "c_bf")
                inputs["tf"] = number("Flange thickness, tf", 0.5, "c_tf")
            with col2:
                inputs["h"] = number("Web clear height, h", 8.0, "c_h")
                inputs["tw"] = number("Web thickness, tw", 0.3, "c_tw")

        elif custom_type == "Tee":
            with col1:
                inputs["bf"] = number("Flange width, bf", 8.0, "t_bf")
                inputs["tf"] = number("Flange thickness, tf", 0.5, "t_tf")
            with col2:
                inputs["d"] = number("Total depth, d", 8.0, "t_d")
                inputs["tw"] = number("Stem thickness, tw", 0.3, "t_tw")

        elif custom_type == "Single angle":
            with col1:
                inputs["b"] = number("Leg b", 4.0, "a_b")
                inputs["d"] = number("Leg d", 4.0, "a_d")
            with col2:
                inputs["t"] = number("Thickness, t", 0.375, "a_t")

        elif custom_type == "Rectangular HSS":
            with col1:
                inputs["b"] = number("Outside width, B (parallel to axis)", 6.0, "r_b")
                inputs["h"] = number("Outside depth, H", 10.0, "r_h")
            with col2:
                inputs["t"] = number("Design wall thickness, t", 0.349, "r_t")
            st.caption("Flat widths are taken as (outside dimension - 3t).")

        elif custom_type == "Round HSS / pipe":
            with col1:
                inputs["D"] = number("Outside diameter, D", 10.0, "o_D")
            with col2:
                inputs["t"] = number("Design wall thickness, t", 0.349, "o_t")

    if has_axis:
        axis_label = st.radio(
            "Bending axis",
            options=["Strong axis (x)", "Weak axis (y)"],
            horizontal=True,
        )
        axis = "strong" if axis_label.startswith("Strong") else "weak"


# ---------------------------------------------------------
# 4. Output detail
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("4. Output Detail")

    verbosity = st.selectbox(
        "Verbosity",
        options=[VERBOSITY_RESULTS, VERBOSITY_SUMMARY, VERBOSITY_FULL],
        index=0,
        format_func=lambda value: {
            VERBOSITY_RESULTS: "0 - Results only",
            VERBOSITY_SUMMARY: "1 - Summary",
            VERBOSITY_FULL: "2 - Full calculation",
        }[value],
    )


# ---------------------------------------------------------
# 5. Classify
# ---------------------------------------------------------

if st.button("Classify", type="primary", use_container_width=True):

    try:

        if Fy <= 0 or E <= 0:
            raise ValueError("Fy and E must be greater than zero.")

        figure = None

        # -------------------------------------------------
        # Elements
        # -------------------------------------------------

        if input_mode == "AISC Database":

            if compression:
                elements = aisc_shape_compression(family_key, props, E, Fy)
            else:
                elements = aisc_shape_elements(family_key, props, E, Fy, axis=axis)

            gap = 0.375
            figure = draw_aisc_shape(
                family_key, props, gap=gap, shape_name=shape_name
            )

            title = shape_name

        else:

            title = custom_type

            if custom_type.startswith("I / W"):

                if any(value <= 0 for value in inputs.values()):
                    raise ValueError("All dimensions must be greater than zero.")

                built_up = custom_type.endswith("built-up")

                if compression:
                    elements = i_shape_compression(
                        inputs["bf"], inputs["tf"], inputs["h"], inputs["tw"],
                        E, Fy, built_up=built_up,
                    )
                else:
                    elements = i_shape_elements(
                        inputs["bf"], inputs["tf"], inputs["h"], inputs["tw"],
                        E, Fy, built_up=built_up, axis=axis,
                    )

                depth = inputs["h"] + 2 * inputs["tf"]
                figure = draw_i_section(
                    depth=depth,
                    flange_width=inputs["bf"],
                    flange_thickness=inputs["tf"],
                    web_thickness=inputs["tw"],
                    centroid_x=inputs["bf"] / 2,
                    centroid_y=depth / 2,
                )

            elif custom_type == "Channel":
                if compression:
                    elements = channel_compression(
                        inputs["bf"], inputs["tf"], inputs["h"], inputs["tw"], E, Fy
                    )
                else:
                    elements = channel_elements(
                        inputs["bf"], inputs["tf"], inputs["h"], inputs["tw"], E, Fy, axis=axis
                    )

            elif custom_type == "Tee":
                tee = tee_compression if compression else tee_elements
                elements = tee(
                    inputs["bf"], inputs["tf"], inputs["d"], inputs["tw"], E, Fy
                )

            elif custom_type == "Single angle":
                angle = angle_compression if compression else angle_elements
                elements = angle(inputs["b"], inputs["d"], inputs["t"], E, Fy)

            elif custom_type == "Rectangular HSS":
                if compression:
                    elements = hss_rect_compression(
                        inputs["b"], inputs["h"], inputs["t"], E, Fy
                    )
                else:
                    elements = hss_rect_elements(
                        inputs["b"], inputs["h"], inputs["t"], E, Fy, axis=axis
                    )

            else:
                round_hss = hss_round_compression if compression else hss_round_elements
                elements = round_hss(inputs["D"], inputs["t"], E, Fy)

        result = classify_section(elements)


        # =================================================
        # VERBOSITY 2
        # =================================================

        if verbosity >= VERBOSITY_FULL:

            st.subheader("Calculation Details")

            st.latex(
                rf"""
                E = {E:g}\ \text{{ksi}},\quad
                F_y = {Fy:g}\ \text{{ksi}}
                """
            )

            for element in elements:

                st.markdown(
                    f"**{element['element']}** - Table {table_name}, Case {element['case']}"
                )

                st.latex(element["lam_tex"])

                if compression:

                    relation = r"\le" if element["state"] == NONSLENDER else ">"

                    st.latex(rf"\lambda_r = {element['lam_r_tex']}")

                    st.latex(
                        rf"""
                        \lambda = {element['lam']:.2f}
                        \;{relation}\;
                        \lambda_r = {element['lam_r']:.2f}
                        \;\Rightarrow\;
                        \text{{{element['state']}}}
                        """
                    )

                else:

                    st.latex(rf"\lambda_p = {element['lam_p_tex']}")
                    st.latex(rf"\lambda_r = {element['lam_r_tex']}")

                    st.latex(
                        rf"""
                        \lambda = {element['lam']:.2f}
                        \;\;\text{{vs}}\;\;
                        \lambda_p = {element['lam_p']:.2f},\;
                        \lambda_r = {element['lam_r']:.2f}
                        \;\Rightarrow\;
                        \text{{{element['state']}}}
                        """
                    )


        # =================================================
        # VERBOSITY 1
        # =================================================

        if verbosity >= VERBOSITY_SUMMARY:

            st.subheader("Slenderness Summary")

            rows = []

            for e in elements:

                row = {
                    "Element": e["element"],
                    f"Case ({table_name})": e["case"],
                    "λ": round(e["lam"], 2),
                }

                if not compression:
                    row["λp"] = round(e["lam_p"], 2)

                row["λr"] = round(e["lam_r"], 2)

                if compression:
                    row["λ / λr"] = round(e["lam"] / e["lam_r"], 2)
                else:
                    row["λ / λp"] = round(e["lam"] / e["lam_p"], 2)

                row["Class"] = e["state"]
                rows.append(row)

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


        # =================================================
        # VERBOSITY 0
        # =================================================

        if figure is not None:

            st.subheader("Section")

            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"displayModeBar": True, "scrollZoom": True},
            )

        st.subheader("Result")

        message = (
            f"{title}: **{result['classification'].upper()}** for "
            f"{'compression' if compression else 'flexure'}"
            f" (governing element: {result['governing_element']}, Fy = {Fy:g} ksi)."
        )

        if result["classification"] in (COMPACT, NONSLENDER):
            st.success(message)
        elif result["classification"] == NONCOMPACT:
            st.warning(message)
        else:
            st.error(message)

        if not compression and result["classification"] == NONCOMPACT:
            st.caption(
                "Flexural strength: lateral-torsional buckling and local buckling "
                "limit states of AISC 360-16 Chapter F apply (e.g. F3 for I-shapes "
                "with noncompact flanges)."
            )

        if compression and result["classification"] != NONSLENDER:
            st.caption(
                "Slender elements reduce the compressive strength: use the "
                "reduction factors Qs, Qa of AISC 360-16 Section E7."
            )

        if result["classification"] not in (COMPACT, NONSLENDER) and verbosity == VERBOSITY_RESULTS:
            governing = next(
                e for e in elements if e["element"] == result["governing_element"]
            )
            limit_name, limit = (
                ("λr", governing["lam_r"]) if compression else ("λp", governing["lam_p"])
            )
            st.caption(
                f"{governing['element']}: λ = {governing['lam']:.2f} > "
                f"{limit_name} = {limit:.2f}."
            )


    except ValueError as error:

        st.error(str(error))


    except Exception as error:

        st.error(f"An unexpected error occurred: {error}")
