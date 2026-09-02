import streamlit as st

from visualization.sections import (
    draw_rectangle,
    draw_circle,
    draw_i_section,
)

from calculations.structural.section_properties import (
    rectangle_properties,
    circle_properties,
    i_section_properties,
)

from styles.main import apply_global_styles


apply_global_styles()


# ---------------------------------------------------------
# Verbosity levels
# ---------------------------------------------------------

VERBOSITY_RESULTS = 0
VERBOSITY_SUMMARY = 1
VERBOSITY_FULL = 2


st.title("Section Properties")


# ---------------------------------------------------------
# 1. Section type
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("1. Section Type")

    section_type = st.selectbox(
        "Select section type",
        options=[
            "Rectangle",
            "Circle",
            "I / W Section",
        ],
    )


# ---------------------------------------------------------
# 2. Geometry
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("2. Geometry")

    if section_type == "Rectangle":

        col1, col2 = st.columns(2)

        with col1:
            width = st.number_input(
                "Width, b",
                min_value=0.0,
                value=10.0,
            )

        with col2:
            height = st.number_input(
                "Height, h",
                min_value=0.0,
                value=20.0,
            )


    elif section_type == "Circle":

        radius = st.number_input(
            "Radius, r",
            min_value=0.0,
            value=5.0,
        )


    elif section_type == "I / W Section":

        col1, col2 = st.columns(2)

        with col1:

            depth = st.number_input(
                "Depth, d",
                min_value=0.0,
                value=12.0,
            )

            flange_thickness = st.number_input(
                "Flange thickness, tf",
                min_value=0.0,
                value=0.75,
            )

        with col2:

            flange_width = st.number_input(
                "Flange width, bf",
                min_value=0.0,
                value=8.0,
            )

            web_thickness = st.number_input(
                "Web thickness, tw",
                min_value=0.0,
                value=0.5,
            )


# ---------------------------------------------------------
# 3. Output detail
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("3. Output Detail")

    verbosity = st.selectbox(
        "Verbosity",
        options=[
            VERBOSITY_RESULTS,
            VERBOSITY_SUMMARY,
            VERBOSITY_FULL,
        ],
        index=0,
        format_func=lambda value: {
            VERBOSITY_RESULTS: "0 - Results only",
            VERBOSITY_SUMMARY: "1 - Summary",
            VERBOSITY_FULL: "2 - Full calculation",
        }[value],
    )


# ---------------------------------------------------------
# 4. Calculate
# ---------------------------------------------------------

if st.button(
    "Calculate",
    type="primary",
    use_container_width=True,
):

    try:

        # -----------------------------------------------------
        # Rectangle
        # -----------------------------------------------------

        if section_type == "Rectangle":

            if width <= 0 or height <= 0:
                raise ValueError(
                    "Width and height must be greater than zero."
                )

            result = rectangle_properties(
                width=width,
                height=height,
            )
            figure = draw_rectangle(
                width=width,
                height=height,
                centroid_x=result["centroid_x"],
                centroid_y=result["centroid_y"],
            )


        # -----------------------------------------------------
        # Circle
        # -----------------------------------------------------

        elif section_type == "Circle":

            if radius <= 0:
                raise ValueError(
                    "Radius must be greater than zero."
                )

            result = circle_properties(
                radius=radius,
            )
            figure = draw_circle(
                radius=radius,
                centroid_x=result["centroid_x"],
                centroid_y=result["centroid_y"],
            )


        # -----------------------------------------------------
        # I / W Section
        # -----------------------------------------------------

        elif section_type == "I / W Section":

            if (
                depth <= 0
                or flange_width <= 0
                or flange_thickness <= 0
                or web_thickness <= 0
            ):
                raise ValueError(
                    "All dimensions must be greater than zero."
                )

            result = i_section_properties(
                depth=depth,
                flange_width=flange_width,
                flange_thickness=flange_thickness,
                web_thickness=web_thickness,
            )

            figure = draw_i_section(
                depth=depth,
                flange_width=flange_width,
                flange_thickness=flange_thickness,
                web_thickness=web_thickness,
                centroid_x=result["centroid_x"],
                centroid_y=result["centroid_y"],
            )


        # =====================================================
        # VERBOSITY 2
        # =====================================================

        if verbosity >= VERBOSITY_FULL:

            st.subheader("Calculation Details")

            if section_type == "Rectangle":

                st.latex(
                    r"""
                    A = bh
                    """
                )

                st.latex(
                    rf"""
                    A =
                    ({width:.4f})
                    ({height:.4f})
                    =
                    {result["area"]:.4f}
                    """
                )

                st.latex(
                    r"""
                    I_x =
                    \frac{bh^3}{12}
                    """
                )

                st.latex(
                    rf"""
                    I_x =
                    \frac{{
                    ({width:.4f})
                    ({height:.4f})^3
                    }}{{
                    12
                    }}
                    =
                    {result["Ix"]:.4f}
                    """
                )

                st.latex(
                    r"""
                    I_y =
                    \frac{hb^3}{12}
                    """
                )

                st.latex(
                    rf"""
                    I_y =
                    \frac{{
                    ({height:.4f})
                    ({width:.4f})^3
                    }}{{
                    12
                    }}
                    =
                    {result["Iy"]:.4f}
                    """
                )


            elif section_type == "Circle":

                st.latex(
                    r"""
                    A = \pi r^2
                    """
                )

                st.latex(
                    rf"""
                    A =
                    \pi
                    ({radius:.4f})^2
                    =
                    {result["area"]:.4f}
                    """
                )

                st.latex(
                    r"""
                    I_x = I_y =
                    \frac{\pi r^4}{4}
                    """
                )

                st.latex(
                    rf"""
                    I_x = I_y =
                    \frac{{
                    \pi
                    ({radius:.4f})^4
                    }}{{
                    4
                    }}
                    =
                    {result["Ix"]:.4f}
                    """
                )


            elif section_type == "I / W Section":

                st.markdown(
                    "### Areas"
                )

                st.latex(
                    rf"""
                    A_f =
                    b_f t_f
                    =
                    ({flange_width:.4f})
                    ({flange_thickness:.4f})
                    =
                    {result["flange_area"]:.4f}
                    """
                )

                st.latex(
                    rf"""
                    h_w =
                    d - 2t_f
                    =
                    {result["web_height"]:.4f}
                    """
                )

                st.latex(
                    rf"""
                    A_w =
                    t_w h_w
                    =
                    ({web_thickness:.4f})
                    ({result["web_height"]:.4f})
                    =
                    {result["web_area"]:.4f}
                    """
                )

                st.markdown(
                    "### Centroid"
                )

                st.latex(
                    rf"""
                    \bar{{x}}
                    =
                    \frac{{b_f}}{{2}}
                    =
                    {result["centroid_x"]:.4f}
                    """
                )

                st.latex(
                    rf"""
                    \bar{{y}}
                    =
                    \frac{{d}}{{2}}
                    =
                    {result["centroid_y"]:.4f}
                    """
                )

                st.markdown(
                    "### Moment of Inertia"
                )

                st.latex(
                    r"""
                    I_x =
                    \sum
                    \left(
                    I_{x,i}
                    +
                    A_i d_i^2
                    \right)
                    """
                )

                st.latex(
                    rf"""
                    I_x =
                    {result["Ix"]:.4f}
                    """
                )

                st.latex(
                    rf"""
                    I_y =
                    {result["Iy"]:.4f}
                    """
                )


        # =====================================================
        # VERBOSITY 1
        # =====================================================

        if verbosity >= VERBOSITY_SUMMARY:

            st.subheader("Section Summary")

            st.write(
                f"Section type: {section_type}"
            )


        # =====================================================
        # VERBOSITY 0
        # Results
        # =====================================================

        st.subheader("Section")

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
            },
        )

        st.subheader("Results")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Area",
            f"{result['area']:.4f}",
        )

        col2.metric(
            "Centroid x̄",
            f"{result['centroid_x']:.4f}",
        )

        col3.metric(
            "Centroid ȳ",
            f"{result['centroid_y']:.4f}",
        )

        col4.metric(
            "Ix",
            f"{result['Ix']:.4f}",
        )

        col5.metric(
            "Iy",
            f"{result['Iy']:.4f}",
        )


    except ValueError as error:

        st.error(
            str(error)
        )


    except Exception as error:

        st.error(
            f"An unexpected error occurred: {error}"
        )