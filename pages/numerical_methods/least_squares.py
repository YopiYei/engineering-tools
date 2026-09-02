from styles.main import apply_global_styles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from calculations.numerical_methods.least_squares import (
    linear_least_squares,
)

from utils.data_loader import load_tabular_data

apply_global_styles()

# Load Data from file
uploaded_file = st.file_uploader(
    "Import data",
    type=["csv", "xlsx", "xls", "json"],
)

# ---------------------------------------------------------
# Verbosity levels
# ---------------------------------------------------------

VERBOSITY_RESULTS = 0
VERBOSITY_SUMMARY = 1
VERBOSITY_FULL = 2


st.title("Least Squares")


# ---------------------------------------------------------
# 1. Input data
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("1. Input Data")

    if uploaded_file is not None:

        try:
            imported_data = load_tabular_data(uploaded_file)

            if "x" not in imported_data.columns or "y" not in imported_data.columns:
                st.error(
                    "The imported file must contain columns named 'x' and 'y'."
                )

                data_source = pd.DataFrame(
                    {
                        "x": [],
                        "y": [],
                    }
                )

            else:
                data_source = imported_data[["x", "y"]]

        except ValueError as error:
            st.error(str(error))

            data_source = pd.DataFrame(
                {
                    "x": [],
                    "y": [],
                }
            )

    else:

        data_source = pd.DataFrame(
            {
                "x": [63, 52, 78, 49, 71, 62, 68, 48, 56],
                "y": [162, 158, 167, 151, 162, 168, 167, 153, 152],
            }
        )

    data = st.data_editor(
            data_source,
            num_rows="dynamic",
            hide_index=False,
            use_container_width=True,
        )

# ---------------------------------------------------------
# Evaluation & Verbosity
# ---------------------------------------------------------

col_eval, col_verbosity = st.columns(2)

with col_eval:

    with st.container(border=True):

        st.subheader("2. Evaluation")

        evaluation_mode = st.radio(
            "What do you want to calculate?",
            options=[
                "Calculate y from x",
                "Calculate x from y",
            ],
        )

        if evaluation_mode == "Calculate y from x":

            evaluation_value = st.number_input(
                "Enter x value",
                value=0.0,
            )

        else:

            evaluation_value = st.number_input(
                "Enter y value",
                value=0.0,
            )


with col_verbosity:

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
# 3. Calculate
# ---------------------------------------------------------
calculate = st.button(
    "Calculate",
    type="primary",
    use_container_width=True,
)

if calculate:

    try:

        clean_data = data.dropna(subset=["x", "y"])

        x = clean_data["x"].to_numpy(dtype=float)
        y = clean_data["y"].to_numpy(dtype=float)

        result = linear_least_squares(x, y)

        slope = result["slope"]
        intercept = result["intercept"]

        # ---------------------------------------------------------
        # Evaluate regression
        # ---------------------------------------------------------

        if evaluation_mode == "Calculate y from x":

            x_input = evaluation_value
            y_output = slope * x_input + intercept

            st.subheader("Evaluation Result")

            # Verbosity >= 1:
            # Show the general equation
            if verbosity >= VERBOSITY_SUMMARY:

                st.latex(
                    r"""
                    \hat{y} = mx + b
                    """
                )

            # Verbosity >= 2:
            # Show numerical substitution
            if verbosity >= VERBOSITY_FULL:

                if intercept >= 0:

                    st.latex(
                        rf"""
                        \hat{{y}}
                        =
                        ({slope:.6f})({x_input:.6f})
                        +
                        {intercept:.6f}
                        =
                        {y_output:.6f}
                        """
                    )

                else:

                    st.latex(
                        rf"""
                        \hat{{y}}
                        =
                        ({slope:.6f})({x_input:.6f})
                        -
                        {abs(intercept):.6f}
                        =
                        {y_output:.6f}
                        """
                    )

            # Always visible
            st.metric(
                "Calculated y",
                f"{y_output:.6f}",
            )


        else:

            y_input = evaluation_value

            st.subheader("Evaluation Result")

            if slope == 0:

                st.warning(
                    "x cannot be calculated because the slope is zero."
                )

            else:

                x_output = (
                    y_input - intercept
                ) / slope

                # Verbosity >= 1:
                # Show general equation
                if verbosity >= VERBOSITY_SUMMARY:

                    st.latex(
                        r"""
                        x =
                        \frac{
                        y - b
                        }{
                        m
                        }
                        """
                    )

                # Verbosity >= 2:
                # Show numerical substitution
                if verbosity >= VERBOSITY_FULL:

                    st.latex(
                        rf"""
                        x =
                        \frac{{
                        {y_input:.6f}
                        -
                        ({intercept:.6f})
                        }}{{
                        {slope:.6f}
                        }}
                        =
                        {x_output:.6f}
                        """
                    )

                # Always visible
                st.metric(
                    "Calculated x",
                    f"{x_output:.6f}",
                )

        correlation_coefficient = result[
            "correlation_coefficient"
        ]

        determination_coefficient = result[
            "determination_coefficient"
        ]

        y_predicted = result["y_predicted"]
        calculation_table = result["calculation_table"]
        summations = result["summations"]

        n = summations["n"]
        sum_x = summations["Σxi"]
        sum_y = summations["Σyi"]
        sum_xy = summations["Σxi·yi"]
        sum_x_squared = summations["Σxi²"]
        sum_y_squared = summations["Σyi²"]


        # =====================================================
        # VERBOSITY 2
        # Full calculation
        # =====================================================

        if verbosity >= VERBOSITY_FULL:

            # -------------------------------------------------
            # Calculation table
            # -------------------------------------------------

            st.subheader("2. Calculation Table")

            display_table = calculation_table.copy()

            sum_row = pd.DataFrame(
                [
                    {
                        "xi": summations["Σxi"],
                        "yi": summations["Σyi"],
                        "xi·yi": summations["Σxi·yi"],
                        "xi²": summations["Σxi²"],
                        "yi²": summations["Σyi²"],
                        "ŷi": np.nan,
                        "βi": np.nan,
                        "βi²": summations["Σβi²"],
                    }
                ],
                index=["Σ"],
            )

            display_table = pd.concat(
                [display_table, sum_row]
            )

            st.dataframe(
                display_table.style.format(
                    {
                        "xi": "{:.4f}",
                        "yi": "{:.4f}",
                        "xi·yi": "{:.4f}",
                        "xi²": "{:.4f}",
                        "yi²": "{:.4f}",
                        "ŷi": "{:.4f}",
                        "βi": "{:.4f}",
                        "βi²": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )


            # -------------------------------------------------
            # Slope
            # -------------------------------------------------

            st.subheader("3. Slope")

            st.latex(
                r"""
                m =
                \frac{
                n\sum x_i y_i
                -
                \sum x_i \sum y_i
                }{
                n\sum x_i^2
                -
                (\sum x_i)^2
                }
                """
            )

            st.latex(
                rf"""
                m =
                \frac{{
                ({n})({sum_xy:.4f})
                -
                ({sum_x:.4f})({sum_y:.4f})
                }}{{
                ({n})({sum_x_squared:.4f})
                -
                ({sum_x:.4f})^2
                }}
                =
                {slope:.6f}
                """
            )


            # -------------------------------------------------
            # Y-intercept
            # -------------------------------------------------

            st.subheader("4. Y-Intercept")

            st.latex(
                r"""
                b =
                \frac{
                \sum y_i
                -
                m\sum x_i
                }{n}
                """
            )

            st.latex(
                rf"""
                b =
                \frac{{
                {sum_y:.4f}
                -
                ({slope:.6f})({sum_x:.4f})
                }}{{
                {n}
                }}
                =
                {intercept:.6f}
                """
            )


            # -------------------------------------------------
            # Correlation coefficient
            # -------------------------------------------------

            st.subheader("5. Correlation Coefficient")

            st.latex(
                r"""
                r =
                \frac{
                n\sum x_i y_i
                -
                \sum x_i \sum y_i
                }{
                \sqrt{
                \left[
                n\sum x_i^2
                -
                (\sum x_i)^2
                \right]
                \left[
                n\sum y_i^2
                -
                (\sum y_i)^2
                \right]
                }
                }
                """
            )

            st.latex(
                rf"""
                r =
                \frac{{
                ({n})({sum_xy:.4f})
                -
                ({sum_x:.4f})({sum_y:.4f})
                }}{{
                \sqrt{{
                \left[
                ({n})({sum_x_squared:.4f})
                -
                ({sum_x:.4f})^2
                \right]
                \left[
                ({n})({sum_y_squared:.4f})
                -
                ({sum_y:.4f})^2
                \right]
                }}
                }}
                =
                {correlation_coefficient:.6f}
                """
            )


            # -------------------------------------------------
            # Coefficient of determination
            # -------------------------------------------------

            st.subheader("6. Coefficient of Determination")

            st.latex(
                rf"""
                R^2
                =
                r^2
                =
                ({correlation_coefficient:.6f})^2
                =
                {determination_coefficient:.6f}
                """
            )


        # =====================================================
        # VERBOSITY 1
        # Summary
        # =====================================================

        if verbosity >= VERBOSITY_SUMMARY:

            st.subheader("Calculation Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            summary_col1.metric(
                "n",
                f"{n}",
            )

            summary_col1.metric(
                "Σxi",
                f"{sum_x:.4f}",
            )

            summary_col2.metric(
                "Σyi",
                f"{sum_y:.4f}",
            )

            summary_col2.metric(
                "Σxi·yi",
                f"{sum_xy:.4f}",
            )

            summary_col3.metric(
                "Σxi²",
                f"{sum_x_squared:.4f}",
            )

            summary_col3.metric(
                "Σyi²",
                f"{sum_y_squared:.4f}",
            )


        # =====================================================
        # VERBOSITY 0
        # Final results
        # Always visible
        # =====================================================

        # -------------------------------------------------
        # Regression equation
        # -------------------------------------------------

        st.subheader("Regression Equation")

        sign = "+" if intercept >= 0 else "-"

        st.latex(
            rf"""
            \hat{{y}}
            =
            {slope:.6f}x
            {sign}
            {abs(intercept):.6f}
            """
        )


    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(
            f"An unexpected error occurred: {error}"
        )