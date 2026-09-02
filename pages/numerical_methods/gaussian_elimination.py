import numpy as np
import pandas as pd
import streamlit as st

from calculations.numerical_methods.gaussian_elimination import (
    gaussian_elimination,
)

from styles.main import apply_global_styles


apply_global_styles()


# ---------------------------------------------------------
# Verbosity levels
# ---------------------------------------------------------

VERBOSITY_RESULTS = 0
VERBOSITY_SUMMARY = 1
VERBOSITY_FULL = 2


st.title("Gaussian Elimination")


# ---------------------------------------------------------
# 1. Number of equations
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("1. System Definition")

    number_of_equations = st.number_input(
        "Number of equations",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
    )

    number_of_equations = int(number_of_equations)


# ---------------------------------------------------------
# 2. Variable names
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("2. Variable Names")

    default_names = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
    ]

    variable_names = []

    columns = st.columns(number_of_equations)

    for i in range(number_of_equations):

        with columns[i]:

            variable_name = st.text_input(
                f"Variable {i + 1}",
                value=default_names[i],
                key=f"variable_{i}",
            )

            variable_names.append(variable_name)


# ---------------------------------------------------------
# 3. Input matrix
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("3. Equation Coefficients")

    table_columns = variable_names + ["Result"]

    default_data = pd.DataFrame(
        0.0,
        index=range(number_of_equations),
        columns=table_columns,
    )

    equation_table = st.data_editor(
        default_data,
        num_rows="fixed",
        hide_index=False,
        use_container_width=True,
        key="gaussian_table",
    )


# ---------------------------------------------------------
# 4. Verbosity
# ---------------------------------------------------------

with st.container(border=True):

    st.subheader("4. Output Detail")

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
# 5. Solve
# ---------------------------------------------------------

if st.button(
    "Solve System",
    type="primary",
    use_container_width=True,
):

    try:

        # -------------------------------------------------
        # Split A and b
        # -------------------------------------------------

        A = equation_table[
            variable_names
        ].to_numpy(dtype=float)

        b = equation_table[
            "Result"
        ].to_numpy(dtype=float)

        result = gaussian_elimination(
            A=A,
            b=b,
        )

        solution = result["solution"]
        upper_matrix = result["upper_matrix"]
        transformed_b = result["transformed_b"]
        steps = result["steps"]


        # =================================================
        # VERBOSITY 2
        # Full calculation
        # =================================================

        if verbosity >= VERBOSITY_FULL:

            st.subheader("Gaussian Elimination Steps")

            for step_number, step in enumerate(
                steps,
                start=1,
            ):

                st.markdown(
                    f"### Step {step_number}"
                )

                # -------------------------------------------------
                # Row swap
                # -------------------------------------------------

                if step["type"] == "swap":

                    row_1 = step["row_1"]
                    row_2 = step["row_2"]

                    st.markdown(
                        f"""
                        Row swap:

                        `R{row_1 + 1} ↔ R{row_2 + 1}`
                        """
                    )

                # -------------------------------------------------
                # Row elimination
                # -------------------------------------------------

                elif step["type"] == "elimination":

                    pivot_row = step["pivot_row"]
                    target_row = step["target_row"]
                    factor = step["factor"]

                    st.markdown(
                        f"""
                        Row operation:

                        `R{target_row + 1} = R{target_row + 1} - ({factor:.6f}) R{pivot_row + 1}`
                        """
                    )

                # -------------------------------------------------
                # Matrix after the operation
                # -------------------------------------------------

                step_matrix = np.column_stack(
                    (
                        step["matrix"],
                        step["rhs"],
                    )
                )

                step_columns = (
                    variable_names
                    + ["Result"]
                )

                step_df = pd.DataFrame(
                    step_matrix,
                    columns=step_columns,
                )

                st.dataframe(
                    step_df.style.format("{:.6f}"),
                    use_container_width=True,
                )


        # =================================================
        # VERBOSITY 1
        # Summary
        # =================================================

        if verbosity >= VERBOSITY_SUMMARY:

            st.subheader("Upper Triangular System")

            upper_system = np.column_stack(
                (
                    upper_matrix,
                    transformed_b,
                )
            )

            upper_df = pd.DataFrame(
                upper_system,
                columns=variable_names + ["Result"],
            )

            st.dataframe(
                upper_df.style.format("{:.6f}"),
                use_container_width=True,
            )


        # =================================================
        # VERBOSITY 0
        # Results
        # =================================================

        st.subheader("Solution")

        result_columns = st.columns(
            number_of_equations
        )

        for i, variable_name in enumerate(
            variable_names
        ):

            result_columns[i].metric(
                variable_name,
                f"{solution[i]:.6f}",
            )


    except ValueError as error:

        st.error(str(error))


    except Exception as error:

        st.error(
            f"An unexpected error occurred: {error}"
        )