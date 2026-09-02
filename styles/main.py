import streamlit as st


def apply_global_styles():

    st.markdown(
        """
        <style>

        /* Main page width */
        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Main title */
        h1 {
            font-weight: 700;
            margin-bottom: 1rem;
        }

        /* Section titles */
        h2 {
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.4rem;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 1rem;
        }

        /* Dataframes */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Inputs */
        div[data-baseweb="input"] {
            border-radius: 8px;
        }

        /* Radio group spacing */
        div[role="radiogroup"] {
            gap: 1rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )