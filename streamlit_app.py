import streamlit as st


st.set_page_config(
    page_title="Engineering Tools",
    page_icon="🧮",
    layout="wide",
)


pages = {
    "Engineering Tools": [
        st.Page(
            "pages/home.py",
            title="Home",
            icon="🏠",
            default=True,
        ),
    ],
    "Numerical Methods": [
        st.Page(
            "pages/numerical_methods/least_squares.py",
            title="Least Squares",
            icon="📈",
        ),
        st.Page(
            "pages/numerical_methods/gaussian_elimination.py",
            title="Gaussian Elimination",
            icon="🧮",
        ),
    ],
    "Structural Tools": [
        st.Page(
            "pages/structural/section_properties.py",
            title="Section Properties",
            icon="📐",
        ),
    ],

}


navigation = st.navigation(pages)

navigation.run()