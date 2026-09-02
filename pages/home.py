import streamlit as st
from styles.main import apply_global_styles
apply_global_styles()

st.title("Engineering Tools")

st.write(
    """
    A collection of interactive tools for engineering,
    mathematics, and numerical methods.
    """
)

st.subheader("Available tools")

st.markdown(
    """
    **Numerical Methods**
    
    - Least Squares
    """
)