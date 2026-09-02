import json
import pandas as pd


def load_tabular_data(uploaded_file):
    """
    Read tabular data from CSV, Excel, or JSON.

    Parameters
    ----------
    uploaded_file
        File uploaded through Streamlit.

    Returns
    -------
    pandas.DataFrame
        Parsed tabular data.

    Raises
    ------
    ValueError
        If the file type is not supported or cannot be interpreted.
    """

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        if file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

        if file_name.endswith(".json"):

            data = json.load(uploaded_file)

            return pd.DataFrame(data)

        raise ValueError(
            "Unsupported file format. Use CSV, XLSX, XLS, or JSON."
        )

    except Exception as error:
        raise ValueError(
            f"Could not read file: {error}"
        ) from error