# utils.py
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(uploaded_file, sheet_name: str, type='csv') -> pd.DataFrame:
    if type == 'excel':
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        return pd.read_excel(xls, sheet_name=sheet_name)
    elif type == 'csv':
        return pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        st.stop()

def filter_dataframe(df, metadata_cols):
    """Filter df based on user selections in the sidebar."""
    df = df.dropna(subset=metadata_cols)

    # Sidebar filters
    for col in metadata_cols:
        options = sorted(df[col].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(f"Filter by {col}", options)
        if selected:
            df = df[df[col].astype(str).isin(selected)]

    # Fill numeric (non-metadata) columns with 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    compound_cols = [col for col in numeric_cols if col not in metadata_cols]
    df[compound_cols] = df[compound_cols].fillna(0)

    return df
