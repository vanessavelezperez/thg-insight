import streamlit as st
import pandas as pd
from utils import load_data, load_npy_peaks, serialize_dataset

st.set_page_config(page_title="VOC Analytics", layout="wide")
st.title("VOC Analytics Dashboard")

# ── Data upload ───────────────────────────────────────────────────────────────
st.header("1. Upload Data")

uploaded_file = st.file_uploader("Upload your data file", type=["xlsx", "xls", "csv"])

if not uploaded_file:
    st.session_state['df'] = None
    st.info("Please upload an Excel or CSV file to begin.")
    st.stop()

if uploaded_file.name.endswith((".xlsx", ".xls")):
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet_name = st.selectbox("Select worksheet", xls.sheet_names)
    df = load_data(uploaded_file, sheet_name, type='excel')
else:
    df = load_data(uploaded_file, sheet_name=None, type='csv')

# Invalidate processed cache if file changed
if st.session_state.get('_loaded_file') != uploaded_file.name:
    st.session_state['_loaded_file'] = uploaded_file.name
    st.session_state.pop('_proc_key', None)
    st.session_state.pop('_proc_df', None)

st.success(f"Loaded {df.shape[0]} samples × {df.shape[1]} columns.")

with st.expander("Preview data"):
    st.dataframe(df.head(10), use_container_width=True)

# ── Metadata merge (optional) ──────────────────────────────────────────────────
st.header("2. Metadata (optional)")
st.caption("If sample metadata lives in a separate file, upload it and merge it in on a shared ID column. Every column from the metadata file becomes a metadata column.")

use_meta_file = st.checkbox("Import metadata from a separate file?", key="use_meta_file")

merged_meta_cols = []
if use_meta_file:
    meta_file = st.file_uploader("Upload metadata file", type=["xlsx", "xls", "csv"], key="meta_uploader")
    if meta_file:
        if meta_file.name.endswith((".xlsx", ".xls")):
            meta_xls = pd.ExcelFile(meta_file, engine="openpyxl")
            meta_sheet = st.selectbox("Select metadata worksheet", meta_xls.sheet_names, key="meta_sheet")
            meta_df = load_data(meta_file, meta_sheet, type='excel')
        else:
            meta_df = load_data(meta_file, sheet_name=None, type='csv')

        with st.expander("Preview metadata"):
            st.dataframe(meta_df.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            data_key = st.selectbox("Common column in data file", df.columns.tolist(), key="data_key")
        with c2:
            meta_key = st.selectbox("Common column in metadata file", meta_df.columns.tolist(), key="meta_key")
        how_label = st.radio(
            "Merge type", ["Keep all data rows (left join)", "Only matching rows (inner join)"],
            horizontal=True, key="meta_how"
        )
        how = 'left' if how_label.startswith("Keep") else 'inner'

        meta_cols_to_add = [c for c in meta_df.columns if c != meta_key]
        overlap = [c for c in meta_cols_to_add if c in df.columns]
        if overlap:
            meta_df = meta_df.rename(columns={c: f"{c}_meta" for c in overlap})
            meta_cols_to_add = [f"{c}_meta" if c in overlap else c for c in meta_cols_to_add]

        df = df.merge(meta_df, left_on=data_key, right_on=meta_key, how=how)
        if meta_key != data_key and meta_key in df.columns:
            df = df.drop(columns=[meta_key])

        merged_meta_cols = list(dict.fromkeys([data_key] + meta_cols_to_add))
        unmatched = int(df[meta_cols_to_add].isna().all(axis=1).sum()) if how == 'left' else 0
        msg = f"Merged {len(meta_cols_to_add)} metadata columns on '{data_key}' ({df.shape[0]} rows)"
        if unmatched:
            st.warning(msg + f" — {unmatched} rows had no metadata match.")
        else:
            st.success(msg)

st.session_state['df'] = df

# ── Column setup ──────────────────────────────────────────────────────────────
st.header("3. Column Setup")
st.caption("Select which columns are metadata. Everything else is treated as a compound.")

all_columns = df.columns.tolist()

with st.form("column_setup"):
    default_meta = [c for c in (merged_meta_cols or st.session_state.get('metadata_cols', [])) if c in all_columns]
    metadata_cols_form = st.multiselect("Select metadata columns", all_columns, default=default_meta)
    submitted = st.form_submit_button("Apply")

if submitted:
    st.session_state['metadata_cols'] = metadata_cols_form
    st.toast("Column selection applied!", icon="✅")

metadata_cols = st.session_state.get('metadata_cols', [])
compound_cols = [c for c in all_columns if c not in metadata_cols]
st.session_state['compound_cols'] = compound_cols

if metadata_cols and compound_cols:
    st.success(f"{len(compound_cols)} compound columns · {len(metadata_cols)} metadata columns")
    st.download_button(
        "Download merged dataset (CSV)",
        data=serialize_dataset(df, metadata_cols + compound_cols),
        file_name="prepared_dataset.csv",
        mime="text/csv",
    )
elif not metadata_cols:
    st.warning("Select at least one metadata column to continue.")
elif not compound_cols:
    st.warning("No compound columns remain — deselect some metadata columns.")

# ── Peak file (optional) ──────────────────────────────────────────────────────
st.header("4. Peak File (optional)")
st.caption("Only needed for the Analytical QC page.")

pk = st.checkbox("Upload a peak file?")
if pk:
    peaks_file = st.file_uploader("Upload peak file (.npy)", type=[".npy"])
    if peaks_file:
        peaks_compound_data = load_npy_peaks(peaks_file)
        st.session_state['peaks_compound_data'] = peaks_compound_data
        st.success(f"Loaded peaks for {peaks_compound_data['Compound'].nunique()} compounds.")
    else:
        st.session_state['peaks_compound_data'] = None
else:
    st.session_state['peaks_compound_data'] = None

# ── Page guide ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Pages")
st.markdown(
    """
| Page | Purpose |
|---|---|
| **Data Overview** | Sample counts, missing values, compound value distributions |
| **Analytical QC** | Peak shape, detectability and reproducibility from raw chromatography |
| **Feature Filtering** | Variance/CV, correlation, detection rate, D-ratio, group fold change |
| **Feature Exploration** | Single-compound deep dive: distributions, outliers, across-group view |
| **Multivariate** | PCA, t-SNE and UMAP for unsupervised structure exploration |
| **Supervised** | PLS-DA and per-compound univariate tests |
"""
)
