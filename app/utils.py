import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict


class ParetoScaler(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mean_) / (self.std_ ** self.alpha)


def get_scaler(scaler_type):
    if scaler_type == "Standard Scaler":
        return StandardScaler()
    elif scaler_type == "Robust Scaler":
        return RobustScaler()
    elif scaler_type == "Pareto Scaler":
        return ParetoScaler()
    return None


@st.cache_data
def load_data(uploaded_file, sheet_name: str, type='csv') -> pd.DataFrame:
    if type == 'excel':
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        return pd.read_excel(xls, sheet_name=sheet_name)
    elif type == 'csv':
        return pd.read_csv(uploaded_file, low_memory=False)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        st.stop()


def setup_page():
    """
    Sidebar controls + data for every analysis page.

    The full dataset is cleaned (fillna, clip, log transform) once and stored.
    It is only re-processed when log_transform or compound_cols change.
    Filters are just row slices on the already-clean data — no re-cleaning.

    Returns: df_filtered, compound_cols, metadata_cols, scaler_type
    """
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.warning("No data loaded. Please upload a file on the Home page first.")
        st.stop()

    metadata_cols = st.session_state.get('metadata_cols', [])
    compound_cols = list(st.session_state.get('compound_cols', []))

    if not metadata_cols or not compound_cols:
        st.warning("Please complete column setup on the Home page first.")
        st.stop()

    df = st.session_state['df']

    # ── Sidebar: preprocessing ────────────────────────────────────────────────
    log_transform = st.sidebar.checkbox("Apply log1p transformation", value=True, key='log_transform')
    scaler_type = st.sidebar.selectbox(
        "Scaler", ["None", "Standard Scaler", "Robust Scaler", "Pareto Scaler"],
        index=1, key='scaler_type'
    )
    isc = st.sidebar.checkbox("Exclude Internal Standards?", key='exclude_standards')
    if isc:
        st.sidebar.multiselect("Standards to exclude", compound_cols, key='standards')
        compound_cols = [c for c in compound_cols if c not in st.session_state.get('standards', [])]

    # ── Sidebar: bin numeric metadata ─────────────────────────────────────────
    numeric_meta_candidates = [
        c for c in metadata_cols
        if c in df.columns and pd.to_numeric(df[c], errors='coerce').notna().mean() > 0.9
    ]
    bin_settings = {}
    if numeric_meta_candidates:
        with st.sidebar.expander("Bin numeric metadata"):
            bin_cols = st.multiselect(
                "Columns to bin", numeric_meta_candidates, key='bin_cols',
                help="Turns a continuous metadata column into categories usable for grouping/colouring."
            )
            for col in bin_cols:
                n_bins = st.slider(f"'{col}' — number of bins", 2, 10, 4, key=f"n_bins__{col}")
                method = st.radio(
                    f"'{col}' — method", ["Equal-width", "Quantile"],
                    key=f"bin_method__{col}", horizontal=True
                )
                bin_settings[col] = (n_bins, method)

    st.sidebar.divider()

    # ── Clean full dataset once; only redo when preprocessing settings change ─
    # Metadata cols are string-converted here so we don't repeat it on every render.
    proc_key = (id(df), tuple(compound_cols), log_transform, tuple(sorted(bin_settings.items())))
    if st.session_state.get('_proc_key') != proc_key:
        proc_df = df.copy()
        try:
            proc_df[compound_cols] = proc_df[compound_cols].replace(
                [np.nan, -np.inf, np.inf], 0
            ).clip(lower=0)
        except Exception:
            proc_df[compound_cols] = proc_df[compound_cols].apply(
                pd.to_numeric, errors='coerce'
            ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        if log_transform:
            proc_df[compound_cols] = np.log1p(proc_df[compound_cols])
        for col in metadata_cols:
            if col in proc_df.columns:
                proc_df[col] = proc_df[col].astype("string").fillna("NA")

        binned_cols = []
        for col, (n_bins, method) in bin_settings.items():
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            try:
                if method == "Quantile":
                    binned = pd.qcut(numeric_vals, q=n_bins, duplicates='drop', precision=2)
                else:
                    binned = pd.cut(numeric_vals, bins=n_bins, precision=2)
                new_col = f"{col} (binned)"
                proc_df[new_col] = binned.astype(str).where(binned.notna(), "NA")
                binned_cols.append(new_col)
            except Exception:
                st.sidebar.warning(f"Could not bin '{col}' — check it has enough distinct numeric values.")

        st.session_state['_proc_df'] = proc_df
        st.session_state['_proc_key'] = proc_key
        st.session_state['_binned_cols'] = binned_cols
        st.session_state.pop('_df_filter_key', None)

    proc_df = st.session_state['_proc_df']
    metadata_cols = metadata_cols + st.session_state.get('_binned_cols', [])

    # ── Build filter options once per dataset ─────────────────────────────────
    if st.session_state.get('_filter_options_id') != proc_key:
        st.session_state['_filter_options'] = {
            col: sorted(proc_df[col].dropna().astype(str).unique())
            for col in metadata_cols if col in proc_df.columns
        }
        st.session_state['_filter_options_id'] = proc_key

    # ── Sidebar: filter widgets ───────────────────────────────────────────────
    st.sidebar.markdown("### Filters")
    filter_selections = {}
    for col in metadata_cols:
        options = st.session_state['_filter_options'].get(col, [])
        selected = st.sidebar.multiselect(col, options, key=f"filter__{col}")
        filter_selections[col] = selected

    # ── Apply filters only when selections change (cache result in session state)
    filter_vals = tuple(tuple(v) for v in filter_selections.values())
    df_filter_key = (proc_key, filter_vals)

    if st.session_state.get('_df_filter_key') != df_filter_key:
        mask = pd.Series(True, index=proc_df.index)
        for col, selected in filter_selections.items():
            if selected:
                mask &= proc_df[col].astype(str).isin(selected)
        df_filtered = proc_df[mask].reset_index(drop=True)
        if df_filtered.empty:
            st.warning("No data after filtering.")
            st.stop()
        st.session_state['_df_filtered'] = df_filtered
        st.session_state['_df_filter_key'] = df_filter_key

    df_filtered = st.session_state.get('_df_filtered')
    if df_filtered is None or df_filtered.empty:
        st.warning("No data after filtering.")
        st.stop()

    return df_filtered, compound_cols, metadata_cols, scaler_type


@st.cache_data
def serialize_dataset(df: pd.DataFrame, ordered_cols: list) -> bytes:
    """Reorder columns and serialize to CSV bytes for download."""
    return df[[c for c in ordered_cols if c in df.columns]].to_csv(index=False).encode("utf-8")


@st.cache_data
def load_npy_peaks(uploaded_file):
    numpy_data = np.load(uploaded_file, allow_pickle=True)
    data = numpy_data.item()
    rows = []
    for filename, peaks in data.items():
        for peak in peaks:
            peak_start = peak.get('peak_start', None)
            peak_end = peak.get('peak_end', None)
            peak_width = peak_end[0] - peak_start[0] if peak_start and peak_end else np.nan
            rows.append({
                "Compound": peak.get("Compound", None),
                "calc_RT": peak.get("calc_RT", None),
                "peak_area": peak.get("peak_area", None),
                "asymmetry": peak.get("asymmetry", None),
                "mass_spec": peak.get("mass_spec", []),
                "peakRT": peak.get("peakRT"),
                "peak_width": peak_width,
                "peak_I": peak.get("peak_I", None),
                "signal": peak.get("mz", None),
                "filename": filename
            })
    return pd.DataFrame(rows)


def match_compounds_by_rt_mass(compound_cols, peak_compounds):
    def parse_rt_mass(name):
        parts = name.split('_')
        if len(parts) < 2:
            return None, None
        return parts[0], parts[-1]

    result_lookup = {}
    for col in compound_cols:
        rt, mass = parse_rt_mass(col)
        if rt is not None:
            result_lookup[(rt, mass)] = col

    mapping = {}
    for peak_name in peak_compounds:
        rt, mass = parse_rt_mass(str(peak_name))
        if (rt, mass) in result_lookup:
            mapping[peak_name] = result_lookup[(rt, mass)]

    return mapping


def parse_mass_spec(name, peak_num, array):
    mass_spec_string = ""
    num_peaks = 0
    for m, i in enumerate(array):
        if i != 0:
            num_peaks += 1
            mass_spec_string += f"{m} {i}; "
    return {
        'Name': name,
        'DB#': peak_num,
        'Num Peaks': num_peaks,
        'mass_spec': mass_spec_string
    }
