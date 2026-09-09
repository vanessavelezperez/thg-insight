import streamlit as st
import numpy as np
from utils import setup_page, match_compounds_by_rt_mass
from plotting import plot_peak_properties, plot_compound_peak_data

st.set_page_config(page_title="Analytical QC", layout="wide")
st.title("Analytical QC")
st.caption("Peak shape, detectability and reproducibility from raw chromatography data.")

df_filtered, compound_cols, metadata_cols, scaler_type = setup_page()

# ── Peaks file check ──────────────────────────────────────────────────────────
peaks_compound_data = st.session_state.get('peaks_compound_data', None)

if peaks_compound_data is None:
    st.info(
        "No peak file loaded. To use this page, upload a `.npy` peak file on the Home page "
        "by checking the 'Upload a peak file?' option."
    )
    st.stop()

# ── Match compounds ───────────────────────────────────────────────────────────
peak_to_result = match_compounds_by_rt_mass(compound_cols, peaks_compound_data['Compound'].unique())

if not peak_to_result:
    st.warning(
        "No compounds in the peak file matched the compound columns in your data. "
        "Check that compound names follow the `{RT}_{name}_{mass}` convention."
    )
    st.stop()

compound_data = peaks_compound_data[peaks_compound_data['Compound'].isin(peak_to_result)].copy()
compound_data['Compound'] = compound_data['Compound'].map(peak_to_result)

st.markdown(
    """
    **Analytical quality metrics:**
    - **Detectability** — fraction of runs where the compound was detected
    - **Reproducibility** — peak area RSD across runs
    - **Peak shape** — width consistency and asymmetry
    - **RT stability** — retention time standard deviation across runs
    """
)

selected_features = st.multiselect(
    "Filter to specific compounds (leave empty for all)",
    options=compound_cols
)

df_peaks = compound_data[compound_data['Compound'].isin(selected_features)] \
    if selected_features else compound_data

df_peaks["detected"] = df_peaks["peak_area"].apply(
    lambda x: x is not None and not np.isnan(x) and x > 0
)
df_peaks['peak_width'] = df_peaks['peak_width'] * 60

compound_stats = df_peaks.groupby('Compound').agg(
    detection_frac=('detected', lambda x: np.mean(x) * 100),
    n_detected=('detected', 'sum'),
    mean_area=('peak_area', 'mean'),
    cv_area=('peak_area', lambda x: (np.std(x) / np.mean(x)) * 100 if np.mean(x) > 0 else np.nan),
    mean_asym=('asymmetry', 'mean'),
    sd_asym=('asymmetry', 'std'),
    mean_width=('peak_width', 'mean'),
    sd_width=('peak_width', 'std'),
    rt_sd=('calc_RT', 'std'),
    mean_rt=('calc_RT', 'mean'),
    av_expr=('peak_area', lambda x: np.mean(x) / len(x))
).reset_index()

plot_peak_properties(compound_stats)

# ── Single compound detail ────────────────────────────────────────────────────
st.header("Compound Details")

available = compound_stats['Compound'].tolist()
if not available:
    st.info("No compounds to show.")
    st.stop()

selected_feature = st.selectbox("Select compound", available)
selected_stats = compound_stats[compound_stats['Compound'] == selected_feature].iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Detection Rate", f"{selected_stats['detection_frac']:.1f}%")
    st.metric("N Detected", f"{selected_stats['n_detected']:.0f}")
with col2:
    st.metric("Mean Area", f"{selected_stats['mean_area']:.2e}")
    st.metric("RSD Area", f"{selected_stats['cv_area']:.1f}%")
with col3:
    st.metric("Mean RT", f"{selected_stats['mean_rt']:.2f}")
    st.metric("RT SD", f"{selected_stats['rt_sd']:.3f}")
with col4:
    st.metric("Mean Width (s)", f"{selected_stats['mean_width']:.1f}")
    st.metric("Mean Asymmetry", f"{selected_stats['mean_asym']:.2f}")

plot_compound_peak_data(compound_data, selected_feature)
