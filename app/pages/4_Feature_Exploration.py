import streamlit as st
from utils import setup_page
from plotting import plot_feature_exploration, plot_compound_across_groups

st.set_page_config(page_title="Feature Exploration", layout="wide")
st.title("Feature Exploration")

df_filtered, compound_cols, metadata_cols, scaler_type = setup_page()

section = st.radio("View", ["Single Feature", "Compound Across Groups"], horizontal=True)

if section == "Single Feature":
    plot_feature_exploration(df_filtered, compound_cols, metadata_cols, scaler_type=scaler_type)

elif section == "Compound Across Groups":
    col1, col2 = st.columns(2)
    with col1:
        selected_compound = st.selectbox("Select compound", compound_cols, key="cag_compound")
    with col2:
        group_col = st.selectbox("Group by (x-axis)", [None] + metadata_cols, key="cag_group")

    if selected_compound and group_col:
        n_groups = df_filtered[group_col].nunique()

        fig = plot_compound_across_groups(
            df_filtered, selected_compound, group_col, metadata_cols=metadata_cols
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Group summary statistics"):
            summary = df_filtered.groupby(group_col)[selected_compound].agg(
                N='count',
                Mean='mean',
                Median='median',
                SD='std',
                CV_pct=lambda x: 100 * x.std() / x.mean() if x.mean() != 0 else float('nan'),
                Detection_pct=lambda x: 100 * (x > 0).mean()
            ).reset_index()
            summary.columns = [group_col, 'N', 'Mean', 'Median', 'SD', 'CV (%)', 'Detection (%)']
            st.dataframe(summary.style.background_gradient(
                cmap='RdYlGn', subset=['Detection (%)']
            ), use_container_width=True, hide_index=True)
