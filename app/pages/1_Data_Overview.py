import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import setup_page

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data Overview")

df_filtered, compound_cols, metadata_cols, scaler_type = setup_page()

# ── Sample counts ─────────────────────────────────────────────────────────────
st.header("Sample Counts")

default_count_cols = [col for col in metadata_cols if df_filtered[col].nunique() <= 30]
selected_count_cols = st.multiselect(
    "Select metadata columns to display",
    options=metadata_cols,
    default=default_count_cols,
    key="overview_count_cols",
)

if selected_count_cols:
    ncols = min(len(selected_count_cols), 4)
    count_cols_layout = st.columns(ncols)
    for i, col in enumerate(selected_count_cols):
        with count_cols_layout[i % ncols]:
            counts = df_filtered[col].value_counts().reset_index()
            counts.columns = [col, 'N']
            st.write(f"**{col}**")
            st.dataframe(counts, hide_index=True, use_container_width=True)

if len(selected_count_cols) > 1:
    st.subheader("Cross-tabulation")
    row_col = st.selectbox("Row", selected_count_cols, key="crosstab_row")
    col_col = st.selectbox("Column", [c for c in selected_count_cols if c != row_col], key="crosstab_col")
    if col_col:
        crosstab = pd.crosstab(df_filtered[row_col], df_filtered[col_col])
        st.dataframe(crosstab, use_container_width=True)

# ── Group-by explorer ────────────────────────────────────────────────────────
st.subheader("Group-by Explorer")
st.caption(
    "Collapse the data to one row per unique combination of the group-by columns, "
    "then summarise the remaining metadata columns for each group."
)

groupby_cols = st.multiselect(
    "Group by",
    options=metadata_cols,
    key="overview_groupby_cols",
)

if groupby_cols:
    remaining_cols = [c for c in metadata_cols if c not in groupby_cols]
    agg_cols = st.multiselect(
        "Summarise these columns per group",
        options=remaining_cols,
        default=remaining_cols,
        key="overview_groupby_agg_cols",
    )
    agg_mode = st.radio(
        "Summary type",
        ["Unique count", "Unique values", "Most common value"],
        horizontal=True,
        key="overview_groupby_mode",
    )

    grouped = df_filtered.groupby(groupby_cols, dropna=False)
    rows = []
    for keys, g in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groupby_cols, keys))
        row["n_samples"] = len(g)
        for c in agg_cols:
            if agg_mode == "Unique count":
                row[c] = g[c].nunique(dropna=False)
            elif agg_mode == "Unique values":
                row[c] = ", ".join(str(v) for v in pd.unique(g[c]))
            else:  # Most common value
                mode = g[c].mode(dropna=False)
                row[c] = mode.iloc[0] if not mode.empty else None
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(groupby_cols, ignore_index=True)
    st.write(f"**{len(result)} groups** &nbsp;·&nbsp; {result['n_samples'].sum()} samples total")
    st.dataframe(result, hide_index=True, use_container_width=True)
    st.download_button(
        "Download grouped table (CSV)",
        result.to_csv(index=False).encode("utf-8"),
        "grouped_counts.csv",
        "text/csv",
        key="overview_groupby_download",
    )

# ── Compound matrix summary ───────────────────────────────────────────────────
st.header("Compound Matrix Summary")

X = df_filtered[compound_cols]
total = len(X)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Samples", total)
with col2:
    st.metric("Compounds", len(compound_cols))
with col3:
    zero_any = (X == 0).any(axis=1).sum()
    st.metric("Samples with any zero", int(zero_any))
with col4:
    fully_detected = (X > 0).all(axis=0).sum()
    st.metric("Compounds detected in all samples", int(fully_detected))

# ── Zero / missing rate per compound ─────────────────────────────────────────
st.subheader("Detection Rate Distribution")
st.caption("Values of 0 are treated as undetected (after log1p transform, 0 → 0).")

detection_rate = (X > 0).mean(axis=0) * 100
det_df = pd.DataFrame({'Compound': compound_cols, 'Detection Rate (%)': detection_rate.values})

bins = [0, 25, 50, 75, 90, 100]
labels = ['0–25%', '25–50%', '50–75%', '75–90%', '90–100%']
det_df['Bin'] = pd.cut(det_df['Detection Rate (%)'], bins=bins, labels=labels, include_lowest=True)
bin_counts = det_df['Bin'].value_counts().sort_index().reset_index()
bin_counts.columns = ['Detection Range', 'Compounds']

fig_bar = px.bar(bin_counts, x='Detection Range', y='Compounds',
                 title='Number of Compounds by Detection Rate',
                 color='Detection Range',
                 color_discrete_sequence=['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850'])
fig_bar.update_layout(showlegend=False, height=350)
st.plotly_chart(fig_bar, use_container_width=True)

with st.expander("View full detection rate table"):
    st.dataframe(det_df.sort_values('Detection Rate (%)', ascending=False), hide_index=True,
                 use_container_width=True)

# ── Value distribution ────────────────────────────────────────────────────────
st.subheader("Compound Value Distribution")

all_vals = X.values.flatten()
all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]

fig_hist = px.histogram(x=all_vals, nbins=100,
                        title="Distribution of all non-zero compound values",
                        labels={'x': 'Value', 'y': 'Frequency'})
fig_hist.update_layout(height=350, showlegend=False)
st.plotly_chart(fig_hist, use_container_width=True)

# Per-group value distribution
if metadata_cols:
    group_col = st.selectbox("Compare distribution by group", metadata_cols, key="dist_group")
    if group_col:
        melted = df_filtered[compound_cols + [group_col]].melt(
            id_vars=[group_col], var_name='Compound', value_name='Value'
        )
        melted = melted[melted['Value'] > 0]
        fig_box = px.box(melted, x=group_col, y='Value', color=group_col,
                         title=f"Compound value distribution by {group_col}",
                         color_discrete_sequence=px.colors.qualitative.Set1,
                         points=False)
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ── Descriptive stats ─────────────────────────────────────────────────────────
st.subheader("Compound Descriptive Statistics")
desc = X.describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'})
desc['cv_%'] = (desc['std'] / desc['mean'].abs() * 100).round(1)
st.dataframe(desc.style.background_gradient(cmap='RdYlGn', subset=['mean']), use_container_width=True)
