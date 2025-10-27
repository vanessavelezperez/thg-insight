import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.colors import qualitative
from streamlit_plotly_events import plotly_events

# --- Set full-width layout ---
st.set_page_config(layout="wide")

# --- Colour palette for lines and categories ---
categorical_colors = qualitative.Set2 + qualitative.Pastel + qualitative.Set3

# --- Load CSV ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Add index as metadata option (avoid duplicates)
    df_plot = df.copy()
    idx_name = "__index__"
    while idx_name in df_plot.columns:
        idx_name += "_1"
    df_plot[idx_name] = df_plot.index

    metadata_cols = df_plot.columns.tolist()

    # --- Column selectors ---
    x_axis = st.selectbox("X-axis", metadata_cols, index=0)
    y_axis = st.selectbox("Y-axis", [c for c in metadata_cols if c != x_axis], index=1)

    # Colour by categorical columns only
    colour_candidates = [col for col in metadata_cols if df_plot[col].nunique() < 50]
    colour_by = st.selectbox("Colour by (categorical only)", colour_candidates if colour_candidates else [None])

    # --- Reference lines ---
    with st.expander("➕ Add reference lines"):
        vlines_input = st.text_input("Vertical lines (comma-separated X values)", value="")
        hlines_input = st.text_input("Horizontal lines (comma-separated Y values)", value="")

        vlines = [float(v.strip()) for v in vlines_input.split(",") if v.strip()]
        hlines = [float(h.strip()) for h in hlines_input.split(",") if h.strip()]

    # --- Plot ---
    fig = px.scatter(
        df_plot,
        x=x_axis,
        y=y_axis,
        color=colour_by if colour_by else None,
        hover_data=metadata_cols,
        color_discrete_sequence=categorical_colors
    )
    fig.update_layout(dragmode="lasso")

    # Add reference lines with cycling colours
    palette = categorical_colors
    for i, v in enumerate(vlines):
        fig.add_vline(x=v, line_dash="dash", line_color=palette[i % len(palette)])
    for j, h in enumerate(hlines):
        fig.add_hline(y=h, line_dash="dash", line_color=palette[j % len(palette)])

    # --- Display single plot and capture selection ---
    selected_points = plotly_events(
        fig,
        select_event=True,
        click_event=False,
        hover_event=False,
        override_height=600,
        override_width="100%"
    )

    st.subheader("Selected Data")
    if selected_points:
        selected_idx = [p["pointIndex"] for p in selected_points]

        # Ensure unique column names before showing
        df_display = df_plot.iloc[selected_idx].loc[:, ~df_plot.columns.duplicated()]
        st.dataframe(df_display)

        st.download_button(
            "Download selection as CSV",
            data=df_display.to_csv(index=False).encode("utf-8"),
            file_name="selection.csv",
            mime="text/csv"
        )
    else:
        st.write("No points selected yet.")
