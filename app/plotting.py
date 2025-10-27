import pandas as pd
import plotly.express as px
import streamlit as st
import networkx as nx
from pyvis.network import Network
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.figure_factory as ff
from streamlit_plotly_events import plotly_events

# --- Standard categorical color palette ---
categorical_colors = px.colors.qualitative.Set1  # visually distinct colors

def plot_scatter(df, components, metadata_cols, method="PCA", explained_var=None):
    n_components = components.shape[1]
    comp_cols = [f"Dim{i+1}" for i in range(n_components)]

    #df = df.reset_index(drop=True)
    df_components = pd.DataFrame(components, columns=comp_cols)
    df_plot = pd.concat([df, df_components], axis=1)
    df_plot["_index"] = df_plot.index

    colour_candidates = [col for col in metadata_cols if df[col].nunique() < 50]
    x_axis = st.selectbox(f"{method} X-axis", comp_cols, index=0)
    y_axis = st.selectbox(f"{method} Y-axis", comp_cols, index=1)
    colour_by = st.selectbox(f"Colour by ({method})", colour_candidates if colour_candidates else [None])

    fig = px.scatter(
        df_plot,
        x=x_axis,
        y=y_axis,
        color=colour_by if colour_by else None,
        hover_data=metadata_cols,
        color_discrete_sequence=categorical_colors
    )

    # --- Capture selection with lasso/box ---
    selected_points = plotly_events(
        fig,
        select_event=True,   # enables lasso/box
        click_event=False,
        hover_event=False,
        override_height=600,
        override_width="100%"
    )

    # --- Show explained variance ---
    if explained_var is not None:
        st.markdown(
            f"**Explained Variance:** {x_axis}: {explained_var[comp_cols.index(x_axis)]:.2f}, "
            f"{y_axis}: {explained_var[comp_cols.index(y_axis)]:.2f}"
        )

    # --- If points selected, show table ---
    if selected_points:
        # selected_points contains x, y coordinates
        selected_idx = []
        for p in selected_points:
            x_val = p['x']
            y_val = p['y']
            # Find matching row in df_plot
            mask = (df_plot[x_axis] == x_val) & (df_plot[y_axis] == y_val)
            idx = df_plot[mask].index.tolist()
            if idx:
                selected_idx.extend(idx)
        
        if selected_idx:
            st.write("### Selected Points")
            st.dataframe(df_plot.loc[selected_idx][metadata_cols + [x_axis, y_axis]])

def draw_pyvis_network(corr_matrix, threshold=0.95, hide_isolated=True):
    G = nx.Graph()
    compounds = corr_matrix.columns
    for i, c1 in enumerate(compounds):
        for j, c2 in enumerate(compounds):
            if i >= j:
                continue
            corr_val = corr_matrix.loc[c1, c2]
            if abs(corr_val) >= threshold:
                G.add_edge(c1, c2, weight=corr_val)

    if hide_isolated:
        G.remove_nodes_from([node for node, degree in dict(G.degree()).items() if degree == 0])

    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.1,
                  spring_length=200, spring_strength=0.05,
                  damping=0.09)
    return net

def plot_feature_exploration(df, compound_cols, metadata_cols, scale=True):
    # --- Select feature ---
    feature_x = st.selectbox("Feature X", compound_cols, index=0)
    color_by = st.selectbox("Color by (optional)", [None] + metadata_cols, index=0)

    # --- Prepare df_plot ---
    df_plot = np.log1p(df[[feature_x]].copy())

    # --- Scale if selected ---
    if scale:
        df_plot[feature_x] = StandardScaler().fit_transform(df_plot[[feature_x]])

    # --- Add metadata columns ---
    meta_to_add = []
    for col in ["Patient ID", "Batch number", color_by]:
        if col and col in df.columns:
            df_plot[col] = df[col]
            meta_to_add.append(col)

    # --- Show basic statistics ---
    st.write("### Basic statistics")
    st.write(df_plot.describe())

    # --- Violin plot ---
    fig_violin = px.violin(
        df_plot,
        x=color_by if color_by else None,
        y=feature_x,
        color=color_by if color_by else None,
        box=True,
        points=False,
        hover_data=meta_to_add,
        color_discrete_sequence=categorical_colors,
        title=f"Violin plot of {feature_x}" + (f" colored by {color_by}" if color_by else "")
    )
    fig_violin.update_traces(width=0.7)
    # remove x tick labels and axis title
    fig_violin.update_layout(
        xaxis=dict(showticklabels=False, title=None),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    # --- Scatter plot ---
    df_plot['_index'] = df_plot.index  # numeric x-axis
    fig_strip = px.scatter(
        df_plot,
        x='_index',
        y=feature_x,
        color=color_by if color_by else None,
        hover_data=meta_to_add,
        color_discrete_sequence=categorical_colors,
        title=f"Scatter plot of {feature_x} colored by {color_by}" if color_by else f"Scatter plot of {feature_x}"
    )
    st.plotly_chart(fig_strip, use_container_width=True)
