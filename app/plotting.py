import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
from pyvis.network import Network
import numpy as np
import plotly.figure_factory as ff
from streamlit_plotly_events import plotly_events
from scipy import stats
from utils import parse_mass_spec, get_scaler

categorical_colors = px.colors.qualitative.Set1


def get_ellipse(x, y):
    if len(x) < 2:
        return [], [], None
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    angle = np.arctan2(*eigenvecs[:, 0][::-1])
    chisq_val = np.sqrt(stats.chi2.ppf(0.95, df=2))
    width, height = 2 * chisq_val * np.sqrt(eigenvals)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array([width / 2 * np.cos(theta), height / 2 * np.sin(theta)])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ellipse_rot = R @ ellipse
    ellipse_x, ellipse_y = ellipse_rot[0] + mean_x, ellipse_rot[1] + mean_y
    return ellipse_x, ellipse_y, (mean_x, mean_y, width, height, R)


def get_ellipse_outliers(df_group, x_col, y_col, mean_x, mean_y, width, height, R):
    points = df_group[[x_col, y_col]].values
    diff = (points - [mean_x, mean_y]) @ R
    inside = (diff[:, 0] / (width / 2)) ** 2 + (diff[:, 1] / (height / 2)) ** 2 <= 1
    return df_group[~inside]


def plot_scatter(df, components, metadata_cols, method="PCA", explained_var=None):
    n_components = components.shape[1]
    comp_cols = [f"Dim{i+1}" for i in range(n_components)]
    df_components = pd.DataFrame(components, columns=comp_cols)
    df_plot = pd.concat([df, df_components], axis=1)
    df_plot["_index"] = df_plot.index

    colour_candidates = [col for col in metadata_cols if df[col].nunique() < 50]
    x_axis = st.selectbox(f"{method} X-axis", comp_cols, index=0)
    y_axis = st.selectbox(f"{method} Y-axis", comp_cols, index=1)
    colour_by = st.selectbox(f"Colour by ({method})", colour_candidates if colour_candidates else [None])
    show_ellipses = st.checkbox("Show confidence ellipses", value=True, key=f"{method}_show_ellipses")

    # Only embed the first metadata column in hover to avoid DataCloneError —
    # plotly_events serialises the whole figure on selection events
    first_meta = metadata_cols[0] if metadata_cols else None
    hover_cols = [c for c in [first_meta] if c and c in df_plot.columns]

    fig = px.scatter(
        df_plot, x=x_axis, y=y_axis,
        color=colour_by if colour_by else None,
        hover_data=hover_cols,
        color_discrete_sequence=categorical_colors
    )

    color_map = {}
    if colour_by and show_ellipses:
        unique_groups = df_plot[colour_by].unique()
        for i, group in enumerate(unique_groups):
            color_map[group] = categorical_colors[i % len(categorical_colors)]

        outlier_frames = []
        for group in unique_groups:
            subset = df_plot[df_plot[colour_by] == group]
            x = subset[x_axis].values
            y = subset[y_axis].values
            ellipse_x, ellipse_y, params = get_ellipse(x, y)
            fig.add_trace(go.Scatter(
                x=ellipse_x, y=ellipse_y, mode='none', fill='toself',
                name=f"{group} CI", fillcolor=color_map[group],
                opacity=0.1, showlegend=False, hoverinfo='skip'
            ))
            if params is not None:
                mean_x, mean_y, width, height, R = params
                outliers = get_ellipse_outliers(subset, x_axis, y_axis, mean_x, mean_y, width, height, R)
                if not outliers.empty:
                    outlier_frames.append(outliers)

        if outlier_frames:
            with st.expander("**Points outside 95% CI**"):
                cols = list(dict.fromkeys(metadata_cols + [colour_by, x_axis, y_axis]))
                st.dataframe(pd.concat(outlier_frames)[cols])

    selected_points = plotly_events(
        fig, select_event=True, click_event=False, hover_event=False,
        override_height=600, override_width="100%", key=f"{method}_scatter"
    )

    if explained_var is not None:
        st.markdown(
            f"**Explained Variance:** {x_axis}: {explained_var[comp_cols.index(x_axis)]:.2f}, "
            f"{y_axis}: {explained_var[comp_cols.index(y_axis)]:.2f}"
        )

    if selected_points:
        selected_idx = []
        for p in selected_points:
            mask = (df_plot[x_axis] == p['x']) & (df_plot[y_axis] == p['y'])
            selected_idx.extend(df_plot[mask].index.tolist())
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
        G.remove_nodes_from([n for n, d in dict(G.degree()).items() if d == 0])

    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.1,
                  spring_length=200, spring_strength=0.05, damping=0.09)
    return net


def plot_feature_violin(df, feature_x, color_by=None, metadata_cols=None):
    meta_to_add = [col for col in (metadata_cols or []) if col and col in df.columns]
    fig = px.violin(
        df, x=color_by if color_by else None, y=feature_x,
        color=color_by if color_by else None,
        box=True, points=False, hover_data=meta_to_add,
        color_discrete_sequence=categorical_colors, title=feature_x
    )
    fig.update_traces(width=0.7)
    fig.update_layout(
        title=dict(text=feature_x, font=dict(size=20, color="black", family="Arial, sans-serif",
                                              weight="bold"), x=0.5, xanchor="center"),
        xaxis=dict(showticklabels=False, title=None),
        yaxis=dict(tickfont=dict(size=14, color="black", family="Arial, sans-serif"), title=None),
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig

def plot_feature_histogram(df, feature_x, color_by=None, metadata_cols=None, show_bars=False, nbins=20, opacity=0.4):
    #meta_to_add = [col for col in (metadata_cols or []) if col and col in df.columns]
    if color_by:
        groups = df[color_by].dropna().unique()
        hist_data = [df.loc[df[color_by] == g, feature_x].dropna() for g in groups]
        labels = [str(g) for g in groups]
    else:
        hist_data = [df[feature_x].dropna()]
        labels = [feature_x]

    bin_size = None
    if show_bars:
        x = df[feature_x].dropna()
        bin_size = (x.max() - x.min()) / nbins

    fig = ff.create_distplot(
        hist_data,
        labels,
        show_hist=show_bars,
        show_rug=False,
        curve_type="kde",
        bin_size=bin_size,
    )

    for tr in fig.data:
        if tr.type == "scatter":
            tr.update(fill="tozeroy")
        elif tr.type == "histogram":
            tr.update(opacity=0.4)

    fig.update_layout(template="plotly_white", title=feature_x, barmode="overlay")

    return fig


def plot_feature_scatter(df, feature_x, color_by=None, metadata_cols=None,
                         lower_thresh=None, upper_thresh=None, method_label=None,
                         enable_selection=True):
    df_plot = df.copy()
    df_plot['_index'] = df_plot.index
    meta_to_add = [col for col in (metadata_cols or []) if col and col in df.columns]
    hover_cols = meta_to_add.copy()

    if lower_thresh is not None:
        df_plot['is_outlier'] = (df_plot[feature_x] < lower_thresh) | (df_plot[feature_x] > upper_thresh)
        df_plot['_outlier_label'] = df_plot['is_outlier'].map({True: 'Outlier', False: 'Inlier'})
        hover_cols.append('is_outlier')

    if lower_thresh is not None:
        fig = px.scatter(
            df_plot, x='_index', y=feature_x, color='_outlier_label', hover_data=hover_cols,
            color_discrete_map={'Inlier': 'green', 'Outlier': 'red'},
            category_orders={'_outlier_label': ['Inlier', 'Outlier']},
            title=f"Scatter plot of {feature_x}" + (f" colored by {color_by}" if color_by else "")
        )
    else:
        fig = px.scatter(
            df_plot, x='_index', y=feature_x,
            color=color_by if color_by else None, hover_data=hover_cols,
            color_discrete_sequence=categorical_colors,
            title=f"Scatter plot of {feature_x}" + (f" colored by {color_by}" if color_by else "")
        )

    if lower_thresh is not None:
        fig.add_hline(y=upper_thresh, line_dash="dash", line_color="red",
                      annotation_text=f"Upper ({method_label})", annotation_position="top left")
        fig.add_hline(y=lower_thresh, line_dash="dash", line_color="red",
                      annotation_text=f"Lower ({method_label})", annotation_position="bottom left")

    if enable_selection:
        selected_points = plotly_events(fig, select_event=True, click_event=False,
                                        hover_event=False, override_height=600, override_width="100%")
        df_selected = pd.DataFrame()
        if selected_points:
            selected_idx = [p["x"] for p in selected_points]
            selected_idx = df_plot[df_plot["_index"].isin(selected_idx)].index.tolist()
            if selected_idx:
                cols_to_show = meta_to_add + [feature_x]
                if 'is_outlier' in df_plot.columns:
                    cols_to_show.append('is_outlier')
                df_selected = df_plot.loc[selected_idx, cols_to_show].copy()
        return fig, df_selected
    else:
        st.plotly_chart(fig, use_container_width=True)
        return fig, None


def format_sig_figs(value, sig_figs=3):
    if pd.isna(value) or value == 0:
        return "0.00"
    try:
        return f"{float(f'{value:.{sig_figs}g}'):.3g}"
    except Exception:
        return str(value)


def plot_feature_exploration(df, compound_cols, metadata_cols, scaler_type="Standard Scaler"):
    from analysis import calculate_outlier_thresholds, detect_outliers_grouped

    feature_x = st.selectbox("Feature X", compound_cols, index=0)
    color_by = st.selectbox("Color by (optional)", [None] + metadata_cols, index=0)

    df_plot = df[[feature_x]].copy()
    scaler = get_scaler(scaler_type)
    if scaler:
        df_plot[feature_x] = scaler.fit_transform(df_plot[[feature_x]])
    for col in metadata_cols:
        if col and col in df.columns:
            df_plot[col] = df[col]

    st.write("### Basic statistics")

    def compute_stats(values):
        return {
            'count': len(values),
            'mean': values.mean(),
            'std': values.std(),
            'MAD': np.median(np.abs(values - values.median())),
            'min': values.min(),
            '25%': values.quantile(0.25),
            '50%': values.median(),
            '75%': values.quantile(0.75),
            'max': values.max(),
            'detection %': np.sum(values > 0) / len(values) * 100
        }

    values = df_plot[feature_x].dropna()
    if color_by and color_by in df_plot.columns:
        stats_data = {'All': compute_stats(values)}
        for group in df_plot[color_by].dropna().unique():
            group_values = df_plot[df_plot[color_by] == group][feature_x].dropna()
            if len(group_values) > 0:
                stats_data[str(group)] = compute_stats(group_values)
        stats_df = pd.DataFrame(stats_data)
    else:
        stats_df = pd.DataFrame(compute_stats(values), index=[feature_x]).T
    st.write(stats_df)

    view = st.radio("Plot", ["Violin", "KDE"], horizontal=True)
    if view == "Violin":
        fig = plot_feature_violin(df_plot, feature_x, color_by=color_by, metadata_cols=metadata_cols)
    elif view == "KDE":
        bars = st.checkbox("Show bars", value=False)
        nbins = st.slider("Number of bins", 1, 100, 20, 1)
        fig = plot_feature_histogram(df_plot, feature_x, color_by=color_by, metadata_cols=metadata_cols, show_bars=bars, nbins=nbins)
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Scatter Plot")
    st.caption("Use box select or lasso select to investigate specific points")

    outlier_method = st.selectbox(
        "Outlier Detection Method",
        ["None", "Mean ± 2 SD", "Mean ± 3 SD", "IQR (1.5×IQR)", "IQR (3×IQR)",
         "Modified Z-score (MAD)", "Percentile (5th/95th)", "Percentile (1st/99th)"],
        index=0
    )

    thresholds = calculate_outlier_thresholds(df_plot[feature_x], outlier_method)
    lower_thresh = thresholds['lower']
    upper_thresh = thresholds['upper']
    method_label = thresholds['label']

    if lower_thresh is not None:
        df_plot['is_outlier'] = (df_plot[feature_x] < lower_thresh) | (df_plot[feature_x] > upper_thresh)
        n_outliers = df_plot['is_outlier'].sum()
        pct_outliers = 100 * n_outliers / len(df_plot)

        inlier_vals = df_plot[~df_plot['is_outlier']][feature_x].dropna()
        inlier_rsd = (inlier_vals.std() / inlier_vals.mean() * 100) if inlier_vals.mean() != 0 else float('nan')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lower Threshold", f"{lower_thresh:.4f}")
        with col2:
            st.metric("Upper Threshold", f"{upper_thresh:.4f}")
        with col3:
            st.metric("Outliers Detected", f"{n_outliers} ({pct_outliers:.1f}%)")
        with col4:
            st.metric("Inlier RSD", f"{inlier_rsd:.1f}%" if not np.isnan(inlier_rsd) else "N/A")

        if n_outliers > 0:
            st.warning(f"**{n_outliers} outliers detected ({pct_outliers:.1f}%)** using {outlier_method}")
    else:
        df_plot['is_outlier'] = False
        n_outliers = 0
        pct_outliers = 0

    fig_scatter, df_selected = plot_feature_scatter(
        df_plot, feature_x, color_by=color_by, metadata_cols=metadata_cols,
        lower_thresh=lower_thresh, upper_thresh=upper_thresh,
        method_label=method_label, enable_selection=True
    )

    if not df_selected.empty:
        st.write("### Selected Points")
        st.dataframe(df_selected, use_container_width=True)

    if n_outliers > 0:
        with st.expander(f"View all {n_outliers} outliers ({pct_outliers:.1f}%)"):
            meta_cols = [col for col in metadata_cols if col in df_plot.columns]
            st.dataframe(df_plot[df_plot['is_outlier']][meta_cols + [feature_x]], use_container_width=True)

    st.write("---")
    st.write("### Grouped Outlier Detection")
    st.caption("Detect outliers within subgroups of your data")

    group_candidates = [col for col in metadata_cols if df[col].nunique() < 0.5 * len(df)]
    group_by = st.selectbox("Group by", [None] + group_candidates, index=0, key="grouped_outlier_group")

    if group_by:
        grouped_method = st.selectbox(
            "Outlier Detection Method (per group)",
            ["None", "Mean ± 2 SD", "Mean ± 3 SD", "IQR (1.5×IQR)", "IQR (3×IQR)",
             "Modified Z-score (MAD)", "Percentile (5th/95th)", "Percentile (1st/99th)"],
            index=5, key="grouped_outlier_method"
        )
        if grouped_method != "None":
            df_result = detect_outliers_grouped(df_plot, feature_x, group_by, method=grouped_method, scale=False)
            method_label_g = calculate_outlier_thresholds(df_plot[feature_x], grouped_method)['label']
            fig_grouped, df_group_outliers = plot_feature_scatter_grouped(
                df_result, feature_x, group_by, metadata_cols, method_label=method_label_g
            )
            st.plotly_chart(fig_grouped, use_container_width=True)

            n_g = int(df_result['is_outlier'].sum())
            pct_g = 100 * n_g / len(df_result)
            if n_g > 0:
                st.warning(f"**{n_g} outliers detected ({pct_g:.1f}%)** using {grouped_method} per group")
                with st.expander(f"View {n_g} grouped outliers"):
                    stat_cols = ['group_mean', 'group_std', 'group_rsd', 'group_lower',
                                 'group_upper', 'z_within_group', 'pct_diff_from_mean']
                    g_meta = [c for c in metadata_cols if c in df_result.columns]
                    show_cols = list(dict.fromkeys(g_meta + [feature_x] + stat_cols))
                    st.dataframe(df_group_outliers[show_cols], use_container_width=True)
            else:
                st.success("No outliers detected within any group.")

    st.write("---")
    st.write("### Multi-Feature Comparison")
    with st.expander("Compare Multiple Features"):
        plot_feature_comparison(df, compound_cols, metadata_cols, scaler_type=scaler_type)

    return df_selected


def plot_feature_scatter_grouped(df_result, feature_col, group_col, metadata_cols, method_label=None):
    groups = df_result[group_col].unique()
    group_positions = {g: i for i, g in enumerate(groups)}
    fig = go.Figure()
    color_map = {'Outlier': 'red', 'Inlier': '#4C72B0'}
    added_legend = set()

    for group in groups:
        subset = df_result[df_result[group_col] == group]
        x_pos = group_positions[group]
        g_lower = subset['group_lower'].iloc[0]
        g_upper = subset['group_upper'].iloc[0]

        if g_lower is not None:
            fig.add_shape(
                type="rect", x0=x_pos - 0.4, x1=x_pos + 0.4, y0=g_lower, y1=g_upper,
                fillcolor="rgba(100,180,100,0.12)", line=dict(color="green", width=1, dash="dash"),
                layer="below"
            )

        for _, row in subset.iterrows():
            label = 'Outlier' if row['is_outlier'] else 'Inlier'
            color = color_map[label]
            show_legend = label not in added_legend
            if show_legend:
                added_legend.add(label)
            hover_parts = [f"<b>{row[group_col]}</b>", f"{feature_col}: {row[feature_col]:.4f}"]
            for col in metadata_cols:
                if col in row and col != group_col:
                    hover_parts.append(f"{col}: {row[col]}")
            hover_parts += [f"group_mean: {row['group_mean']:.4f}",
                            f"group_std: {row['group_std']:.4f}",
                            f"z_within_group: {row['z_within_group']:.2f}"]
            fig.add_trace(go.Scatter(
                x=[x_pos + np.random.uniform(-0.2, 0.2)], y=[row[feature_col]],
                mode='markers',
                marker=dict(color=color, size=7, opacity=0.8,
                            line=dict(color='black', width=0.5) if row['is_outlier'] else dict(width=0)),
                name=label, legendgroup=label, showlegend=show_legend,
                hovertext="<br>".join(hover_parts), hoverinfo='text'
            ))

    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(group_positions.values()),
                   ticktext=list(group_positions.keys()), title=group_col),
        yaxis_title=feature_col,
        title=f"{feature_col} by {group_col}" + (f" — thresholds: {method_label}" if method_label else ""),
        height=550, template="plotly_white", legend_title="Status"
    )
    return fig, df_result[df_result['is_outlier']].copy()


def plot_feature_comparison(df, compound_cols, metadata_cols, scaler_type="Standard Scaler"):
    selected_features = st.multiselect(
        "Select features to compare", compound_cols,
        default=compound_cols[:3] if len(compound_cols) >= 3 else compound_cols
    )
    if not selected_features:
        st.info("Please select at least one feature to compare")
        return

    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.radio("Plot type", ["Violin", "Scatter"], horizontal=True)
    with col2:
        color_by = st.selectbox("Color by", [None] + metadata_cols, index=0, key="comparison_color")

    df_plot = df[selected_features].copy()
    scaler = get_scaler(scaler_type)
    if scaler:
        df_plot = pd.DataFrame(scaler.fit_transform(df_plot), columns=selected_features, index=df_plot.index)
    for col in metadata_cols:
        if col and col in df.columns:
            df_plot[col] = df[col]

    id_vars = [col for col in metadata_cols if col in df_plot.columns]
    df_melted = df_plot.melt(id_vars=id_vars, value_vars=selected_features,
                              var_name='Feature', value_name='Value')

    if plot_type == "Violin":
        fig = px.violin(df_melted, x='Feature', y='Value',
                        color=color_by if color_by else 'Feature',
                        box=True, points=False, color_discrete_sequence=categorical_colors,
                        title="Feature Comparison")
        fig.update_layout(height=500)
    else:
        df_melted['_index'] = df_melted.index
        fig = px.scatter(df_melted, x='_index', y='Value',
                         color=color_by if color_by else 'Feature',
                         facet_col='Feature', facet_col_wrap=min(3, len(selected_features)),
                         hover_data=id_vars, color_discrete_sequence=categorical_colors,
                         title="Feature Comparison")
        fig.update_layout(height=400 * ((len(selected_features) + 2) // 3))

    st.plotly_chart(fig, use_container_width=True)


def plot_boxplot(group_names, groups, feature_series):
    fig = go.Figure()
    for name, values in zip(group_names, groups):
        fig.add_trace(go.Box(
            y=values, name=name, boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(size=4, opacity=0.6), line=dict(width=1.5)
        ))
    fig.update_layout(
        width=800, height=500,
        title=dict(text=f"Distribution of {feature_series}", x=0.5, xanchor="center",
                   font=dict(size=22, color="black")),
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(tickfont=dict(size=18, color="black"), ticks="outside",
                   tickcolor="black", ticklen=6),
        yaxis=dict(tickfont=dict(size=18, color="black"),
                   title=dict(text="Intensity", font=dict(size=18, color="black")),
                   ticks="outside", tickcolor="black", ticklen=6),
        margin=dict(l=60, r=60, t=70, b=60),
    )
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, xref="paper", yref="paper",
                  line=dict(width=3, color="black"), layer="below")
    return fig


def plot_peak_plotly(peak_data, all_mz=True):
    RT = peak_data['RT_vals']
    mz = peak_data['mz']
    mass = peak_data['mass']
    peak_start = peak_data.get('peak_start', None)
    peak_end = peak_data.get('peak_end', None)
    peakRT = peak_data.get('peakRT', 0)
    calc_RT = peak_data.get('calc_RT', 0)
    compound = peak_data.get('Compound', 'Unknown')
    asym = peak_data.get('asymmetry', None)

    fig = go.Figure()
    mass_ = mass[0] if isinstance(mass, list) and len(mass) > 0 else mass
    fig.add_trace(go.Scatter(x=RT, y=mz[0], mode='lines',
                             name=f"m/z {mass_} (Target ion)", line_color='#da9100'))
    if isinstance(mass, list) and len(mass) > 1:
        if len(mz) > 1:
            fig.add_trace(go.Scatter(x=RT, y=mz[1], mode='lines',
                                     name=f"m/z {mass[1]} (Qualifier ion 1)", line_color='#00a693'))
        if len(mz) > 2:
            fig.add_trace(go.Scatter(x=RT, y=mz[2], mode='lines',
                                     name=f"m/z {mass[2]} (Qualifier ion 2)", line_color='#74afda'))
    if peakRT:
        fig.add_trace(go.Scatter(x=[peakRT, peakRT], y=[min(mz[0]), max(mz[0])], mode='lines',
                                 line=dict(color='red', dash='dot'), name="Expected RT"))
    if calc_RT:
        fig.add_trace(go.Scatter(x=[calc_RT, calc_RT], y=[min(mz[0]), max(mz[0])], mode='lines',
                                 line=dict(color='purple', dash='dash'), name="Calculated RT"))
    if peak_start is not None and peak_end is not None:
        mask = (RT >= peak_start[0]) & (RT <= peak_end[0])
        fig.add_trace(go.Scatter(
            x=np.concatenate([RT[mask], [peak_end[0], peak_start[0]]]),
            y=np.concatenate([mz[0][mask], [0, 0]]),
            fill='tozeroy', mode='none', fillcolor='rgba(0, 100, 255, 0.2)', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[peak_start[0], peak_end[0]], y=[mz[0][mask][0], mz[0][mask][-1]],
            mode='markers', marker=dict(color='red', size=10, symbol='diamond'),
            name='Peak start/end'
        ))

    fig.update_layout(
        title=(f"{compound} - Calculated RT: {calc_RT:.2f if calc_RT else 'N/A'} | "
               f"Expected RT {peakRT:.2f if peakRT else 'N/A'}, "
               f"Asymmetry: {asym:.2f if asym else 'N/A'}"),
        xaxis_title="Retention Time (RT)", yaxis_title="Intensity", showlegend=True,
    )
    return fig


def plot_peak_properties(compound_stats):
    df_plot = compound_stats.copy()
    df_plot['_index'] = df_plot.index

    if "peak_detect_sel" not in st.session_state:
        st.session_state.peak_detect_sel = []
    if "peak_shape_sel" not in st.session_state:
        st.session_state.peak_shape_sel = []

    st.write("### Compound Detectability vs Reproducibility")
    st.caption("Use box select or lasso select to investigate specific compounds")

    fig = px.scatter(df_plot, x='detection_frac', y='cv_area', size='mean_area',
                     hover_data=['Compound', 'mean_rt', 'mean_width'],
                     title='Compound Detectability vs Reproducibility (size=mean_area)',
                     color_discrete_sequence=['#636EFA'])
    fig.update_layout(xaxis_title='Detection', yaxis_title='Peak Area RSD',
                      autosize=True, height=650)

    events_1 = plotly_events(fig, select_event=True, click_event=False, hover_event=False,
                              override_height=650, override_width="100%",
                              key="peak_props_detectability")
    if events_1:
        st.session_state.peak_detect_sel = events_1

    if st.session_state.peak_detect_sel:
        selected_idx_1 = []
        for p in st.session_state.peak_detect_sel:
            mask = (df_plot['detection_frac'] == p['x']) & (df_plot['cv_area'] == p['y'])
            selected_idx_1.extend(df_plot[mask].index.tolist())
        if selected_idx_1:
            st.write("#### Selected Compounds (Detectability Plot)")
            cols_to_show = [c for c in ['Compound', 'detection_frac', 'cv_area', 'mean_area',
                                         'rt_sd', 'mean_rt', 'mean_width'] if c in df_plot.columns]
            st.dataframe(df_plot.loc[selected_idx_1, cols_to_show], use_container_width=True)

    st.write("### Compound Peak Shape Quality")
    st.caption("Use box select or lasso select to investigate specific compounds")

    fig_shape = px.scatter(df_plot, x='mean_width', y='sd_width', size='mean_area',
                           hover_data=['Compound', 'rt_sd', 'sd_asym', 'detection_frac'],
                           title='Compound Peak Width mean vs sd (size=mean_area)')
    fig_shape.update_layout(xaxis_title='Mean Peak Width (secs)', yaxis_title='SD Peak Width (secs)',
                             height=650)

    events_2 = plotly_events(fig_shape, select_event=True, click_event=False, hover_event=False,
                              override_height=650, override_width="100%", key="peak_props_shape")
    if events_2:
        st.session_state.peak_shape_sel = events_2

    if st.session_state.peak_shape_sel:
        selected_idx_2 = []
        for p in st.session_state.peak_shape_sel:
            mask = (np.isclose(df_plot['mean_width'], p['x'], rtol=1e-9) &
                    np.isclose(df_plot['sd_width'], p['y'], rtol=1e-9))
            selected_idx_2.extend(df_plot[mask].index.tolist())
        if selected_idx_2:
            st.write("#### Selected Compounds (Peak Shape Plot)")
            cols_to_show = [c for c in ['Compound', 'mean_width', 'sd_width', 'mean_area',
                                         'mean_asym', 'sd_asym', 'rt_sd', 'detection_frac']
                            if c in df_plot.columns]
            st.dataframe(df_plot.loc[selected_idx_2, cols_to_show], use_container_width=True)

    st.write("### Average Expression")
    fig_exp = px.scatter(df_plot, x='detection_frac', y='av_expr', size='mean_area',
                         hover_data=['Compound', 'rt_sd', 'sd_asym', 'detection_frac'],
                         title='Average Expression against detection rate')
    fig_exp.update_layout(xaxis_title='Detection rate', yaxis_title='Average Expression', height=650)
    st.plotly_chart(fig_exp)


def plot_compound_peak_data(df, feature):
    data = df[df['Compound'] == feature]
    signals = np.vstack(data[data['detected']]['signal'].tolist())
    st.plotly_chart(plot_signal_overlay(signals))

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Peak Area", "RT", "Peak Width", "Tailing"))
    fig.add_trace(go.Histogram(x=data['peak_area']), row=1, col=1)
    fig.add_trace(go.Histogram(x=data['calc_RT']), row=1, col=2)
    fig.add_trace(go.Histogram(x=data['peak_width']), row=2, col=1)
    fig.add_trace(go.Histogram(x=data['asymmetry']), row=2, col=2)
    fig.update_layout(showlegend=False,
                      title_text=f"Distributions of Peak properties for {feature}", height=700)
    st.plotly_chart(fig, use_container_width=True)

    df_filtered = data[(data['peak_area'] > 0) & data['peak_area'].notna() &
                       np.isfinite(data['peak_area'])]
    if len(df_filtered) > 0:
        mass_spec_arrays = np.vstack(df_filtered['mass_spec'].tolist())
        mass_spec = mass_spec_arrays.mean(axis=0)
    else:
        mass_spec = None

    x, y, hover_text = [], [], []
    for m, i in enumerate(mass_spec):
        if np.isfinite(i) and i > 0:
            x += [m, m, None]
            y += [0, i, None]
            hover_text += [f"m = {m}<br>i = {i}", f"m = {m}<br>i = {i}", None]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="black", width=2),
                              hoverinfo="text", text=hover_text, showlegend=False))
    fig2.update_layout(title_text=f"Mean Mass Spectrum for {feature}")
    st.plotly_chart(fig2)

    msp_data = parse_mass_spec(feature, 1, mass_spec)
    text_content = ""
    for key, value in msp_data.items():
        text_content += (value + "\n") if key == "mass_spec" else f"{key}: {value}\n"

    st.download_button(label="Download MSP File", data=text_content,
                       file_name="msp.txt", mime="text/plain")
    st.write("## Feature Dataframe")
    cols = [x for x in data.columns if x not in ['mass_spec', 'signal']]
    st.dataframe(data[cols])


def plot_signal_overlay(signals):
    signals = np.array(signals)
    mean_signal = np.mean(signals, axis=0)
    x = np.arange(len(mean_signal))
    fig = go.Figure()
    for signal in signals:
        fig.add_trace(go.Scatter(x=x, y=signal, mode='lines',
                                 line=dict(color='rgba(150, 150, 150, 0.15)', width=1),
                                 showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=mean_signal, mode='lines',
                             line=dict(color='purple', width=3), name='Mean Signal'))
    fig.update_layout(title='Signal Overlay', xaxis_title='Time / Index',
                      yaxis_title='I', template='plotly_white', height=700)
    return fig


# ── New plotting functions ────────────────────────────────────────────────────

def plot_d_ratio_scatter(d_ratio_df, d_thresh=0.5, rsd_thresh=30):
    """
    Scatter: D-ratio (x) vs Median RSD (y). Bottom-left = best quality.
    Colour gradient green→red. Threshold lines. Outlier labels.
    """
    df = d_ratio_df.dropna(subset=['d_ratio', 'median_rsd']).copy()

    def quality_color(row):
        if row['d_ratio'] <= d_thresh and row['median_rsd'] <= rsd_thresh:
            return 'Good'
        elif row['d_ratio'] > d_thresh and row['median_rsd'] > rsd_thresh:
            return 'Poor'
        else:
            return 'Borderline'

    df['quality'] = df.apply(quality_color, axis=1)

    color_map = {'Good': '#2ca02c', 'Borderline': '#ff7f0e', 'Poor': '#d62728'}

    fig = px.scatter(
        df, x='d_ratio', y='median_rsd', color='quality',
        color_discrete_map=color_map,
        hover_data=['compound'],
        labels={'d_ratio': 'D-ratio (within / total SD)', 'median_rsd': 'Median RSD (%)'},
        title='Compound Quality: D-ratio vs Median RSD<br>(bottom-left = best)'
    )

    fig.add_vline(x=d_thresh, line_dash='dash', line_color='gray',
                  annotation_text=f'D-ratio = {d_thresh}', annotation_position='top right')
    fig.add_hline(y=rsd_thresh, line_dash='dot', line_color='gray',
                  annotation_text=f'RSD = {rsd_thresh}%', annotation_position='top left')

    outliers = df[(df['d_ratio'] > d_thresh) | (df['median_rsd'] > rsd_thresh)]
    for _, row in outliers.iterrows():
        fig.add_annotation(x=row['d_ratio'], y=row['median_rsd'], text=row['compound'],
                           showarrow=False, font=dict(size=8), xanchor='left',
                           yanchor='bottom', xshift=4)

    fig.update_layout(height=600, template='plotly_white')
    return fig


def plot_volcano(fc_df, fc_thresh=1.0, p_thresh=0.05):
    """Volcano plot: log FC (x) vs -log10(p) (y), coloured by significance."""
    df = fc_df.dropna(subset=['log_fc', 'neg_log10_p']).copy()
    df['significance'] = 'Not significant'
    df.loc[(df['log_fc'] > fc_thresh) & (df['pval'] < p_thresh), 'significance'] = 'Up'
    df.loc[(df['log_fc'] < -fc_thresh) & (df['pval'] < p_thresh), 'significance'] = 'Down'

    color_map = {'Not significant': '#aec7e8', 'Up': '#d62728', 'Down': '#1f77b4'}

    fig = px.scatter(
        df, x='log_fc', y='neg_log10_p', color='significance',
        color_discrete_map=color_map, hover_data=['compound'],
        labels={'log_fc': 'Log Fold Change (group1 − group2)', 'neg_log10_p': '−log₁₀(p-value)'},
        title='Volcano Plot'
    )

    fig.add_vline(x=fc_thresh, line_dash='dash', line_color='gray')
    fig.add_vline(x=-fc_thresh, line_dash='dash', line_color='gray')
    fig.add_hline(y=-np.log10(p_thresh), line_dash='dash', line_color='gray',
                  annotation_text=f'p = {p_thresh}', annotation_position='top left')

    sig_up = df[df['significance'] == 'Up']
    sig_down = df[df['significance'] == 'Down']
    for _, row in pd.concat([sig_up, sig_down]).iterrows():
        fig.add_annotation(x=row['log_fc'], y=row['neg_log10_p'], text=row['compound'],
                           showarrow=False, font=dict(size=8), xanchor='left',
                           yanchor='bottom', xshift=4)

    fig.update_layout(height=550, template='plotly_white')
    return fig


def plot_detection_heatmap(detection_df):
    """Heatmap of detection rates: compounds (rows) × groups (cols)."""
    group_cols = [c for c in detection_df.columns if c != 'Compound']
    matrix = detection_df.set_index('Compound')[group_cols]

    fig = px.imshow(
        matrix,
        color_continuous_scale='RdYlGn',
        zmin=0, zmax=100,
        aspect='auto',
        labels=dict(color='Detection %'),
        title='Detection Rate (%) per Compound per Group'
    )
    fig.update_layout(height=max(400, len(matrix) * 12), template='plotly_white')
    return fig


def plot_compound_across_groups(df, compound, group_col, metadata_cols=None):
    """
    Plot a compound's values across group levels.
    Individual points per group, population IQR band + median as reference.
    """
    df_plot = df.copy()
    try:
        groups = sorted(df_plot[group_col].dropna().unique())
    except TypeError:
        groups = list(df_plot[group_col].dropna().unique())

    group_positions = {g: i for i, g in enumerate(groups)}

    pop_median = df_plot[compound].median()
    pop_q1 = df_plot[compound].quantile(0.25)
    pop_q3 = df_plot[compound].quantile(0.75)

    fig = go.Figure()

    # Population IQR band
    fig.add_hrect(y0=pop_q1, y1=pop_q3, fillcolor="rgba(200,200,200,0.3)", line_width=0,
                  annotation_text="Pop IQR", annotation_position="top left")
    fig.add_hline(y=pop_median, line_dash="dash", line_color="gray",
                  annotation_text="Pop median", annotation_position="right")

    rng = np.random.default_rng(42)

    for i, group in enumerate(groups):
        subset = df_plot[df_plot[group_col] == group]
        color = categorical_colors[i % len(categorical_colors)]
        x_vals = group_positions[group] + rng.uniform(-0.2, 0.2, size=len(subset))

        hover_parts = []
        for idx, row in subset.iterrows():
            parts = [f"<b>{group}</b>", f"{compound}: {row[compound]:.4f}"]
            if metadata_cols:
                for col in metadata_cols:
                    if col in row.index and col != group_col:
                        parts.append(f"{col}: {row[col]}")
            hover_parts.append("<br>".join(parts))

        fig.add_trace(go.Scatter(
            x=x_vals, y=subset[compound].values,
            mode='markers', name=str(group),
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(color='white', width=0.5)),
            hovertext=hover_parts, hoverinfo='text'
        ))

        # Group median lollipop
        g_median = subset[compound].median()
        fig.add_trace(go.Scatter(
            x=[group_positions[group], group_positions[group]], y=[pop_median, g_median],
            mode='lines', line=dict(color=color, width=2), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=[group_positions[group]], y=[g_median], mode='markers',
            marker=dict(color=color, size=14, symbol='circle',
                        line=dict(color='white', width=2)),
            showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(group_positions.values()),
                   ticktext=[str(g) for g in groups], title=group_col),
        yaxis_title=compound,
        title=f"{compound} across {group_col}",
        height=500, template='plotly_white'
    )
    return fig
