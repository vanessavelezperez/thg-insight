import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile

from utils import setup_page
from analysis import (variance, cv, calculate_detection_rate, calculate_d_ratio, calculate_log_fc,
                      compute_correlation_matrix, calculate_cv_by_group)
from plotting import (draw_pyvis_network, plot_d_ratio_scatter,
                      plot_volcano, plot_detection_heatmap)

st.set_page_config(page_title="Feature Filtering", layout="wide")
st.title("Feature Filtering")

df_filtered, compound_cols, metadata_cols, scaler_type = setup_page()
X = df_filtered[compound_cols]

section = st.radio(
    "Section",
    ["Variance & CV", "Correlation", "Detection Rate", "D-ratio", "Group Comparison"],
    horizontal=True
)

# ── Variance & CV ─────────────────────────────────────────────────────────────
if section == "Variance & CV":
    st.header("Variance & CV")
    st.caption("Use Robust or Pareto scaler to better preserve relative variance structure.")

    metric_choice = st.radio("Metric", ["Variance", "CV (%)"], horizontal=True)

    if metric_choice == "Variance":
        metric_df = variance(X, scaler_type)
        metric_col = "variance"
        low_thresh, high_thresh = 0.1, 0.5
        level_labels = {"low": "Low (<0.1)", "mid": "Medium (0.1–0.5)", "high": "High (>0.5)"}
        y_label = "Variance (scaled)"
    else:
        metric_df = cv(X, scaler_type)
        metric_col = "cv"
        low_thresh, high_thresh = 20, 50
        level_labels = {"low": "Low (<20%)", "mid": "Medium (20–50%)", "high": "High (>50%)"}
        y_label = "CV (%)"

    metric_df["level"] = metric_df[metric_col].apply(
        lambda x: level_labels["low"] if x < low_thresh
        else level_labels["mid"] if x < high_thresh
        else level_labels["high"]
    )

    ptile5 = np.percentile(metric_df[metric_col].dropna(), 1)
    ptile95 = np.percentile(metric_df[metric_col].dropna(), 99)

    fig = px.scatter(
        metric_df, x=range(len(metric_df)), y=metric_col,
        color="level", hover_data=["feature"],
        title=f"Feature {metric_choice} Analysis",
        labels={"x": "Feature Index", metric_col: y_label},
        color_discrete_map={
            level_labels["low"]: "red",
            level_labels["mid"]: "orange",
            level_labels["high"]: "green",
        }
    )
    fig.add_hline(y=ptile5, line_dash="dash", line_color="blue",
                  annotation_text=f"2.5th percentile: {round(ptile5, 2)}")
    fig.add_hline(y=ptile95, line_dash="dash", line_color="purple",
                  annotation_text=f"97.5th percentile: {round(ptile95, 2)}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metric_df, use_container_width=True)

# ── Correlation ───────────────────────────────────────────────────────────────
elif section == "Correlation":
    st.header("Compound Correlation Network")
    st.caption("Find highly correlated features that may be redundant.")

    corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    corr_threshold = st.slider("Correlation threshold", 0.7, 1.0, 0.95, 0.01)
    corr_matrix = compute_correlation_matrix(df_filtered, tuple(compound_cols), corr_method)

    st.markdown(f"Showing edges with |correlation| ≥ {corr_threshold:.2f}")
    net = draw_pyvis_network(corr_matrix, threshold=corr_threshold)
    path = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
    net.save_graph(path)
    st.components.v1.html(open(path, 'r', encoding='utf-8').read(), height=650)

    high_corr_pairs = (
        corr_matrix.abs().unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
    )
    high_corr_pairs = high_corr_pairs[
        (high_corr_pairs < 1) & (high_corr_pairs > corr_threshold)
    ]
    high_corr_df = high_corr_pairs.reset_index()
    high_corr_df.columns = ['feature_1', 'feature_2', 'correlation']
    st.write(f"**{len(high_corr_df)} high-correlation pairs**")
    st.dataframe(high_corr_df, use_container_width=True)

# ── Detection Rate ────────────────────────────────────────────────────────────
elif section == "Detection Rate":
    st.header("Detection Rate")
    st.caption("Percentage of samples where each compound was detected (value > 0).")

    group_for_det = st.selectbox(
        "Break down by group column (optional)", [None] + metadata_cols, key="det_group"
    )

    det_df = calculate_detection_rate(df_filtered, compound_cols, group_col=group_for_det)

    overall_mean = det_df['Overall'].mean()
    fully_detected = (det_df['Overall'] == 100).sum()
    never_detected = (det_df['Overall'] == 0).sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean detection rate", f"{overall_mean:.1f}%")
    with col2:
        st.metric("Detected in all samples", int(fully_detected))
    with col3:
        st.metric("Never detected", int(never_detected))

    det_thresh = st.slider("Highlight compounds below detection threshold (%)", 0, 100, 50, 5)
    below_thresh = det_df[det_df['Overall'] < det_thresh]
    if len(below_thresh) > 0:
        st.warning(f"{len(below_thresh)} compounds detected in fewer than {det_thresh}% of samples overall.")

    if group_for_det:
        fig_heatmap = plot_detection_heatmap(det_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    fig_bar = px.histogram(
        det_df, x='Overall', nbins=20,
        title='Distribution of overall detection rates across compounds',
        labels={'Overall': 'Detection Rate (%)', 'count': 'Number of compounds'}
    )
    fig_bar.add_vline(x=det_thresh, line_dash='dash', line_color='red',
                      annotation_text=f'Threshold: {det_thresh}%')
    fig_bar.update_layout(height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("**Full detection rate table**")
    st.dataframe(
        det_df.sort_values('Overall', ascending=False).style.background_gradient(
            cmap='RdYlGn', subset=[c for c in det_df.columns if c != 'Compound']
        ),
        use_container_width=True
    )

# ── D-ratio ───────────────────────────────────────────────────────────────────
elif section == "D-ratio":
    st.header("D-ratio")
    st.caption(
        "D-ratio = pooled within-group SD / total SD. "
        "Low D-ratio means the compound varies more between groups than within — good for discriminant analysis."
    )

    group_for_d = st.selectbox("Group column for D-ratio", metadata_cols, key="dratio_group")

    if group_for_d:
        d_thresh = st.slider("D-ratio threshold", 0.0, 1.0, 0.5, 0.05)
        rsd_thresh = st.slider("Median RSD threshold (%)", 0, 100, 30, 5)

        d_df = calculate_d_ratio(df_filtered, compound_cols, group_for_d)

        good = ((d_df['d_ratio'] <= d_thresh) & (d_df['median_rsd'] <= rsd_thresh)).sum()
        poor = ((d_df['d_ratio'] > d_thresh) | (d_df['median_rsd'] > rsd_thresh)).sum()
        good_compounds = d_df.loc[
            (d_df['d_ratio'] <= d_thresh) & (d_df['median_rsd'] <= rsd_thresh), 'compound'
        ].tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Good quality (bottom-left)", int(good))
        with col2:
            st.metric("Flagged compounds", int(poor))

        fig_d = plot_d_ratio_scatter(d_df, d_thresh=d_thresh, rsd_thresh=rsd_thresh)
        st.plotly_chart(fig_d, use_container_width=True)

        with st.expander("View D-ratio table"):
            st.dataframe(
                d_df.sort_values('d_ratio').style.background_gradient(
                    cmap='RdYlGn_r', subset=['d_ratio']
                ),
                use_container_width=True
            )

        st.divider()
        st.subheader(f"RSD by {group_for_d} (using good-quality compounds)")
        st.caption(
            f"For each {group_for_d} with ≥2 samples, compute RSD (%) per compound restricted to the "
            "good-quality compound set above, then average across compounds."
        )

        rsd_by_group_thresh = st.slider(f"Flag {group_for_d} with mean RSD above (%)", 0, 200, 30, 5)
        if not good_compounds:
            st.warning("No compounds pass the D-ratio/RSD thresholds above — nothing to check.")
        else:
            cv_long = calculate_cv_by_group(df_filtered, good_compounds, group_for_d)

            if cv_long.empty:
                st.warning(f"No {group_for_d} has ≥2 non-empty samples — RSD cannot be computed.")
            else:
                group_rsd = cv_long.groupby(group_for_d)['CV %'].agg(
                    mean_rsd='mean', n_compounds='count'
                ).reset_index()
                group_rsd['flagged'] = group_rsd['mean_rsd'] > rsd_by_group_thresh

                n_flagged = int(group_rsd['flagged'].sum())
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{group_for_d} evaluated", len(group_rsd))
                with col2:
                    st.metric(f"Flagged (RSD > {rsd_by_group_thresh}%)", n_flagged)

                if n_flagged > 0:
                    st.warning(
                        f"{n_flagged} {group_for_d}(s) show mean RSD above {rsd_by_group_thresh}% "
                        f"across the {cv_long['Compound'].nunique()} good-quality compounds."
                    )

                rsd = group_rsd['mean_rsd']
                for col, (label, val) in zip(st.columns(5), [
                    ("Mean RSD", rsd.mean()), ("Median RSD", rsd.median()),
                    ("Q1", rsd.quantile(0.25)), ("Q3", rsd.quantile(0.75)), ("Max", rsd.max()),
                ]):
                    col.metric(label, f"{val:.1f}%")

                fig_rsd_hist = px.histogram(
                    group_rsd, x='mean_rsd', nbins=30,
                    color_discrete_sequence=['#1f77b4'],
                    labels={'mean_rsd': 'Mean RSD (%)'},
                    title=f"Distribution of mean RSD across {group_for_d}",
                )
                fig_rsd_hist.update_layout(bargap=0.05, height=300, yaxis_title=f"# {group_for_d}")
                fig_rsd_hist.add_vline(x=rsd.median(), line_color='black',
                                       annotation_text=f"median {rsd.median():.1f}%")
                fig_rsd_hist.add_vline(x=rsd_by_group_thresh, line_dash='dash', line_color='gray',
                                       annotation_text=f"{rsd_by_group_thresh}%",
                                       annotation_position='bottom right')
                st.plotly_chart(fig_rsd_hist, use_container_width=True)

                fig_group_rsd = px.bar(
                    group_rsd.sort_values('mean_rsd', ascending=False),
                    x=group_for_d, y='mean_rsd', color='flagged',
                    color_discrete_map={True: '#d62728', False: '#1f77b4'},
                    labels={'mean_rsd': 'Mean RSD (%)'},
                    title=f"RSD by {group_for_d} across {cv_long['Compound'].nunique()} good-quality compounds"
                )
                fig_group_rsd.add_hline(y=rsd_by_group_thresh, line_dash='dash', line_color='gray',
                                        annotation_text=f'{rsd_by_group_thresh}%')
                st.plotly_chart(fig_group_rsd, use_container_width=True)

                with st.expander("View RSD table"):
                    st.dataframe(
                        group_rsd.sort_values('mean_rsd', ascending=False),
                        use_container_width=True
                    )

# ── Group Comparison ──────────────────────────────────────────────────────────
elif section == "Group Comparison":
    st.header("Group Comparison")
    st.caption(
        "Log fold change between two groups. "
        "Since data is log-transformed, FC = difference in means. "
        "P-values from Mann–Whitney U test."
    )

    group_col_fc = st.selectbox("Metadata column", metadata_cols, key="fc_group_col")

    if group_col_fc:
        group_options = sorted(df_filtered[group_col_fc].dropna().unique().tolist(), key=str)

        if len(group_options) < 2:
            st.warning("Need at least 2 groups in the selected column.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                group1 = st.selectbox("Group 1 (numerator)", group_options, index=0)
            with col2:
                group2 = st.selectbox("Group 2 (denominator)", group_options,
                                      index=1 if len(group_options) > 1 else 0)

            if group1 == group2:
                st.warning("Please select two different groups.")
            else:
                fc_thresh = st.slider("Log FC threshold", 0.0, 5.0, 1.0, 0.1)
                p_thresh = st.select_slider("P-value threshold", options=[0.1, 0.05, 0.01, 0.001],
                                            value=0.05)

                fc_df = calculate_log_fc(df_filtered, compound_cols, group_col_fc, group1, group2)

                n_up = ((fc_df['log_fc'] > fc_thresh) & (fc_df['pval'] < p_thresh)).sum()
                n_down = ((fc_df['log_fc'] < -fc_thresh) & (fc_df['pval'] < p_thresh)).sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Up in {group1}", int(n_up))
                with col2:
                    st.metric(f"Down in {group1}", int(n_down))
                with col3:
                    st.metric("Not significant", int(len(fc_df) - n_up - n_down))

                fig_v = plot_volcano(fc_df, fc_thresh=fc_thresh, p_thresh=p_thresh)
                st.plotly_chart(fig_v, use_container_width=True)

                with st.expander("View fold change table"):
                    display_df = fc_df.sort_values('neg_log10_p', ascending=False)
                    st.dataframe(display_df.style.background_gradient(
                        cmap='RdBu_r', subset=['log_fc']
                    ), use_container_width=True)
