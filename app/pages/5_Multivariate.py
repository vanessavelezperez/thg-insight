import streamlit as st
import pandas as pd
import numpy as np
from utils import setup_page
from analysis import run_pca, run_tsne, run_umap, run_kmeans, run_hdbscan
from plotting import plot_scatter

st.set_page_config(page_title="Multivariate", layout="wide")
st.title("Multivariate Analysis")
st.caption("Unsupervised dimensionality reduction and clustering to explore natural structure in your data.")

df_filtered, compound_cols, metadata_cols, scaler_type = setup_page()
X = df_filtered[compound_cols]

selected_features = st.multiselect(
    "Filter compounds to include (leave empty for all)",
    options=compound_cols, key="method_features"
)
X_sel = X[selected_features] if selected_features else X

analysis_section = st.radio(
    "Analysis", ["Dimensionality Reduction", "Clustering"], horizontal=True
)

# ── Dimensionality Reduction ────────────────────────────────────────────────────
if analysis_section == "Dimensionality Reduction":
    method = st.radio("Method", ["PCA", "t-SNE", "UMAP"], horizontal=True)

    if method == "PCA":
        max_components = min(X_sel.shape[0], X_sel.shape[1])
        n_components = st.slider(
            "Number of principal components", min_value=2,
            max_value=max_components, value=min(3, max_components)
        )
        pca, components = run_pca(X_sel, n_components=n_components, scaler_type=scaler_type)
        plot_scatter(df_filtered, components, metadata_cols, method="PCA",
                     explained_var=pca.explained_variance_ratio_)

        st.subheader("Explained Variance")
        for i, var in enumerate(pca.explained_variance_ratio_):
            st.write(f"PC{i+1}: {var:.2%}")

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings, index=X_sel.columns,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        st.subheader("PCA Loadings")
        pd.set_option("styler.render.max_elements", 300000)
        st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=None),
                     use_container_width=True)

    elif method == "t-SNE":
        st.caption("Non-linear dimensionality reduction. Good for revealing local cluster structure.")
        tsne, components = run_tsne(X_sel, n_components=2, scaler_type=scaler_type)
        plot_scatter(df_filtered, components, metadata_cols, method="t-SNE")

    elif method == "UMAP":
        st.caption("Non-linear dimensionality reduction that preserves both local and global structure.")
        with st.expander("UMAP Parameters"):
            n_neighbors = st.slider(
                "n_neighbors (local vs global structure)", min_value=5, max_value=50, value=15,
                help="Lower values focus on local structure, higher values on global structure"
            )
            min_dist = st.slider(
                "min_dist (cluster tightness)", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                help="Lower values create tighter clusters, higher values spread points out"
            )
        umap_model, components = run_umap(
            X_sel, n_components=2, scaler_type=scaler_type,
            n_neighbors=n_neighbors, min_dist=min_dist
        )
        plot_scatter(df_filtered, components, metadata_cols, method="UMAP")

# ── Clustering ───────────────────────────────────────────────────────────────────
else:
    st.caption("Group samples into clusters based on compound similarity.")
    clustering_type = st.radio(
        "Clustering type", ["K-means (semi-supervised)", "HDBSCAN (unsupervised)"], horizontal=True
    )

    if clustering_type == "K-means (semi-supervised)":
        n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
        model, cluster_labels, silhouette, centroids_df = run_kmeans(
            X_sel, n_clusters=n_clusters, scaler_type=scaler_type
        )
        st.metric("Silhouette Score", f"{silhouette:.3f}" if not np.isnan(silhouette) else "N/A")

    else:
        st.caption("Cluster count is determined automatically from the data's density structure.")
        max_cluster_size = max(3, len(X_sel) // 2)
        with st.expander("HDBSCAN Parameters"):
            min_cluster_size = st.slider(
                "min_cluster_size (smallest grouping considered a real cluster)",
                min_value=2, max_value=max_cluster_size, value=min(5, max_cluster_size),
                help="Higher values require larger, more conservative clusters"
            )
            min_samples = st.slider(
                "min_samples (higher = more conservative, more noise)",
                min_value=1, max_value=min_cluster_size, value=min(5, min_cluster_size)
            )
        model, cluster_labels, silhouette, centroids_df = run_hdbscan(
            X_sel, min_cluster_size=min_cluster_size, min_samples=min_samples, scaler_type=scaler_type
        )

        n_noise = int((cluster_labels == -1).sum())
        n_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters found", n_found)
        col2.metric("Noise points", f"{n_noise} ({100 * n_noise / len(cluster_labels):.1f}%)")
        col3.metric("Silhouette Score (excl. noise)", f"{silhouette:.3f}" if not np.isnan(silhouette) else "N/A")

    cluster_label_strs = ["Noise" if c == -1 else f"Cluster {c}" for c in cluster_labels]

    pca, components = run_pca(X_sel, n_components=3, scaler_type=scaler_type)
    df_clustered = df_filtered.copy()
    df_clustered["Cluster"] = cluster_label_strs
    plot_scatter(df_clustered, components, metadata_cols + ["Cluster"], method="Clustering",
                 explained_var=pca.explained_variance_ratio_)

    st.subheader("Cluster Sizes")
    st.dataframe(pd.Series(cluster_label_strs).value_counts().rename("N Samples"),
                 use_container_width=True)

    if not centroids_df.empty:
        st.subheader("Cluster Centroids (scaled feature means)")
        pd.set_option("styler.render.max_elements", 300000)
        st.dataframe(centroids_df.style.background_gradient(cmap='coolwarm', axis=None),
                     use_container_width=True)
