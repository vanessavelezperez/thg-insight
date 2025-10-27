# main.py
import streamlit as st
import pandas as pd
from analysis import run_pca, run_tsne, run_plsda, run_oplsda
from plotting import plot_scatter, draw_pyvis_network
from utils import load_data, filter_dataframe
import numpy as np
import io

# --- Page setup ---
st.set_page_config(page_title="Results", layout="wide")
st.title("🔬 Analytics Dashboard")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your Excel/csv file", type=["xlsx", "xls", "csv"])
if not uploaded_file:
    st.info("Please upload an Excel file to get started.")
    st.stop()

# Load Excel and choose sheet
if uploaded_file.name.endswith((".xlsx", ".xls")):
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet_name = st.selectbox("Select worksheet", xls.sheet_names)
    df = load_data(uploaded_file, sheet_name, type='excel')
else:
    sheet_name = None
    df = load_data(uploaded_file, sheet_name=None, type='csv')

# --- Metadata & compounds ---
all_columns = df.columns.tolist()
metadata_cols = st.sidebar.multiselect("Select metadata columns", all_columns)
compound_cols = [col for col in all_columns if col not in metadata_cols]

if not metadata_cols or not compound_cols:
    st.sidebar.warning("Please select at least one metadata column and one compound column.")
    st.stop()

# --- Navigation ---
page = st.sidebar.radio("Choose a page", ["PCA", "t-SNE", "PLS-DA","Compound Correlation Network", "Feature Exploration", "Univariate Tests"])

# --- Common filters ---
df_filtered = filter_dataframe(df, metadata_cols)
df_filtered = df_filtered.reset_index(drop=True)

scale = st.sidebar.checkbox("Standardise compound data", value=True)
X = df_filtered[compound_cols]
if X.empty:
    st.warning("No data available after filtering.")
    st.stop()

# --- Analysis Pages ---
if page == "PCA":
    st.header("📊 PCA Visualisation")
    st.markdown("#### Optional: Filter features (compounds) to include")
    selected_features = st.multiselect(
        "Select compounds (leave empty to include all)",
        options=compound_cols
    )
    X_pca = np.log1p(X)
    X_pca = X_pca[selected_features] if selected_features else X_pca
    pca, components = run_pca(X_pca, n_components=3, scale=scale)
    plot_scatter(df_filtered, components, metadata_cols, method="PCA", explained_var=pca.explained_variance_ratio_)

    # --- Explained variance ---
    st.markdown("### Explained Variance")
    for i, var in enumerate(pca.explained_variance_ratio_):
        st.write(f"PC{i+1}: {var:.2%}")

    # --- Loadings table ---
    st.markdown("### PCA Loadings")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X_pca.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    pd.set_option("styler.render.max_elements", 300000)
    st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=None))

elif page == "t-SNE":
    st.header("🔎 t-SNE Visualisation")
    tsne, components = run_tsne(X, n_components=2, scale=scale)
    plot_scatter(df_filtered, components, metadata_cols, method="t-SNE")

elif page == "PLS-DA":
    st.header("🧪 PLS-DA Visualisation")
    
    # Choose the target variable (class)
    target_col = st.selectbox("Select target (categorical) column", metadata_cols)
    y = df_filtered[target_col].astype('category').cat.codes

    colour_by = st.selectbox(f"Colour by metadata column", metadata_cols)


    # Optional: Filter features
    st.markdown("#### Optional: Filter compounds to include")
    selected_features = st.multiselect(
        "Select compounds (leave empty to include all)",
        options=compound_cols
    )
    X = np.log1p(X)
    X_opls = X[selected_features] if selected_features else X

    n_components=2
    pls, scores, loadings = run_plsda(X_opls, y, n_components=n_components, scale=scale)

    plot_scatter(df_filtered, scores, metadata_cols, method='PLS-DA')

    # Loadings table
    st.markdown("### PLS-DA Loadings")
    loadings_df = pd.DataFrame(loadings[:, :n_components], index=X_opls.columns,
                               columns=[f"Comp{i+1}" for i in range(n_components)])
    st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=None))

elif page == "Compound Correlation Network":
    import tempfile
    st.header("Compound Correlation Network")
    corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    corr_threshold = st.slider("Correlation threshold", 0.7, 1.0, 0.95, 0.01)
    corr_matrix = df_filtered[compound_cols].corr(method=corr_method)
    st.markdown(f"Showing edges with abs(correlation) >= {corr_threshold:.2f}")
    net = draw_pyvis_network(corr_matrix, threshold=corr_threshold)
    path = tempfile.NamedTemporaryFile(suffix=".html", delete=False).name
    net.save_graph(path)
    st.components.v1.html(open(path, 'r', encoding='utf-8').read(), height=650)

elif page == "Feature Exploration":
    st.header("Feature Exploration")

    from plotting import plot_feature_exploration
    plot_feature_exploration(df_filtered, compound_cols, metadata_cols,scale=scale)

elif page == "Univariate Tests":
    st.header("📈 Univariate Feature Tests")

    # --- Select target variable (binary) ---
    target_col = st.selectbox("Select target feature", metadata_cols)
    y = df_filtered[target_col].astype('category')

    # --- Select control group for Glass's Δ ---
    control_group = st.radio(
        "Select control group for Glass's Δ",
        options=list(y.cat.categories),
        index=0  # default to first category
    )

    # --- Select one compound to explore ---
    selected_feature = st.selectbox("Select compound", compound_cols)

    if selected_feature:
        from analysis import run_univariate_tests
        results, fig, explanations = run_univariate_tests(
            df_filtered[selected_feature], y, control_group=control_group
        )

        # --- Statistical Test Explanations (same as before) ---
        st.markdown("### 🧾 Statistical Test Explanations")
        st.markdown(
            """
            **Normality Test (Shapiro-Wilk)**
            - Tests whether a group's distribution is approximately normal.
            - p > 0.05 → parametric tests are valid
            - p ≤ 0.05 → use non-parametric tests

            **t-test (Welch)**
            - Compares means of two independent groups.
            - Welch version does not assume equal variances.
            - p < 0.05 → significant difference in means

            **Mann–Whitney U test**
            - Non-parametric test comparing distributions/ranks.
            - p < 0.05 → distributions differ significantly

            **Effect Sizes**
            - Cohen's d → standardized mean difference
            - Glass's Δ → uses SD of the selected control group
            - Rank-biserial correlation → non-parametric effect size
            """
        )

        # --- Results Table, Summary, and Plot ---
        st.markdown("### Test Results")
        st.table(results)

        st.markdown("### Quick Summary")
        st.write("**Normality:**")
        for line in explanations["normality"]:
            st.write("-", line)
        st.write("**Test Choice:**", explanations["test_choice"])
        st.write("**Effect Sizes:**", explanations["effect_sizes"])
        st.write("**P-value Interpretation:**", explanations["pval_interpretation"])

        st.markdown("### Distribution Plot")
        st.pyplot(fig)