import streamlit as st
import pandas as pd
from utils import setup_page
from analysis import run_plsda, run_univariate_tests
from plotting import plot_scatter

st.set_page_config(page_title="Supervised", layout="wide")
st.title("Supervised Analysis")

df_filtered, compound_cols, metadata_cols, scaler_type = setup_page()
X = df_filtered[compound_cols]

section = st.radio("Method", ["PLS-DA", "Univariate Tests"], horizontal=True)

# ── PLS-DA ────────────────────────────────────────────────────────────────────
if section == "PLS-DA":
    st.header("PLS-DA")

    target_col = st.selectbox("Select target (categorical) column", metadata_cols, key="plsda_target")
    y = df_filtered[target_col].astype('category').cat.codes

    selected_features = st.multiselect(
        "Filter compounds to include (leave empty for all)",
        options=compound_cols, key="plsda_features"
    )
    X_pls = X[selected_features] if selected_features else X

    n_components = 2
    pls, scores, loadings, vip_scores = run_plsda(X_pls, y, n_components=n_components,
                                                   scaler_type=scaler_type)
    plot_scatter(df_filtered, scores, metadata_cols, method='PLS-DA')

    st.subheader("VIP Scores")
    st.caption("VIP > 1 indicates important variables for the model.")
    vip_df = pd.DataFrame({
        'Feature': X_pls.columns,
        'VIP Score': vip_scores
    }).sort_values('VIP Score', ascending=False).reset_index(drop=True)

    st.dataframe(
        vip_df.style.background_gradient(cmap='YlOrRd', subset=['VIP Score']).hide(axis='index'),
        use_container_width=True
    )

    with st.expander("PLS-DA Loadings"):
        loadings_df = pd.DataFrame(
            loadings[:, :n_components], index=X_pls.columns,
            columns=[f"Comp{i+1}" for i in range(n_components)]
        )
        st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=None),
                     use_container_width=True)

# ── Univariate Tests ──────────────────────────────────────────────────────────
elif section == "Univariate Tests":
    st.header("Univariate Feature Tests")

    target_col = st.selectbox("Select target column", metadata_cols, key="uni_target")
    y = df_filtered[target_col].astype('category')

    if y.nunique() != 2:
        st.warning(
            f"Univariate tests require exactly 2 groups. "
            f"'{target_col}' has {y.nunique()} unique values — please select a binary column."
        )
    else:
        control_group = st.radio(
            "Select control group for Glass's Δ",
            options=list(y.cat.categories), index=0
        )

        selected_feature = st.selectbox("Select compound", compound_cols, key="uni_compound")

        if selected_feature:
            results, fig, explanations = run_univariate_tests(
                X[selected_feature], y, control_group=control_group
            )

            st.subheader("Distribution Plot")
            st.plotly_chart(fig, use_container_width=False)

            st.subheader("Statistical Test Explanations")
            st.markdown(
                """
                **Normality Test (Shapiro-Wilk)**
                - p > 0.05 → parametric tests are appropriate
                - p ≤ 0.05 → use non-parametric tests

                **t-test (Welch)**
                - Compares means of two independent groups without assuming equal variances.
                - p < 0.05 → significant difference in means

                **Mann–Whitney U test**
                - Non-parametric test comparing ranks. p < 0.05 → distributions differ significantly.

                **Logistic Regression**
                - Models probability of binary outcome from compound value.
                - p < 0.05 → compound significantly predicts group membership.

                **Effect Sizes**
                - Cohen's d → standardised mean difference
                - Glass's Δ → uses SD of the control group
                - Rank-biserial → non-parametric effect size
                """
            )

            st.subheader("Test Results")
            st.table(results)

            st.subheader("Quick Summary")
            st.write("**Normality:**")
            for line in explanations["normality"]:
                st.write("-", line)
            st.write("**Test Choice:**", explanations["test_choice"])
            st.write("**Effect Sizes:**", explanations["effect_sizes"])
            st.write("**P-value Interpretation:**", explanations["pval_interpretation"])
