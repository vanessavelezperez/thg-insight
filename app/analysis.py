# analysis.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

@st.cache_resource
def run_pca(data, n_components=2, scale=True):
    data_scaled = StandardScaler().fit_transform(data) if scale else data.values
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(data_scaled)
    return pca, components

@st.cache_resource
def run_tsne(data, n_components=2, scale=True):
    data_scaled = StandardScaler().fit_transform(data) if scale else data.values
    tsne = TSNE(n_components=n_components, random_state=42)
    components = tsne.fit_transform(data_scaled)
    return tsne, components

@st.cache_resource
def run_plsda(X, y, n_components=2, scale=True):
    """
    Runs OPLS-DA on X (features) with y (labels).
    Returns: pls_model, scores (T), loadings (P)
    """
    if scale:
        X = StandardScaler().fit_transform(X)

    # Fit PLSRegression
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    # Get scores (T) and loadings (P)
    scores = pls.x_scores_
    loadings = pls.x_loadings_

    return pls, scores, loadings

def run_oplsda(X, y, n_components=2, scale=True):
    """
    Runs OPLS-DA on X (features) with y (labels).
    Returns: pls_model, scores (T), loadings (P)
    """
    pls, scores, loadings = run_plsda(X, y, n_components=n_components, scale=scale)

    #reconstruct X from predictive component
    X_hat = scores @ loadings.T #dot product
    X_orthogonal = X - X_hat

    

    return pls, scores, X_orthogonal

def cohen_d(x, y):
    """Effect size for t-test, with safety checks"""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan  # Not enough data to compute
    dof = nx + ny - 2
    pooled_var = ((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / dof
    if pooled_var == 0:
        return np.nan  # Avoid division by zero if variance is zero
    pooled_std = np.sqrt(pooled_var)
    return (x.mean() - y.mean()) / pooled_std

@st.cache_resource
def rank_biserial(x, y):
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
        n1, n2 = len(x), len(y)
        return 1 - 2 * u_stat / (n1 * n2)
    except:
        return np.nan

@st.cache_resource
def run_univariate_tests(feature_series, labels, control_group=None):
    """
    Run univariate tests for one compound against a binary label.
    Returns:
        results: pd.DataFrame with p-value, effect sizes
        fig: matplotlib boxplot of distributions
        explanations: dict of bullet-point explanations for Streamlit
    """
    if labels.nunique() != 2:
        raise ValueError("run_univariate_tests only supports binary labels.")

    group_names = labels.cat.categories.tolist()
    groups = [feature_series[labels == cat] for cat in group_names]

    # --- Normality tests ---
    normal_p = []
    normality_explanations = []
    for i, g in enumerate(groups):
        if len(g) < 3:
            normal_p.append(np.nan)
            normality_explanations.append(f"{group_names[i]}: not enough data for normality test")
        else:
            p = stats.shapiro(g)[1]
            normal_p.append(p)
            if p > 0.05:
                normality_explanations.append(f"{group_names[i]}: roughly normal (p={p:.3f})")
            else:
                normality_explanations.append(f"{group_names[i]}: non-normal (p={p:.3f})")

    # --- Choose statistical test ---
    if all(p > 0.05 for p in normal_p if not np.isnan(p)):
        test_name = "t-test (Welch)"
        stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        test_choice_expl = "Parametric t-test (Welch) used; compares group means assuming normality."
    else:
        test_name = "Mann–Whitney U"
        stat, pval = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        test_choice_expl = "Non-parametric Mann–Whitney U used; compares group ranks without normality assumption."

    # --- Effect sizes ---
    cohen = cohen_d(groups[0], groups[1])

    # Glass's Δ using selected control
    if control_group:
        if control_group not in group_names:
            raise ValueError("Control group must match one of the label categories")
        control_idx = group_names.index(control_group)
        treatment_idx = 1 - control_idx
    else:
        # default: first group is control
        control_idx = 0
        treatment_idx = 1

    control = groups[control_idx]
    treatment = groups[treatment_idx]

    if len(control) < 2 or control.std(ddof=1) == 0:
        glass = np.nan
    else:
        glass = (treatment.mean() - control.mean()) / control.std(ddof=1)

    rb = rank_biserial(groups[0], groups[1])

    effect_sizes_expl = f"Cohen's d: {cohen:.3f}, Glass's Δ (control={group_names[control_idx]}): {glass:.3f}, Rank-biserial: {rb:.3f}"

    # --- p-value explanation ---
    if pval < 0.05:
        pval_expl = f"p = {pval:.3f} → statistically significant difference"
    else:
        pval_expl = f"p = {pval:.3f} → no statistically significant difference"

    # --- Prepare results table ---
    results = pd.DataFrame({
        "Test": [test_name],
        "p-value": [pval],
        "Cohen's d": [cohen],
        "Glass's Δ": [glass],
        "Rank-biserial": [rb],
        "Normality p (group1)": [normal_p[0]],
        "Normality p (group2)": [normal_p[1]],
    })

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(groups, labels=group_names)
    ax.set_title(f"Distribution of {feature_series.name}")
    ax.set_ylabel("Intensity")

    # --- Explanations dict for Streamlit ---
    explanations = {
        "normality": normality_explanations,
        "test_choice": test_choice_expl,
        "effect_sizes": effect_sizes_expl,
        "pval_interpretation": pval_expl,
    }

    return results, fig, explanations

def get_ellipse(x, y, n_std=2.0):
    if len(x) < 2:
        return [], []

    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(*eigenvecs[:, 0][::-1])

    # Chi-squared value for 95% confidence
    chisq_val = np.sqrt(stats.chi2.ppf(0.95, df=2))

    width, height = 2 * n_std * chisq_val * np.sqrt(eigenvals)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array([width / 2 * np.cos(theta), height / 2 * np.sin(theta)])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ellipse_rot = R @ ellipse
    ellipse_x, ellipse_y = ellipse_rot[0] + mean_x, ellipse_rot[1] + mean_y

    return ellipse_x, ellipse_y