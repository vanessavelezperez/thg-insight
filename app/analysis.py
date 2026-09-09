from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from plotting import plot_boxplot
from umap import UMAP
from utils import get_scaler


@st.cache_data
def variance(data, scaler_type):
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    variances = data_scaled.var(axis=0)
    return pd.DataFrame({
        'feature': data.columns,
        'variance': variances
    }).sort_values('variance', ascending=False)


@st.cache_data
def cv(data, scaler_type):
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    means = data_scaled.mean(axis=0)
    stds = data_scaled.std(axis=0)
    cvs = np.where(means != 0, 100 * stds / np.abs(means), np.nan)
    return pd.DataFrame({
        'feature': data.columns,
        'cv': cvs
    }).sort_values('cv', ascending=False)


@st.cache_data
def run_pca(data, n_components=2, scaler_type="Standard Scaler"):
    """
    Scales the data, then fits PCA: finds orthogonal directions (components) ranked
    PC1, PC2, ... by how much variance of the scaled data each one explains.

    Two different "components" are in play, easy to conflate:
    - the `components` this function returns are sample SCORES — each sample's
      coordinates in PC space (shape n_samples x n_components), i.e. what gets plotted
      on a PCA scatter plot.
    - `pca.components_` (an attribute of the returned `pca` object) are the LOADING
      VECTORS — each row is one PC's direction in the original feature space (shape
      n_components x n_features), unit-length. pca.components_[k, i] is how much
      feature i contributes to PC k's direction, before any variance rescaling.

    Callers (e.g. the Multivariate page) turn pca.components_ into interpretable
    "loadings" via `pca.components_.T * sqrt(pca.explained_variance_)`: rescaling each
    PC's unit-length direction by the amount of variance that PC actually explains
    turns "which way does this PC point" into "how strongly does each original variable
    correlate with this PC's scores" — that's what makes a loadings heatmap readable
    (large positive/negative loading = that compound is a strong driver of that PC).
    """
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(data_scaled)
    return pca, components


@st.cache_data
def run_tsne(data, n_components=2, scaler_type="Standard Scaler"):
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    tsne = TSNE(n_components=n_components, random_state=42)
    components = tsne.fit_transform(data_scaled)
    return tsne, components


@st.cache_data
def run_kmeans(data, n_clusters=3, scaler_type="Standard Scaler"):
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    silhouette = silhouette_score(data_scaled, labels) if n_clusters > 1 else np.nan

    centroids_df = pd.DataFrame(
        kmeans.cluster_centers_, columns=data.columns,
        index=[f"Cluster {i}" for i in range(n_clusters)]
    )
    return kmeans, labels, silhouette, centroids_df


@st.cache_data
def run_hdbscan(data, min_cluster_size=5, min_samples=None, scaler_type="Standard Scaler"):
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdb.fit_predict(data_scaled)

    mask = labels != -1
    unique_clusters = sorted(set(labels[mask]))
    if len(unique_clusters) > 1 and mask.sum() > 1:
        silhouette = silhouette_score(data_scaled[mask], labels[mask])
    else:
        silhouette = np.nan

    centroids_df = pd.DataFrame(
        [data_scaled[labels == c].mean(axis=0) for c in unique_clusters],
        columns=data.columns, index=[f"Cluster {c}" for c in unique_clusters]
    )
    return hdb, labels, silhouette, centroids_df


def calculate_vip_scores(pls_model):
    """
    VIP (Variable Importance in Projection) scores for a fitted PLS model — one score
    per X variable, summarising how much that variable contributes across all PLS
    components, weighted by how much of Y each component actually explains.

    t = x_scores_ (n_samples x h): each sample's coordinates on each PLS component.
    w = x_weights_ (p_features x h): NOT x_loadings_ — weights are the orthogonal
        per-component directions PLS builds each component from (before deflation),
        which is what VIP is conventionally defined on.
    q = y_loadings_ (n_targets x h): how much each component explains of Y.

    s[j] = (t_j.T @ t_j) * (q_j.T @ q_j) is component j's contribution to explained Y
    variance (SSY); total_s is the SSY explained by the whole model. For variable i,
    its weight on component j is normalised within that component:
    (w[i, j] / ||w[:, j]||)^2 — this turns the raw weight into "share of component j's
    total squared weight that variable i holds," which sums to 1 across all variables
    for a given component. VIP[i] is the s-weighted average of those shares across all
    h components, scaled by sqrt(p).

    That scaling is what makes VIP > 1 a meaningful cutoff: because each component's
    normalised weights sum to 1 across variables, the mean of VIP² across all p
    variables works out to exactly 1 — so VIP > 1 means "more influential than the
    average variable," the standard rule of thumb for flagging important compounds.
    """
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    p, h = w.shape
    vip = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip


@st.cache_data
def run_plsda(X, y, n_components=2, scaler_type="Standard Scaler"):
    scaler = get_scaler(scaler_type)
    if scaler:
        X = scaler.fit_transform(X)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    scores = pls.x_scores_
    loadings = pls.x_loadings_
    vip_scores = calculate_vip_scores(pls)
    return pls, scores, loadings, vip_scores


@st.cache_data
def run_umap(data, n_components=2, scaler_type="Standard Scaler", n_neighbors=15, min_dist=0.1):
    scaler = get_scaler(scaler_type)
    data_scaled = scaler.fit_transform(data) if scaler else data.values
    umap_model = UMAP(n_components=n_components, random_state=42,
                      n_neighbors=n_neighbors, min_dist=min_dist)
    components = umap_model.fit_transform(data_scaled)
    return umap_model, components


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    dof = nx + ny - 2
    pooled_var = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / dof
    if pooled_var == 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled_var)


@st.cache_resource
def rank_biserial(x, y):
    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
        n1, n2 = len(x), len(y)
        return 1 - 2 * u_stat / (n1 * n2)
    except Exception:
        return np.nan


def run_univariate_tests(feature_series, labels, control_group=None):
    if labels.nunique() != 2:
        raise ValueError("run_univariate_tests only supports binary labels.")

    group_names = labels.cat.categories.tolist()
    groups = [feature_series[labels == cat] for cat in group_names]

    normal_p = []
    normality_explanations = []
    for i, g in enumerate(groups):
        if len(g) < 3:
            normal_p.append(np.nan)
            normality_explanations.append(f"{group_names[i]}: not enough data for normality test")
        else:
            p = stats.shapiro(g)[1]
            normal_p.append(p)
            label = "roughly normal" if p > 0.05 else "non-normal"
            normality_explanations.append(f"{group_names[i]}: {label} (p={p:.3f})")

    if all(p > 0.05 for p in normal_p if not np.isnan(p)):
        test_name = "t-test (Welch)"
        stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        test_choice_expl = "Parametric t-test (Welch) used; compares group means assuming normality."
    else:
        test_name = "Mann–Whitney U"
        stat, pval = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        test_choice_expl = "Non-parametric Mann–Whitney U used; compares group ranks without normality assumption."

    X_lr = feature_series.values.reshape(-1, 1)
    y_lr = labels.cat.codes.values
    try:
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_lr, y_lr)
        from scipy.special import expit
        prob = lr_model.predict_proba(X_lr)[:, 1]
        W = prob * (1 - prob)
        X_with_intercept = np.column_stack([np.ones(len(X_lr)), X_lr])
        hessian = X_with_intercept.T @ np.diag(W) @ X_with_intercept
        if np.linalg.det(hessian) != 0:
            var_covar = np.linalg.inv(hessian)
            se_coef = np.sqrt(var_covar[1, 1])
            z_stat = lr_model.coef_[0, 0] / se_coef
            lr_pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            lr_coef = lr_model.coef_[0, 0]
        else:
            lr_pval = np.nan
            lr_coef = lr_model.coef_[0, 0]
    except Exception:
        lr_pval = np.nan
        lr_coef = np.nan

    cohen = cohen_d(groups[0], groups[1])

    if control_group:
        if control_group not in group_names:
            raise ValueError("Control group must match one of the label categories")
        control_idx = group_names.index(control_group)
        treatment_idx = 1 - control_idx
    else:
        control_idx = 0
        treatment_idx = 1

    control = groups[control_idx]
    treatment = groups[treatment_idx]

    if len(control) < 2 or control.std(ddof=1) == 0:
        glass = np.nan
    else:
        glass = (treatment.mean() - control.mean()) / control.std(ddof=1)

    rb = rank_biserial(groups[0], groups[1])
    effect_sizes_expl = (
        f"Cohen's d: {cohen:.3f}, "
        f"Glass's Δ (control={group_names[control_idx]}): {glass:.3f}, "
        f"Rank-biserial: {rb:.3f}"
    )

    pval_expl = (
        f"p = {pval:.3f} → statistically significant difference"
        if pval < 0.05
        else f"p = {pval:.3f} → no statistically significant difference"
    )

    results = pd.DataFrame({
        "Test": [test_name, "Logistic Regression"],
        "p-value": [pval, lr_pval],
        "Coefficient": [np.nan, lr_coef],
        "Cohen's d": [cohen, np.nan],
        "Glass's Δ": [glass, np.nan],
        "Rank-biserial": [rb, np.nan],
        "Normality p (group1)": [normal_p[0], np.nan],
        "Normality p (group2)": [normal_p[1], np.nan],
    })

    fig = plot_boxplot(group_names, groups, feature_series.name)

    explanations = {
        "normality": normality_explanations,
        "test_choice": test_choice_expl,
        "effect_sizes": effect_sizes_expl,
        "pval_interpretation": pval_expl,
    }

    return results, fig, explanations


def calculate_outlier_thresholds(values, method):
    values = pd.Series(values).dropna()

    if method == "Mean ± 2 SD":
        mean_val, std_val = values.mean(), values.std()
        return {'lower': mean_val - 2 * std_val, 'upper': mean_val + 2 * std_val,
                'label': '±2 SD', 'stats': {'mean': mean_val, 'std': std_val}}
    elif method == "Mean ± 3 SD":
        mean_val, std_val = values.mean(), values.std()
        return {'lower': mean_val - 3 * std_val, 'upper': mean_val + 3 * std_val,
                'label': '±3 SD', 'stats': {'mean': mean_val, 'std': std_val}}
    elif method == "IQR (1.5×IQR)":
        Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
        IQR = Q3 - Q1
        return {'lower': Q1 - 1.5 * IQR, 'upper': Q3 + 1.5 * IQR,
                'label': 'IQR (1.5×)', 'stats': {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}}
    elif method == "IQR (3×IQR)":
        Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
        IQR = Q3 - Q1
        return {'lower': Q1 - 3 * IQR, 'upper': Q3 + 3 * IQR,
                'label': 'IQR (3×)', 'stats': {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}}
    elif method == "Modified Z-score (MAD)":
        median_val = values.median()
        MAD = np.median(np.abs(values - median_val))
        threshold = 3.5
        return {'lower': median_val - threshold * MAD / 0.6745,
                'upper': median_val + threshold * MAD / 0.6745,
                'label': 'MAD (3.5)', 'stats': {'median': median_val, 'MAD': MAD}}
    elif method == "Percentile (5th/95th)":
        return {'lower': values.quantile(0.05), 'upper': values.quantile(0.95),
                'label': '5th/95th %ile', 'stats': {}}
    elif method == "Percentile (1st/99th)":
        return {'lower': values.quantile(0.01), 'upper': values.quantile(0.99),
                'label': '1st/99th %ile', 'stats': {}}
    else:
        return {'lower': None, 'upper': None, 'label': None, 'stats': {}}


@st.cache_data
def detect_outliers(df, compound_cols, method="IQR (1.5×IQR)", scale=True):
    results = []
    for col in compound_cols:
        values = df[col].copy()
        if scale:
            from sklearn.preprocessing import StandardScaler as _SS
            values = pd.Series(
                _SS().fit_transform(values.values.reshape(-1, 1)).flatten(),
                index=values.index
            )
        thresholds = calculate_outlier_thresholds(values, method)
        if thresholds['lower'] is not None:
            is_outlier = (values < thresholds['lower']) | (values > thresholds['upper'])
            n_outliers = is_outlier.sum()
            pct_outliers = 100 * n_outliers / len(values)
            cv_val = 100 * values.std() / values.mean() if values.mean() != 0 else np.nan
            results.append({
                'Compound': col, 'N_Outliers': n_outliers, 'Pct_Outliers': pct_outliers,
                'Lower_Threshold': thresholds['lower'], 'Upper_Threshold': thresholds['upper'],
                'Mean': values.mean(), 'Median': values.median(), 'Std': values.std(),
                'CV_%': cv_val, 'Min': values.min(), 'Max': values.max()
            })
        else:
            results.append({
                'Compound': col, 'N_Outliers': 0, 'Pct_Outliers': 0,
                'Lower_Threshold': None, 'Upper_Threshold': None,
                'Mean': values.mean(), 'Median': values.median(), 'Std': values.std(),
                'CV_%': 100 * values.std() / values.mean() if values.mean() != 0 else np.nan,
                'Min': values.min(), 'Max': values.max()
            })
    return pd.DataFrame(results)


@st.cache_data
def detect_outliers_grouped(df, feature_col, group_col, method="IQR (1.5×IQR)", scale=True):
    from sklearn.preprocessing import StandardScaler as _SS
    rows = []
    for group, subset in df.groupby(group_col):
        values = subset[feature_col].copy()
        if scale:
            scaled = pd.Series(
                _SS().fit_transform(values.values.reshape(-1, 1)).flatten(),
                index=values.index
            )
        else:
            scaled = values

        thresh = calculate_outlier_thresholds(scaled, method)
        g_mean = scaled.mean()
        g_std = scaled.std()

        for idx in subset.index:
            v = scaled.loc[idx]
            is_out = (v < thresh['lower'] or v > thresh['upper']) if thresh['lower'] is not None else False
            z = (v - g_mean) / g_std if g_std and g_std != 0 else np.nan
            row = subset.loc[idx].to_dict()
            row.update({
                'group_mean': round(g_mean, 4),
                'group_std': round(g_std, 4),
                'group_rsd': round((g_std / g_mean) * 100) if g_mean != 0 else np.nan,
                'group_lower': round(thresh['lower'], 4) if thresh['lower'] is not None else None,
                'group_upper': round(thresh['upper'], 4) if thresh['upper'] is not None else None,
                'z_within_group': round(z, 2) if not np.isnan(z) else np.nan,
                'pct_diff_from_mean': np.abs(round(100 * (v - g_mean) / g_mean, 2)) if g_mean != 0 else np.nan,
                'is_outlier': is_out,
            })
            rows.append(row)

    return pd.DataFrame(rows, index=df.index)


@st.cache_data
def compute_correlation_matrix(df, compound_cols, method):
    return df[list(compound_cols)].corr(method=method)


@st.cache_data
def calculate_cv_by_group(df, compound_cols, group_col):
    results = []
    for compound in compound_cols:
        for group_name, group_data in df.groupby(group_col):
            values = group_data[compound].dropna()
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                cv_val = 100 * std_val / mean_val if mean_val != 0 else np.nan
                results.append({
                    'Compound': compound, group_col: group_name, 'N': len(values),
                    'Mean': mean_val, 'Std': std_val, 'CV %': cv_val,
                    'Min': values.min(), 'Max': values.max()
                })
    return pd.DataFrame(results)


# ── New analysis functions ────────────────────────────────────────────────────

@st.cache_data
def calculate_detection_rate(df, compound_cols, group_col=None):
    """
    Detection rate (% samples where value > 0) per compound.
    Returns wide DataFrame: compounds as rows, Overall + group columns.
    """
    result = {'Compound': compound_cols}
    result['Overall'] = [(df[col] > 0).mean() * 100 for col in compound_cols]

    if group_col and group_col in df.columns:
        for group in sorted(df[group_col].dropna().unique()):
            subset = df[df[group_col] == group]
            result[str(group)] = [(subset[col] > 0).mean() * 100 for col in compound_cols]

    return pd.DataFrame(result)


@st.cache_data
def calculate_d_ratio(df, compound_cols, group_col):
    """
    D-ratio = pooled within-group SD / total SD per compound.
    Lower D-ratio = more between-group variation relative to within-group noise (better for discriminant analysis).
    """
    results = []
    for col in compound_cols:
        total_sd = df[col].std()
        if total_sd == 0 or pd.isna(total_sd):
            results.append({'compound': col, 'd_ratio': np.nan, 'median_rsd': np.nan})
            continue

        group_stats = df.groupby(group_col)[col].agg(['std', 'mean'])
        within_sd = group_stats['std'].mean()
        d_ratio = within_sd / total_sd

        rsd_vals = 100 * group_stats['std'] / group_stats['mean'].abs()
        median_rsd = rsd_vals.replace([np.inf, -np.inf], np.nan).median()

        results.append({'compound': col, 'd_ratio': d_ratio, 'median_rsd': median_rsd})

    return pd.DataFrame(results)


@st.cache_data
def calculate_log_fc(df, compound_cols, group_col, group1, group2):
    """
    Log fold change between two groups (assumes data is already log-transformed,
    so FC = difference in log space).
    Returns DataFrame with log_fc, pval, neg_log10_p per compound.
    """
    g1 = df[df[group_col] == group1]
    g2 = df[df[group_col] == group2]

    results = []
    for col in compound_cols:
        g1_vals = g1[col].dropna()
        g2_vals = g2[col].dropna()
        log_fc = g1_vals.mean() - g2_vals.mean()

        if len(g1_vals) >= 3 and len(g2_vals) >= 3:
            _, pval = stats.mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
        else:
            pval = np.nan

        results.append({
            'compound': col,
            'log_fc': log_fc,
            'pval': pval,
            'neg_log10_p': -np.log10(pval) if pd.notna(pval) and pval > 0 else np.nan
        })

    return pd.DataFrame(results)
