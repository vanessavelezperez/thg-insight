# THG Insight — VOC Analytics Dashboard

A Streamlit app for exploring volatile organic compound (VOC) feature matrices: one row per sample, metadata columns (e.g. patient, batch, sample type) plus one intensity column per compound. Upload a table and work through data overview, QC, feature filtering, single-compound exploration, multivariate structure and supervised analysis, all in the browser with no code.

## What's here

```
app/
├── Home.py                     # entry point: upload data, merge metadata, pick metadata columns, optional peak file
├── pages/
│   ├── 1_Data_Overview.py      # sample counts, missing values, value distributions
│   ├── 2_Analytical_QC.py      # peak shape, detectability, reproducibility (needs a peak file)
│   ├── 3_Feature_Filtering.py  # variance/CV, correlation network, detection rate, D-ratio + RSD by group, fold change
│   ├── 4_Feature_Exploration.py# single-compound deep dive: distributions, outliers, across groups
│   ├── 5_Multivariate.py       # PCA, t-SNE, UMAP
│   └── 6_Supervised.py         # PLS-DA and per-compound univariate tests
├── analysis.py                 # calculations (CV, D-ratio, detection rate, fold change, PCA/UMAP, PLS-DA, ...)
├── plotting.py                 # Plotly / pyvis figures
└── utils.py                    # data loading, shared sidebar (preprocessing + filters), helpers
.streamlit/config.toml          # Streamlit settings (max upload size)
environment.yml                 # conda environment
```

## Install

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda).

```bash
git clone <repo-url>
cd thg-insight
conda env create -f environment.yml
conda activate thg-insight
```

To update an existing environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

## Run

From the repo root:

```bash
conda activate thg-insight
streamlit run app/Home.py
```

The app opens at http://localhost:8501. Stop it with `Ctrl+C`.

Always start it from the repo root: Streamlit only picks up the project settings in [`.streamlit/config.toml`](.streamlit/config.toml) from the folder you run it in. Those settings raise the upload limit from Streamlit's default 200 MB to 10 GB (`maxUploadSize = 10000`). Edit that file to change it.

## Using the app

1. **Upload data** on the Home page: one row per sample. Files must be `.csv` or `.xlsx`; save older `.xls` files as `.xlsx` first. For Excel, pick the worksheet.
2. **Metadata (optional):** if sample metadata is in a separate file, upload it and merge on a shared ID column.
3. **Column setup:** select which columns are metadata. Every other column is treated as a compound.
4. **Peak file (optional):** a `.npy` peak file, only needed for the Analytical QC page.
5. Open any page from the sidebar.

### Shared sidebar (every analysis page)

- **Apply log1p transformation:** on by default.
- **Scaler:** None, Standard, Robust or Pareto (used by the variance/CV and multivariate views).
- **Exclude internal standards:** drop selected compounds from all analyses.
- **Bin numeric metadata:** turn a continuous column (e.g. age) into categories for grouping.
- **Filters:** subset samples by any metadata column.

### How the data is preprocessed

Applied to a working copy when the data is loaded. The uploaded file is never changed.

- In compound columns, missing values and ±inf become **0**, and negative values are clipped to 0.
- log1p is applied if the sidebar box is ticked.
- Missing metadata values become the text `"NA"`.

So by the time any analysis runs, a non-detect and a missing value are both 0.

### Peak file format (Analytical QC)

A NumPy `.npy` file containing a pickled dict, `{filename: [peak, ...]}`. Each peak is a dict with keys such as `Compound`, `calc_RT`, `peak_area`, `asymmetry`, `mass_spec`, `peakRT`, `peak_start`, `peak_end`, `peak_I` and `mz`. A peak is matched to a compound column when the first and last `_`-separated parts of its `Compound` name match those of the column name.

## Notes

- Only load `.npy` peak files you trust: they are read with `allow_pickle=True`, which can run code.
- Large datasets: results are cached per input, so the first run of a section is the slowest.
