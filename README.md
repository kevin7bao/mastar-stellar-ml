# Stellar Parameter Estimation and Anomaly Detection in the MaStar Spectral Library

**SIADS 696: Milestone II — University of Michigan Master of Applied Data Science**

**Team:** Zachary Snow · Kevin Bao · Bonseok Koo

---

## Project Overview

This project applies supervised and unsupervised machine learning to the [MaStar (MaNGA Stellar Library)](https://www.sdss.org/dr17/mastar/) from SDSS Data Release 17. MaStar is one of the largest empirical stellar spectral libraries ever assembled, covering a broad range of stellar types across a wavelength range of ~3,622–10,354 Å at 4,563 wavelength bins per spectrum.

**Part A — Supervised Learning:** We predict three continuous stellar parameters — effective temperature (T_eff), surface gravity (log g), and metallicity ([Fe/H]) — directly from continuum-normalized spectral flux. These are regression tasks on high-dimensional input (4,563 features per star).

**Part B — Unsupervised Learning:** We apply dimensionality reduction (PCA) and clustering/anomaly detection (k-means, LOF) to the full quality-filtered library to discover natural groupings in stellar population space and flag spectroscopically anomalous stars that warrant further astrophysical investigation.

---

## Dataset

| Property | Value |
|---|---|
| Source | SDSS DR17 MaStar Good Spectra (`mastar-goodspec-v3_1_1-v1_7_7.fits.gz`) |
| Full working dataset (quality-filtered, `MJDQUAL == 0`) | ~15,200 stars |
| Labeled subset (stars with INPUT_TEFF / INPUT_LOGG / INPUT_FE_H) | ~3,291 stars |
| Spectral resolution | 4,563 wavelength bins per spectrum |
| Wavelength range | ~3,622 – 10,354 Å |
| Feature representation | Continuum-normalized flux |

**Important:** The raw FITS files are ~5–10 GB and are **not committed to this repo.** See [Data Access](#data-access) below.

### Preprocessing Output Files (Google Drive)

| File | Shape | Description |
|---|---|---|
| `mastar_labeled_meta.csv` | 3,291 × 5 | Metadata + labels for supervised learning (MANGAID, SPECTRAL_TYPE, INPUT_TEFF, INPUT_LOGG, INPUT_FE_H) |
| `mastar_labeled_spectra.npy` | 3,291 × 4,563 | Continuum-normalized flux arrays for labeled stars |
| `mastar_full_meta.csv` | 15,200 × cols | Metadata for all quality-filtered stars (unsupervised) |
| `mastar_full_spectra.npy` | 15,200 × 4,563 | Continuum-normalized flux arrays for all good stars |
| `mastar_wave.npy` | 4,563 | Shared wavelength grid in Angstroms |

---

## Repository Structure

```
mastar-stellar-ml/
│
├── README.md                        ← You are here
├── .gitignore                       ← Excludes raw FITS, large .npy, Drive paths
├── requirements.txt                 ← Python dependencies
│
├── notebooks/
│   ├── 01_preprocessing/
│   │   └── siads696_preprocessing.ipynb     ← Data download, filtering, normalization
│   ├── 02_eda/
│   │   └── siads696_eda.ipynb               ← Exploratory analysis (both subsets)
│   ├── 03_supervised/
│   │   └── siads696_supervised.ipynb        ← Part A: regression models
│   └── 04_unsupervised/
│       └── siads696_unsupervised.ipynb      ← Part B: PCA, k-means, LOF
│
├── src/
│   └── preprocessing_utils.py       ← Shared utility functions (visit selection, normalization)
│
├── data/
│   ├── raw/                         ← EMPTY (see Data Access — files too large for git)
│   ├── processed/                   ← EMPTY (outputs written to Google Drive)
│   └── samples/
│       └── mastar_sample_100.csv    ← 100-record sample for repo compliance
│
├── results/
│   ├── supervised/                  ← Saved model artifacts, metric CSVs
│   ├── unsupervised/                ← Cluster assignments, anomaly flags
│   └── figures/                     ← All plots included in the final report
│
└── reports/
    └── siads696_final_report.pdf    ← Final submitted report
```

---

## Methods

### Part A: Supervised Learning

- **Task:** Regression (predict T_eff, log g, [Fe/H] from spectral flux)
- **Training set:** ~3,291 labeled stars with stratified train/test split
- **Models:** ≥2 model families (e.g., Random Forest, Ridge Regression / MLP)
- **Evaluation:** RMSE, R², learning curves, feature importance, ablation analysis, sensitivity analysis, failure analysis (≥3 failure categories, ≥3 example records each)
- **Key challenge:** 4,563-dimensional input with class imbalance (K-type stars ~59% of labeled set)

### Part B: Unsupervised Learning

- **Task:** Structure discovery and anomaly detection on the full ~15,200 star dataset
- **Pipeline:** Zero-coverage audit → StandardScaler → PCA (scree plot for component selection) → k-means (elbow/silhouette/Davies-Bouldin sweep) → LOF (contamination=0.05, sensitivity analysis at 0.02/0.05/0.10)
- **Visualization:** Kiel diagram (T_eff vs log g), UMAP projections (exploratory)
- **Sanity check:** Anomaly flags cross-referenced against the labeled subset as partial ground truth
- **Known design constraint:** LOF cannot distinguish instrument coverage artifacts (wavelength zero-padding at red end) from genuine astrophysical anomalies — this is addressed in our zero-coverage audit and discussed explicitly in the report

---

## Data Access

Raw files are too large for GitHub (~5 GB compressed, ~10 GB decompressed). The preprocessing notebook downloads them directly from SDSS:

```
https://data.sdss.org/sas/dr17/manga/spectro/mastar/v3_1_1/v1_7_7/mastar-goodspec-v3_1_1-v1_7_7.fits.gz
```

Processed outputs (`.csv`, `.npy`) are stored in the team's shared Google Drive at `/content/drive/MyDrive/SIADS 696/data`. To load them in Colab:

```python
import numpy as np
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')
DRIVE_DIR = "/content/drive/MyDrive/SIADS 696/data"

labeled_meta  = pd.read_csv(f"{DRIVE_DIR}/mastar_labeled_meta.csv")
labeled_flux  = np.load(f"{DRIVE_DIR}/mastar_labeled_spectra.npy", mmap_mode='r')
full_meta     = pd.read_csv(f"{DRIVE_DIR}/mastar_full_meta.csv")
full_flux     = np.load(f"{DRIVE_DIR}/mastar_full_spectra.npy", mmap_mode='r')
wave          = np.load(f"{DRIVE_DIR}/mastar_wave.npy")
```

> **Note:** Use `mmap_mode='r'` — the full flux arrays are hundreds of MB and should not be loaded entirely into RAM.

A 100-record sample (`data/samples/mastar_sample_100.csv`) is committed to the repo to satisfy the course data submission requirement for datasets over 25 MB.

---

## Setup & Reproducibility

This project was developed in Google Colab. All notebooks are self-contained and mount Google Drive for data access.

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `astropy`, `scipy`, `tqdm`

To reproduce results, run notebooks in order:
1. `notebooks/01_preprocessing/siads696_preprocessing.ipynb`
2. `notebooks/02_eda/siads696_eda.ipynb`
3. `notebooks/03_supervised/siads696_supervised.ipynb`
4. `notebooks/04_unsupervised/siads696_unsupervised.ipynb`

---

## Key Findings (TBD — to be updated as analysis completes)

- [ ] Supervised: Best model and RMSE for T_eff, log g, [Fe/H]
- [ ] Supervised: Primary failure categories identified
- [ ] Unsupervised: Number of PCA components selected and % variance explained
- [ ] Unsupervised: Optimal k for k-means
- [ ] Unsupervised: LOF anomaly count at contamination=0.05 and cross-validation against labeled set

---

## Team & Contributions

| Team Member | Primary Responsibilities |
|---|---|
| Zach Snow | Data loading, preprocessing pipeline, EDA |
| Bonseok Koo | Supervised learning (Part A) |
| Kevin Bao | Unsupervised learning (Part B) |

---

## References

Yan, R., et al. "MaNGA Stellar Library (MaStar): Spectra of ~59,000 Stars." *The Astrophysical Journal Supplement Series* 253.1 (2021): 26.

SDSS Collaboration. *SDSS DR17 MaStar Data Access.* https://www.sdss.org/dr17/mastar/

---

*SIADS 696 Milestone II, University of Michigan School of Information, Summer 2026*
