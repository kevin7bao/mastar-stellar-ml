# Stellar Parameter Estimation and Anomaly Detection in the MaStar Spectral Library

**SIADS 696: Milestone II - University of Michigan Master of Applied Data Science**

**Team:** Zachary Snow · Kevin Bao · Bonseok Koo

---

## Project Overview

This project applies supervised and unsupervised machine learning to the [MaStar (MaNGA Stellar Library)](https://www.sdss.org/dr17/mastar/) from SDSS Data Release 17. MaStar is one of the largest empirical stellar spectral libraries ever assembled, covering a broad range of stellar types across a wavelength range of ~3,622–10,354 Å at 4,563 wavelength bins per spectrum.

**Part A - Supervised Learning:** We predict three continuous stellar parameters - effective temperature (T_eff), surface gravity (log g), and metallicity ([Fe/H]) - directly from continuum-normalized spectral flux. These are regression tasks on high-dimensional input (4,563 features per star).

**Part B - Unsupervised Learning:** We apply dimensionality reduction (PCA) and clustering/anomaly detection (k-means, LOF) to the full quality-filtered library to discover natural groupings in stellar population space and flag spectroscopically anomalous stars that warrant further astrophysical investigation.

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
│   │   └── siads696_preprocessing.ipynb     ← Data download, filtering, normalization, EDA
│   ├── 02_supervised/
│   │   └── siads696_supervised.ipynb        ← Part A: regression models
│   └── 03_unsupervised/
│       └── siads696_unsupervised.ipynb      ← Part B: PCA, k-means, LOF
│
├── data/
│   ├── raw/                         ← EMPTY (see Data Access - files too large for git)
│   ├── processed/                   ← EMPTY (outputs written to Google Drive)
│   └── samples/
│       └── mastar_sample_100.csv    ← 100-record sample for repo compliance
│
└── reports/
    └── siads696_final_report.pdf    ← Final submitted report
```

---

## Methods

### Part A: Supervised Learning

- **Task:** Regression (predict T_eff, log g, [Fe/H] from spectral flux)
- **Training set:** ~3,291 labeled stars with stratified train/test split
- **Models:** Ridge Regression (linear baseline), Random Forest (nonlinear tree ensemble), XGBoost (gradient boosting) - compared with and without PCA as feature representation
- **Evaluation:** RMSE, R², learning curves, feature importance, ablation analysis, sensitivity analysis, failure analysis (≥3 failure categories, ≥3 example records each)
- **Key challenge:** 4,563-dimensional input with class imbalance (K-type stars ~59% of labeled set)

### Part B: Unsupervised Learning

- **Task:** Structure discovery and anomaly detection on the full ~15,200 star dataset
- **Pipeline:** Zero-coverage audit → StandardScaler → PCA (N=50, parallel analysis + broken stick) → k-means (elbow/silhouette/Davies-Bouldin sweep) → Isolation Forest (primary, contamination=0.05) + LOF (cross-reference) → 263 high-confidence anomalies at overlap
- **Visualization:** Kiel diagram (T_eff vs log g), UMAP projections (exploratory)
- **Sanity check:** Anomaly flags cross-referenced against the labeled subset as partial ground truth
- **Known design constraint:** LOF cannot distinguish instrument coverage artifacts (wavelength zero-padding at red end) from genuine astrophysical anomalies - this is addressed in our zero-coverage audit and discussed explicitly in the report

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

> **Note:** Use `mmap_mode='r'` - the full flux arrays are hundreds of MB and should not be loaded entirely into RAM.

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
2. `notebooks/02_supervised/siads696_supervised.ipynb`
3. `notebooks/03_unsupervised/siads696_unsupervised.ipynb`

---

## Key Findings

**Supervised Learning**
- Best model: XGBoost (no PCA) - CV R² of 0.962 / 0.890 / 0.882 for Teff, log g, [Fe/H]
- Corresponding MAEs: 128 K, 0.265 dex, 0.166 dex
- Primary failure categories: rare M-type stars, very metal-poor K-type stars, A-type stars at temperature extremes

**Unsupervised Learning**
- PCA: 50 components retained (~59% variance explained, below noise floor at PC 176)
- K-means optimal k=2 by silhouette score; k=6 used for interpretability sanity check
- Isolation Forest flagged 760 stars (5% contamination); 263 overlap with LOF as highest-confidence anomalies

---

## Team & Contributions

| Team Member | Primary Responsibilities |
|---|---|
| Zach Snow | Data loading, preprocessing pipeline, EDA |
| Bonseok Koo | Supervised learning (Part A) |
| Kevin Bao | Unsupervised learning (Part B) |

---

## Generative AI Acknowledgment

Claude (Anthropic) was used for assistance in the development of the data preprocessing 
pipeline, including debugging FITS file handling and memory management for large array 
loading. All modeling, analysis, and written report content is the 
original work of the project team. Per SIADS 696 course policy, GenAI was not used in 
the written report.

---

## References

Yan, R., et al. (2019). MaStar: The MaNGA Stellar Library. *The Astrophysical Journal Supplement Series*, 240(1), 14. https://doi.org/10.3847/1538-4365/aaf5fb

Fabbro, S., Venn, K. A., O'Briain, T., et al. (2018). An application of deep learning in the analysis of stellar spectra. *Monthly Notices of the Royal Astronomical Society*, 475(3), 2978–2993. https://arxiv.org/abs/1709.09182

Anders, F., Khalatyan, A., Queiroz, A., Nepal, S., & Chiappini, C. (2023). Parameters for >300 million Gaia stars: Bayesian inference vs. machine learning. *Highlights of Spanish Astrophysics XI*. https://arxiv.org/abs/2302.06995

Reis, I., Baron, D., & Shahaf, S. (2018). Detecting outliers and learning complex structures with large spectroscopic surveys: a case study with APOGEE stars. *Monthly Notices of the Royal Astronomical Society*, 476(2), 2117–2136. https://arxiv.org/abs/1711.00022

SDSS Collaboration. (2022). *SDSS DR17 MaStar data access.* https://www.sdss4.org/dr17/mastar/mastar-data-access/

---

*SIADS 696 Milestone II, University of Michigan School of Information, Summer 2026*
