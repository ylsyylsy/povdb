# POVDB — Potential Odorous Virtual Database

A machine-learning-enhanced **Olfactory Data Hub** for smart-city odor management.

This repository provides the tools to **use the Olfactory Data Hub** built for
odor-free urban management: predict whether a chemical is odorous and estimate
its olfactory threshold, search the Potential Odorous Virtual Database (POVDB),
and annotate non-targeted GC-QTOF peaks with odor contributions.

> **Citation:** Wang et al., *Embedding an Olfactory Data Hub in Smart Cities
> for Odor-Free Urban Management*, Science Advances (2026).
> Corresponding authors: Nanyang Yu (yuny@nju.edu.cn), Si Wei (weisi@nju.edu.cn).

---

## Table of Contents

- [What's in this repository](#whats-in-this-repository)
- [Installation](#installation)
- [Quick start](#quick-start)
  - [1. Predict odor properties from SMILES](#1-predict-odor-properties-from-smiles)
  - [2. Search the POVDB spectral library](#2-search-the-povdb-spectral-library)
  - [3. Annotate non-targeted analysis peaks](#3-annotate-non-targeted-analysis-peaks)
- [How the prediction model works](#how-the-prediction-model-works)
- [Reproducing the model](#reproducing-the-model)
- [The POVDB library](#the-povdb-library)
- [Repository layout](#repository-layout)
- [Data availability](#data-availability)
- [License](#license)
- [Contact](#contact)

---

## What's in this repository

| Component | Description |
|-----------|-------------|
| **Odor prediction** | Pre-trained model (`models/best_model.joblib`) that takes a SMILES string and returns odorous/odorless classification plus a quantitative olfactory threshold (ppm). |
| **POVDB search** | Query the odorous spectral library by name, SMILES, formula, exact mass, or experimental spectrum matching. |
| **Peak annotation** | Map GC-QTOF / MS-DIAL peak tables to POVDB entries and compute odor contribution = (peak area) / (olfactory threshold). |
| **Training scripts** | Reproduce the prediction models from the training data (`data/`). |
| **Sample library** | `data/sample_povdb.msp` (30 records) demonstrating the MSP format of the full 737,519-record library. |

---

## Installation

Requires **Python ≥ 3.6** with [RDKit](https://www.rdkit.org/) and scikit-learn.

```bash
# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The pre-trained model is committed at `models/best_model.joblib` — no
training is required for prediction. If you prefer to train it yourself, see
[Reproducing the model](#reproducing-the-model).

---

## Quick start

### 1. Predict odor properties from SMILES

**Command line:**

```bash
# single molecule
python scripts/predict.py --smiles "CCSCC"

# batch from CSV
python scripts/predict.py --csv examples/example_data.csv --output results.csv

# from stdin (one SMILES per line)
echo "CC(=O)C" | python scripts/predict.py
```

**Python API:**

```python
from povdb import OdorPredictor

predictor = OdorPredictor(model_dir="models")

result = predictor.predict("CCSCC")   # diethyl sulfide
print(result)
# {'smiles': 'CCSCC', 'is_odorous': True, 'threshold_ppm': 5.6e-05, 'threshold_bin': 1}
```

`threshold_ppm` is the predicted olfactory threshold in ppm; `is_odorous` is
`True` when the threshold is below 1 ppm.

Run `python examples/example_predict.py` for a complete walkthrough.

### 2. Search the POVDB spectral library

```bash
python scripts/search_povdb.py --name acetone
python scripts/search_povdb.py --smiles "CC(=O)C"
python scripts/search_povdb.py --formula C3H6O
python scripts/search_povdb.py --mass 58.0419 --tolerance 0.005
python scripts/search_povdb.py --spectrum my_peaks.txt --top-k 5
```

**Python API:**

```python
from povdb.query import POVDBQuery

db = POVDBQuery()                    # loads data/sample_povdb.msp by default

db.query_by_name("acetone")          # -> [record dicts]
db.query_by_smiles("CC(=O)C")        # exact standardized-SMILES match
db.find_similar("CC(=O)C", 0.6)      # MACCS Tanimoto >= 0.6 (applicability domain)
db.search_spectrum([(43, 999), (58, 500)])   # experimental m/z -> virtual spectra
```

### 3. Annotate non-targeted analysis peaks

Given a peak table from MS-DIAL / GC-QTOF analysis (columns such as `SMILES`,
`Name`, `Peak Area`):

```bash
python scripts/annotate_peaks.py --input peaks.tsv \
    --smiles-column SMILES --area-column "Peak Area" --output annotated.tsv
```

**Python API:**

```python
from povdb.annotator import PeakAnnotator

annotator = PeakAnnotator(model_dir="models")
result = annotator.annotate("peaks.tsv", smiles_column="SMILES",
                            area_column="Peak Area")
print(annotator.summarize(result))
```

The annotator adds three columns:

- `Is_Odorous` — odorous vs. odorless
- `Threshold_ppm` — predicted olfactory threshold
- `Odor_Contribution` — (peak area) / (threshold), the odour potential of each compound

---

## How the prediction model works

The manuscript describes a **two-step** prediction framework:

1. **Step 1 — binary classification.** An ANN (MACCS 167-bit fingerprint,
   hidden layer of 16 units) classifies each chemical as *odorous*
   (olfactory threshold < 1 ppm) or *odorless* (≥ 1 ppm).
2. **Step 2 — binning regression.** A Random Forest regressor trained on
   Morgan fingerprints (radius 2, 2048 bits) predicts the olfactory
   threshold bin (label 0–5).

The **shipped model** in this repository is the trained Random Forest
regressor, which directly outputs the threshold bin label:

```
RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10,
                      min_samples_leaf=1, bootstrap=True)
```

The physical threshold is recovered as:

```
threshold (ppm) = 10 ** (label − 5)
```

| Label | Threshold range (ppm) | Class |
|-------|------------------------|-------|
| 0 | 1e-5 – 1e-4 | odorous (extremely low) |
| 1 | 1e-4 – 1e-3 | odorous (very low) |
| 2 | 1e-3 – 1e-2 | odorous (low) |
| 3 | 1e-2 – 1e-1 | odorous (moderate) |
| 4 | 1e-1 – 1.0 | odorous (high) |
| 5 | ≥ 1.0 | **odorless** |

---

## Reproducing the model

```bash
# Train the Random Forest regressor on the full training set
python scripts/train_model.py --data data/5_data_quan.csv \
    --output models/best_model.joblib
```

### Training data

- **Primary data source:** olfactory thresholds compiled from the Japanese
  triangle odour bag method (Nagata, 1976–1988) and the petrochemical
  odour-threshold compilation (Hellman & Small, 1974), covering 207 odorous
  compounds (all ≥ 99.5% purity, evaluated by a trained panel).
- **Fuzzy hybrid integration** produced a final training set of **173 odorous
  compounds** (`data/5_data.csv`) plus **35 odorless compounds**
  (`data/5_data_quan.csv`, 208 records in total).
- Odorous threshold range: 7.7e-7 – 0.96 ppm.

### Performance

- Internal validation: **94.4%** true positive rate (odorous),
  **100.0%** true negative rate (odorless).
- External validation on 10 independent compounds (Leonardos et al.):
  **85.7% / 83.3%**.
- **Applicability domain:** a prediction is considered reliable only when the
  query molecule has a MACCS-fingerprint Tanimoto similarity **≥ 0.6** with at
  least **2** training compounds. Use `POVDBQuery.find_similar()` to check this.

---

## The POVDB library

The **Potential Odorous Virtual Database (POVDB)** contains **737,519
molecules** predicted odorous from a pool of ~110 million PubChem compounds,
each stored as an MSP spectral record with:

```
NAME, CID, EXACTMASS, FORMULA, INCHIKEY, SMILES,
RETENTIONTIMEINDEX, IONMODE, INSTRUMENTTYPE, COMMENT,
Num Peaks + virtual EI-MS (m/z, intensity)
```

Each entry includes a **virtual EI-MS spectrum**, validated against NIST17:
15,444 shared compounds achieved an average spectral dot-product of **0.82**.

The full library (~600 MB MSP) is available from the corresponding authors on
request (see [Data availability](#data-availability)). A 30-record sample is
shipped at `data/sample_povdb.msp` for format reference and testing.

### Building / cleaning the library

```bash
python scripts/build_povdb.py --input povdb.msp --output povdb_clean.msp \
    [--standardize]
```

This pipeline removes isotope-labelled records (SMILES containing `[13C]`,
`[2H]`, `[D]`, …) and optionally standardizes SMILES to the PubChem style
(largest fragment, charge neutralization, canonical tautomer).

---

## Non-targeted analysis workflow

A typical odor-source analysis using this hub:

```
GC-QTOF (Agilent 8890 + 7250) ──> MS-DIAL v4.90 deconvolution
    ──> NIST17 library search (score >= 70)
    ──> POVDB spectral matching (scripts/search_povdb.py --spectrum)
    ──> odor annotation (scripts/annotate_peaks.py)
    ──> odor contribution ranking: Σ (peak area / threshold)
```

---

## Repository layout

```
povdb_github/
├── povdb/                  # Core Python package
│   ├── __init__.py
│   ├── utils.py            # Fingerprints, SMILES standardization, MSP I/O
│   ├── predictor.py        # OdorPredictor (odorous + threshold)
│   ├── query.py            # POVDBQuery (spectral library search)
│   └── annotator.py        # PeakAnnotator (odor contribution)
├── scripts/
│   ├── train_model.py      # Train the Random Forest threshold model
│   ├── predict.py          # CLI odor prediction
│   ├── search_povdb.py     # CLI POVDB search
│   ├── annotate_peaks.py   # CLI peak-table annotation
│   └── build_povdb.py      # MSP library build / cleaning
├── data/
│   ├── 5_data.csv          # Training data: 173 odorous compounds
│   ├── 5_data_quan.csv     # Training data: 208 records (incl. 35 odorless)
│   └── sample_povdb.msp    # 30-record sample of the POVDB library
├── models/
│   └── best_model.joblib   # Pre-trained Random Forest model
├── examples/
│   ├── example_predict.py  # End-to-end walkthrough
│   └── example_data.csv    # Example SMILES batch
├── requirements.txt
├── LICENSE
└── .gitignore
```

---
