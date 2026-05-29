# Machine-Learning Package for xTB-to-ORCA/DFT Correction in Calixarene and Thiacalixarene Systems

## Overview

This package accompanies the manuscript on machine-learning correction of low-cost xTB-derived quantities toward reference values obtained from ORCA/DFT calculations for calixarene and thiacalixarene systems.

The primary modelling endpoints are:

- **HOMO energy** (eV)
- **LUMO energy** (eV)
- **Final Energy** (Eh)
These results are stored separately under `exploratory_final_energy/` because the available calculation workflow indicates that this field may include values generated under non-identical computational regimes. 

The package contains:

- processed full and pruned datasets;
- frozen split definitions and cross-validation split artifacts;
- reported model-selection and holdout-evaluation outputs;
- object-level predictions and diagnostic tables;
- serialized evaluation and deployment model bundles;
- analysis notebooks;
- the Python environment specification.

---

## Scientific Scope

The package supports three distinct evaluation protocols:

| Protocol | Dataset size | Train size | Test size | Groups in train/test | Interpretation |
|---|---:|---:|---:|---:|---|
| Full within-group interpolation | 266 | 211 | 55 | 22 / 22 | Primary benchmark: interpolation within chemical groups represented in training |
| Grouped unseen-groups split | 266 | 223 | 43 | 17 / 5 | Internal stress test: test chemical groups are absent from training |
| Pruning sensitivity analysis | 263 | 210 | 53 | 22 / 22 | Post hoc sensitivity analysis after removal of three high-impact records |

The pruned dataset removes the following records:

- `Group 11 / 129.log`
- `Group 11 / 135.log`
- `Group 13 / 174.log`

Their removal is used to quantify sensitivity of the reported metrics to selected high-impact records; it does not, by itself, justify permanent exclusion of these structures from the scientific dataset.

---

## Directory Structure

```text
ml_package/
├── data/
│   ├── calix_database_full.csv
│   └── calix_database_pruned.csv
│
├── environment/
│   └── requirements.txt
│
├── models/
│   ├── deployment/
│   │   ├── full_within_group/
│   │   ├── grouped_unseen_groups/
│   │   └── pruning_sensitivity/
│   ├── evaluation/
│   │   ├── full_within_group/
│   │   ├── grouped_unseen_groups/
│   │   └── pruning_sensitivity/
│   └── exploratory_final_energy/
│       ├── deployment/
│       └── evaluation/
│
├── notebooks/
│   ├── HOMO_LUMO_grouped.ipynb
│   └── HOMO_LUMO_within_group_holdout.ipynb
│
├── outputs/
│   ├── full_within_group/
│   │   ├── diagnostics/
│   │   ├── exploratory_final_energy/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── predictions/
│   ├── grouped_unseen_groups/
│   │   ├── diagnostics/
│   │   ├── exploratory_final_energy/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── predictions/
│   └── pruning_sensitivity/
│       ├── diagnostics/
│       ├── exploratory_final_energy/
│       ├── figures/
│       ├── metrics/
│       └── predictions/
│
└── splits/
    ├── cv/
    │   ├── full_within_group/
    │   ├── grouped_unseen_groups/
    │   └── pruning_sensitivity/
    └── main/
```

---

## Data Files

| File | Description | Intended use |
|---|---|---|
| `data/calix_database_full.csv` | Processed full dataset containing 266 molecular records assigned to 22 chemical groups | Source dataset for the full within-group and grouped unseen-groups analyses |
| `data/calix_database_pruned.csv` | Processed sensitivity dataset containing 263 molecular records after removal of three selected high-impact records | Source dataset for the pruning sensitivity analysis only |

The chemical-group labels define the validation structure used in this study. They represent chemically assigned groups used to distinguish interpolation within represented groups from evaluation on groups withheld from training.

---

## Main Evaluation Splits

The `splits/main/` directory contains the frozen train/test assignments used for the reported holdout evaluations.

| File | Description |
|---|---|
| `within_group_full_split.csv` | Record-level train/test assignment for the full within-group protocol |
| `within_group_full_split.json` | Metadata and indices for the full within-group frozen split |
| `within_group_full_group_summary.csv` | Group-level summary of the full within-group split |
| `grouped_split.csv` | Record-level train/test assignment for the grouped unseen-groups protocol |
| `grouped_split.json` | Metadata and indices for the grouped frozen split |
| `grouped_split_group_summary.csv` | Group-level summary of the grouped split |
| `within_group_pruned_split.csv` | Record-level train/test assignment for the pruned within-group protocol |
| `within_group_pruned_split.json` | Metadata and indices for the pruned frozen split |
| `within_group_pruned_group_summary.csv` | Group-level summary of the pruned within-group split |

The `splits/cv/` directory contains the split artifacts used during model comparison and hyperparameter selection:

- `full_within_group/`: repeated-holdout and inner-shuffle split artifacts for the full dataset;
- `grouped_unseen_groups/`: outer and inner grouped cross-validation split artifacts;
- `pruning_sensitivity/`: repeated-holdout and inner-shuffle split artifacts for the pruned dataset.

---

## Reported Primary Results: HOMO and LUMO

The primary results reported in the manuscript concern HOMO and LUMO correction. Holdout metrics for the selected models are provided in each protocol-specific `outputs/*/metrics/holdout_test_metrics.csv` file.

| Evaluation protocol | Target | Selected model | MAE (eV) | RMSE (eV) | R² | Q95 absolute error (eV) | Maximum absolute error (eV) | Test records |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Full within-group interpolation | HOMO | RandomForest | 0.1284 | 0.2772 | 0.7559 | 0.3856 | 1.7727 | 55 |
| Full within-group interpolation | LUMO | XGBoost | 0.1452 | 0.1923 | 0.9254 | 0.3316 | 0.5564 | 55 |
| Grouped unseen-groups split | HOMO | KernelRidge | 0.1575 | 0.2330 | 0.8570 | 0.4534 | 0.7868 | 43 |
| Grouped unseen-groups split | LUMO | Ridge | 0.1421 | 0.2075 | 0.8690 | 0.4557 | 0.5602 | 43 |
| Pruning sensitivity analysis | HOMO | RandomForest | 0.1050 | 0.1468 | 0.9040 | 0.3061 | 0.4624 | 53 |
| Pruning sensitivity analysis | LUMO | RandomForest | 0.1366 | 0.1925 | 0.9275 | 0.4285 | 0.5140 | 53 |

In the reported frozen holdouts, the selected HOMO and LUMO correction models produced smaller absolute errors than the uncorrected xTB baseline for all evaluated test records in all three protocols.

---

## Exploratory Final Energy Results

Results for the database field labelled `Final Energy` are stored in dedicated `exploratory_final_energy/` directories and are intentionally separated from the primary HOMO/LUMO results.

| Evaluation protocol | Selected model | MAE (Eh) | RMSE (Eh) | R² | Q95 absolute error (Eh) | Maximum absolute error (Eh) | Test records |
|---|---|---:|---:|---:|---:|---:|---:|
| Full within-group interpolation | RandomForest | 1384.0 | 2642.0 | 0.6987 | 6613.7 | 11032.4 | 55 |
| Grouped unseen-groups split | RandomForest | 1821.7 | 3290.6 | 0.2134 | 8477.4 | 9804.3 | 43 |
| Pruning sensitivity analysis | RandomForest | 1150.6 | 2184.8 | 0.7305 | 5609.5 | 7344.3 | 53 |

These values document the exploratory model behaviour for the stored database field. They should not be used to claim a uniform energy-correction model, thermodynamic ranking capability, or transferability of absolute energy predictions without a separate calculation-level audit of the target definition.

---

## Output Files

Each protocol-specific directory under `outputs/` contains the following file classes.

### `metrics/`

These files contain selected-model information, cross-validation summaries, frozen-holdout metrics, and model configuration metadata.

| File type | Description |
|---|---|
| `holdout_test_metrics.csv` | Final frozen-holdout metrics for selected models and the uncorrected xTB baseline |
| `model_comparison_final.csv` | Consolidated comparison of final model results |
| `final_model_selection.csv` | Selected model for each analysed target |
| `final_model_artifacts.json` | Stored configuration metadata for final model artifacts |
| `*_summary_main.csv` | Cross-validation summary for the primary HOMO/LUMO targets |
| `*_fold_metrics.csv` | Fold-level metrics supporting the cross-validation summary |
| `*_best_params.json` | Selected hyperparameter settings |
| `comparison_homo_lumo_*.csv` | Model-family comparison for HOMO and LUMO |

### `predictions/`

| File | Description |
|---|---|
| `test_results_hybrid_homo_lumo_energy.csv` | Record-level frozen-test predictions, reference values, baseline values, and residual information for the selected target models |

### `diagnostics/`

| File | Description |
|---|---|
| `homo_diagnostics.csv` | Object-level diagnostic information for HOMO prediction errors |
| `lumo_diagnostics.csv` | Object-level diagnostic information for LUMO prediction errors |

### `figures/`

| File | Description |
|---|---|
| `hybrid_model_parity_plots.png` | Parity plots generated for the corresponding evaluation protocol |

### `exploratory_final_energy/`

This directory contains Final Energy-specific model comparison and diagnostic files. These files are retained for transparency but are not treated as evidence for a homogeneous primary prediction endpoint.

---

## Model Artifacts

### Evaluation Models

The `models/evaluation/` directory contains serialized model and bundle files associated with the reported protocol-specific evaluations.

| Protocol | HOMO model | LUMO model |
|---|---|---|
| Full within-group interpolation | RandomForest | XGBoost |
| Grouped unseen-groups split | KernelRidge | Ridge |
| Pruning sensitivity analysis | RandomForest | RandomForest |

Evaluation model artifacts should be used when inspecting the reported model configurations and the corresponding held-out evaluation workflow.

### Deployment Bundles

The `models/deployment/` directory contains serialized bundles labelled as trained on all available data for the corresponding protocol/model configuration.

These bundles are intended for prospective prediction workflows. They are **not independent test evaluations** and must not be used to report generalization performance.

### Exploratory Final Energy Models

The `models/exploratory_final_energy/` directory contains evaluation and deployment model artifacts for the secondary Final Energy analysis. Their interpretation is subject to the methodological limitation stated above.

> **Security note:** `.pkl` files should be loaded only in a trusted Python environment and only when their origin is trusted.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/HOMO_LUMO_within_group_holdout.ipynb` | Analysis workflow associated with within-group evaluations, including the full-dataset benchmark and/or pruning sensitivity analysis as implemented in the supplied notebook |
| `notebooks/HOMO_LUMO_grouped.ipynb` | Analysis workflow associated with the grouped unseen-groups evaluation |

The exported split artifacts and output tables stored in this package are the records supporting the reported manuscript results.

Before rerunning the notebooks from this reorganized package, verify that their internal file paths reference:

```text
data/calix_database_full.csv
data/calix_database_pruned.csv
```

and that newly generated outputs are written to the intended protocol-specific subdirectories.

---

## Environment Setup

A Python environment specification is provided in:

```text
environment/requirements.txt
```

Example setup on Windows:

```cmd
cd /d "path\to\ml_package"

py -m venv .venv
call .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r environment\requirements.txt
```

To open the notebooks:

```cmd
jupyter notebook
```

If Jupyter Notebook is not installed by the environment specification, install it in the activated environment:

```cmd
python -m pip install notebook
jupyter notebook
```

Because the package includes serialized model artifacts, rerunning the notebooks and recording the exact software environment is recommended before reuse of the models outside the associated manuscript analysis.
