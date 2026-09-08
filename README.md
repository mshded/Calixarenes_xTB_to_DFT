# Machine Learning Correction of xTB-Calculated Frontier Orbital Energies in Calixarene and Thiacalixarene Systems

## Overview

This repository contains the data, machine-learning models, fixed data partitions, cross-validation splits, prediction outputs, diagnostic tables, and analysis notebooks associated with a study of **xTB-to-ORCA/DFT correction for calixarene and thiacalixarene derivatives**.

The study evaluates whether low-cost xTB-derived descriptors can be corrected toward reference quantum-chemical values for structurally diverse calixarene and thiacalixarene systems using classical machine-learning models.

The repository focuses on two primary prediction targets:

* **HOMO energy**, eV
* **LUMO energy**, eV

An additional target is provided as a separate exploratory analysis:

* **Final Energy**, Eh

Results for `Final Energy` are intentionally stored separately under `exploratory_final_energy/`, because interpretation of this field requires particular caution with respect to molecular size, structural complexity, and possible heterogeneity of the underlying calculation workflow.

---

## Dataset

The full dataset contains **266 molecular structures** assigned to **22 chemically defined groups** of calixarene and thiacalixarene derivatives.

The chemical space includes, among others:

* lower-rim unsubstituted calixarenes and thiacalixarenes;
* lower-rim protected thiacalixarenes;
* alkoxy- and bromoalkoxy-substituted derivatives;
* azo dye derivatives;
* pyrazole, terpyridine, and triazole derivatives;
* multithiacalixarene systems;
* oxyethylene, crown-ether, thioether, thioester, and thiol-containing derivatives.

Two processed datasets are provided:

| File                             | Records | Purpose                                                                                            |
| -------------------------------- | ------: | -------------------------------------------------------------------------------------------------- |
| `data/calix_database_full.csv`   |     266 | Full dataset used for the primary within-group benchmark and the grouped unseen-groups stress test |
| `data/calix_database_pruned.csv` |     263 | Sensitivity-analysis dataset obtained after removal of three influential records                    |

The pruned dataset excludes:

* `Group 11 / 129.log`
* `Group 11 / 135.log`
* `Group 13 / 174.log`

The pruned dataset is provided only for **sensitivity analysis**. Removal of these records improves several metrics, but does not by itself justify their permanent exclusion from the scientific dataset.

---

## Scientific Scope

This repository supports three distinct evaluation protocols.

| Protocol                        | Dataset size | Training size | Evaluation size | Groups in training/evaluation | Scientific interpretation                                                                                                                  |
| ------------------------------- | -----------: | ------------: | --------------: | ----------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Full within-group interpolation |          266 |           211 |              55 |                       22 / 22 | Primary benchmark based on ten repeated within-group 80/20 partitions; the fixed 211/55 partition was used for record-level diagnostics    |
| Grouped unseen-groups split     |          266 |           223 |              43 |                        17 / 5 | Internal grouped test for records from five chemical groups excluded from model selection and training                                     |
| Pruning sensitivity analysis    |          263 |           210 |              53 |                       22 / 22 | Post hoc sensitivity analysis after removal of three influential records; the fixed 210/53 partition was used for record-level diagnostics |

### Interpretation of the protocols

**Full within-group interpolation** is the principal practical scenario of this study. Model families were compared using ten repeated within-group 80/20 partitions of all 266 records. A fixed partition containing 211 training and 55 evaluation records was used for record-level predictions, error analysis, and comparison with the GFN2-xTB baseline. Because these 55 records had previously participated in the repeated model comparison, this partition is diagnostic rather than an independent test after model selection.

**Grouped unseen-groups split** is an internal grouped test. The 43 test records belong to five chemical groups excluded from model-family selection, preprocessing, hyperparameter optimization, and training. The results describe prediction for these five withheld groups within the available dataset and do not constitute external validation.

**Pruning sensitivity analysis** evaluates how strongly the benchmark metrics depend on three influential records. It is a post hoc sensitivity analysis and is not treated as the default dataset or the primary reported result. Its fixed evaluation partition differs from that of the full dataset, with 47 records shared between them; changes in their metrics are therefore descriptive rather than paired estimates of improvement.

---

## Repository Structure

```text
Calixarenes_xTB_to_DFT/
│
├── README.md
│
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
│   │
│   ├── evaluation/
│   │   ├── full_within_group/
│   │   ├── grouped_unseen_groups/
│   │   └── pruning_sensitivity/
│   │
│   └── exploratory_final_energy/
│       ├── deployment/
│       │   ├── full_within_group/
│       │   ├── grouped_unseen_groups/
│       │   └── pruning_sensitivity/
│       │
│       └── evaluation/
│           ├── full_within_group/
│           ├── grouped_unseen_groups/
│           └── pruning_sensitivity/
│
├── notebooks/
│   ├── HOMO_LUMO_grouped.ipynb
│   ├── HOMO_LUMO_within_group_holdout.ipynb
│   └── HOMO_LUMO_within_group_holdout_pruned.ipynb
│
├── scripts/
│   └── minimal_training_only_selection.py
│
├── outputs/
│   ├── full_within_group/
│   │   ├── diagnostics/
│   │   ├── exploratory_final_energy/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── predictions/
│   │
│   ├── grouped_unseen_groups/
│   │   ├── diagnostics/
│   │   ├── exploratory_final_energy/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── predictions/
│   │
│   ├── pruning_sensitivity/
│   │   ├── diagnostics/
│   │   ├── exploratory_final_energy/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── predictions/
│   │
│   └── corrected_selection/
│       ├── full_within_group/
│       └── pruning_sensitivity/
│
└── splits/
    ├── cv/
    │   ├── full_within_group/
    │   ├── grouped_unseen_groups/
    │   └── pruning_sensitivity/
    │
    └── main/
```

---

## Reported Primary Results: HOMO and LUMO

The principal results of the study concern correction of HOMO and LUMO energies.

### Fixed-partition performance

| Evaluation protocol             | Target | Selected model | MAE, eV | RMSE, eV |     R² | Q95 absolute error, eV | Maximum absolute error, eV | Evaluation records |
| ------------------------------- | ------ | -------------- | ------: | -------: | -----: | ---------------------: | -------------------------: | -----------: |
| Full within-group interpolation | HOMO   | RandomForest   |  0.1284 |   0.2772 | 0.7559 |                 0.3856 |                     1.7727 |           55 |
| Full within-group interpolation | LUMO   | XGBoost        |  0.1452 |   0.1923 | 0.9254 |                 0.3316 |                     0.5564 |           55 |
| Grouped unseen-groups split     | HOMO   | KernelRidge    |  0.1575 |   0.2330 | 0.8570 |                 0.4534 |                     0.7868 |           43 |
| Grouped unseen-groups split     | LUMO   | Ridge          |  0.1421 |   0.2075 | 0.8690 |                 0.4557 |                     0.5602 |           43 |
| Pruning sensitivity analysis    | HOMO   | RandomForest   |  0.1050 |   0.1468 | 0.9040 |                 0.3061 |                     0.4624 |           53 |
| Pruning sensitivity analysis    | LUMO   | RandomForest   |  0.1366 |   0.1925 | 0.9275 |                 0.4285 |                     0.5140 |           53 |

On all three fixed evaluation partitions, the selected HOMO and LUMO correction models produced smaller absolute errors than the uncorrected GFN2-xTB baseline for every evaluated record. The full and pruned within-group results are diagnostic, whereas the grouped results were obtained for five chemical groups excluded from model selection and training.

A retrospective check restricted HOMO and LUMO model-family selection to the corresponding fixed training partitions. It retained RandomForest for HOMO and XGBoost for LUMO on the full dataset. On the pruned dataset, it selected XGBoost for both targets; evaluation on the same 53 records gave MAEs of 0.1188 eV for HOMO and 0.1310 eV for LUMO. This check was not performed for `Final Energy` and does not replace the original pruned sensitivity results reported above.

### Main interpretation

* The **full within-group protocol** provides the primary practical evidence for prediction within already represented calixarene and thiacalixarene families.
* The **grouped unseen-groups protocol** shows that HOMO and LUMO correction remained effective for the five chemical groups excluded from model selection and training.
* The **pruning sensitivity protocol** shows that three influential records have a measurable effect on the error tail, especially for HOMO.

---

## Exploratory Final Energy Results

Results for the database field labelled `Final Energy` are provided as a separate exploratory analysis.

| Evaluation protocol             | Selected model | MAE, Eh | RMSE, Eh |     R² | Q95 absolute error, Eh | Maximum absolute error, Eh | Evaluation records |
| ------------------------------- | -------------- | ------: | -------: | -----: | ---------------------: | -------------------------: | -----------: |
| Full within-group interpolation | RandomForest   |  1384.0 |   2642.0 | 0.6987 |                 6613.7 |                    11032.4 |           55 |
| Grouped unseen-groups split     | RandomForest   |  1821.7 |   3290.6 | 0.2134 |                 8477.4 |                     9804.3 |           43 |
| Pruning sensitivity analysis    | RandomForest   |  1150.6 |   2184.8 | 0.7305 |                 5609.5 |                     7344.3 |           53 |

The `Final Energy` models reduce the error relative to the raw xTB baseline in the evaluated settings, but these results require a more conservative interpretation than the HOMO/LUMO models.

In particular:

* `Final Energy` is an extensive molecular quantity and is therefore strongly influenced by molecular size and composition;
* the largest and most structurally complex derivatives contribute substantially to the error tail;
* grouped prediction performance is notably weaker for `Final Energy` than for HOMO and LUMO;
* these models should not be interpreted as a universal replacement for reference-level energy calculations.

Accordingly, all corresponding artifacts are stored separately in:

```text
models/exploratory_final_energy/
outputs/*/exploratory_final_energy/
```

---

## Data Files

| File                             | Description                                                                               | Intended use                                         |
| -------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `data/calix_database_full.csv`   | Processed full dataset with 266 molecular records and 22 chemical groups                  | Full within-group and grouped unseen-groups analyses |
| `data/calix_database_pruned.csv` | Processed sensitivity dataset with 263 records after removal of three selected structures | Pruning sensitivity analysis only                    |

The chemical-group assignments define the validation design used in this work. They distinguish:

* interpolation within chemical families represented during training;
* evaluation on chemical groups withheld from training.

---

## Fixed Data Partitions and Cross-Validation Splits

The `splits/main/` directory contains the fixed data-partition assignments used for record-level evaluation. The full and pruned within-group partitions support diagnostic analyses and are not independent test sets after model-family selection. The grouped partition is a test partition because its five chemical groups were excluded from model selection and training.

| File                                    | Description                                                               |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `within_group_full_split.csv`           | Record-level training/evaluation assignment for the full within-group diagnostic partition |
| `within_group_full_split.json`          | Metadata and indices for the fixed full within-group diagnostic partition |
| `within_group_full_group_summary.csv`   | Group-level summary of the full within-group split                        |
| `grouped_split.csv`                     | Record-level train/test assignment for the grouped unseen-groups protocol |
| `grouped_split.json`                    | Metadata and indices for the fixed grouped test partition                 |
| `grouped_split_group_summary.csv`       | Group-level summary of the grouped split                                  |
| `within_group_pruned_split.csv`         | Record-level training/evaluation assignment for the pruned sensitivity analysis |
| `within_group_pruned_split.json`        | Metadata and indices for the fixed pruned diagnostic partition            |
| `within_group_pruned_group_summary.csv` | Group-level summary of the pruned within-group split                      |

The `splits/cv/` directory contains the cross-validation artifacts used during model comparison and model selection:

| Directory                          | Contents                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------- |
| `splits/cv/full_within_group/`     | Repeated-holdout and inner-shuffle split artifacts for the full dataset   |
| `splits/cv/grouped_unseen_groups/` | Outer and inner grouped cross-validation split artifacts                  |
| `splits/cv/pruning_sensitivity/`   | Repeated-holdout and inner-shuffle split artifacts for the pruned dataset |

These split artifacts are included to preserve the exact evaluation design underlying the reported results.

---

## Output Files

Each evaluation protocol has a corresponding directory under `outputs/`.

### `outputs/full_within_group/`

Reported outputs for the primary interpolation benchmark on the complete dataset.

### `outputs/grouped_unseen_groups/`

Reported outputs for the internal grouped test on five chemical groups excluded from model selection and training.

### `outputs/pruning_sensitivity/`

Reported outputs for the sensitivity analysis after removal of three diagnosed high-impact records.

### `outputs/corrected_selection/`

Retrospective HOMO and LUMO model-family selection restricted to the fixed training partitions. For the full dataset, the selected model families remained RandomForest for HOMO and XGBoost for LUMO. For the pruned dataset, the check selected XGBoost for both targets and includes the corresponding fixed-partition predictions and model bundles. These results supplement the original analyses and do not replace them. Final Energy was not included in this check.

The file `holdout_test_metrics_corrected.csv` combines the new pruned HOMO and LUMO results with unchanged exploratory Final Energy rows from the original pruned analysis. The Final Energy rows are included only for reference and are not results of the retrospective training-only model-selection check.

Within each protocol directory, outputs are organized as follows.

### `metrics/`

| File type                    | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| `holdout_test_metrics.csv`   | Fixed-partition metrics for the selected models and the xTB baseline |
| `model_comparison_final.csv` | Consolidated comparison of final candidate models                   |
| `final_model_selection.csv`  | Selected model for each analysed target                             |
| `final_model_artifacts.json` | Metadata associated with the stored final model artifacts           |
| `*_summary_main.csv`         | Cross-validation summaries for HOMO and LUMO                        |
| `*_fold_metrics.csv`         | Fold-level cross-validation metrics                                 |
| `*_best_params.json`         | Selected hyperparameter values                                      |
| `comparison_homo_lumo_*.csv` | Model-family comparisons for HOMO and LUMO                          |

### `predictions/`

| File                                       | Description                                                                                           |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `test_results_hybrid_homo_lumo_energy.csv` | Record-level fixed-partition predictions, reference values, xTB baseline values, and residual information |

### `diagnostics/`

| File                   | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `homo_diagnostics.csv` | Object-level diagnostic information for HOMO prediction errors |
| `lumo_diagnostics.csv` | Object-level diagnostic information for LUMO prediction errors |

### `figures/`

| File                            | Description                                            |
| ------------------------------- | ------------------------------------------------------ |
| `hybrid_model_parity_plots.png` | Parity plots for the selected protocol-specific models |

### `exploratory_final_energy/`

This directory contains energy-specific model comparisons and diagnostic outputs. These artifacts are retained for transparency and secondary analysis, but they are not treated as evidence for a uniformly transferable primary prediction endpoint.

---

## Model Artifacts

### Evaluation models

The `models/evaluation/` directory contains serialized model files and model bundles associated with the reported protocol-specific fixed-partition evaluations.

| Protocol                        | HOMO model   | LUMO model   |
| ------------------------------- | ------------ | ------------ |
| Full within-group interpolation | RandomForest | XGBoost      |
| Grouped unseen-groups split     | KernelRidge  | Ridge        |
| Pruning sensitivity analysis    | RandomForest | RandomForest |

Evaluation artifacts are intended for reproducing or examining the reported evaluation workflows. They should not be interpreted as models trained on the complete available dataset.

The RandomForest models listed for the pruned dataset correspond to the original post hoc sensitivity analysis. The retrospective training-only check selected XGBoost for both targets and is reported separately; it does not replace the original pruned results.

### Deployment models

The `models/deployment/` directory contains model bundles trained on all available records under the corresponding protocol-defined model configuration.

| Protocol-derived configuration      | HOMO deployment model | LUMO deployment model |
| ----------------------------------- | --------------------- | --------------------- |
| Full within-group configuration     | RandomForest          | XGBoost               |
| Grouped unseen-groups configuration | KernelRidge           | Ridge                 |
| Pruning sensitivity configuration   | RandomForest          | RandomForest          |

Deployment bundles are intended for prospective prediction workflows. Because they are trained on all available records, they must not be used to report independent generalization performance.

### Exploratory Final Energy models

The `models/exploratory_final_energy/` directory contains separate evaluation and deployment model artifacts for the exploratory `Final Energy` analysis.

These models are provided for transparency and limited exploratory use only. They should not be interpreted as validated general-purpose replacements for ORCA/DFT energy calculations.

> **Security note:** serialized `.pkl` files should only be loaded in a trusted Python environment and only when their origin is trusted.

---

## Notebooks

| Notebook                                                | Purpose                                                          |
| ------------------------------------------------------- | ---------------------------------------------------------------- |
| `notebooks/HOMO_LUMO_within_group_holdout.ipynb`        | Full within-group interpolation workflow on the complete dataset |
| `notebooks/HOMO_LUMO_grouped.ipynb`                     | Grouped unseen-groups evaluation workflow                        |
| `notebooks/HOMO_LUMO_within_group_holdout_pruned.ipynb` | Pruning sensitivity workflow on the reduced dataset              |

Before rerunning the notebooks, verify that the internal dataset paths correspond to the repository structure:

```text
data/calix_database_full.csv
data/calix_database_pruned.csv
```

Newly generated output files should be written to the corresponding protocol-specific directories under `outputs/`.

---

## Analysis Scripts

| Script | Purpose |
| ------ | ------- |
| `scripts/minimal_training_only_selection.py` | Retrospective HOMO and LUMO model-family selection restricted to the fixed training partitions |

The script automatically resolves the repository root from its location in the `scripts/` directory and writes the resulting artifacts to `outputs/corrected_selection/`. This check supplements the original complete-data repeated comparisons and does not replace them. It does not repeat the grouped analysis or the exploratory Final Energy analysis.

---

## Environment Setup

The Python environment specification is provided in:

```text
environment/requirements.txt
```

### Windows installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/mshded/Calixarenes_xTB_to_DFT.git
cd Calixarenes_xTB_to_DFT

py -m venv .venv
call .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r environment\requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

If Jupyter Notebook is not already installed in the environment, install it after activating the virtual environment:

```bash
python -m pip install notebook
jupyter notebook
```

---

## Recommended Model Use

The models in this repository should be selected according to the chemical scenario and the target property.

| Use case                                                                      | Recommended target | Recommended model source                   | Interpretation                                                               |
| ----------------------------------------------------------------------------- | ------------------ | ------------------------------------------ | ---------------------------------------------------------------------------- |
| New derivative belonging to a represented calixarene or thiacalixarene family | HOMO               | `models/deployment/full_within_group/`     | Preferred practical correction workflow                                      |
| New derivative belonging to a represented calixarene or thiacalixarene family | LUMO               | `models/deployment/full_within_group/`     | Preferred practical correction workflow                                      |
| Structure belonging to a chemical group not represented during training       | HOMO or LUMO       | `models/deployment/grouped_unseen_groups/` | Exploratory estimate informed by the internal five-group test; additional validation is required |
| Investigation of sensitivity to diagnosed high-impact records                 | HOMO or LUMO       | `models/deployment/pruning_sensitivity/`   | Sensitivity-analysis model, not the default model                            |
| Approximate exploration of the stored `Final Energy` field                    | Final Energy       | `models/exploratory_final_energy/`         | Exploratory estimate only; not a replacement for reference calculations      |

---

## Applicability Domain and Limitations

The models were developed specifically for the chemical space represented in this repository: calixarene and thiacalixarene derivatives distributed across 22 chemically assigned groups.

Predictions are most directly supported for structures that are chemically related to the represented families and that remain within a comparable range of molecular size and functional-group complexity.

Particular caution is required for:

* structures belonging to chemical groups absent from the training data;
* very large multithiacalixarene or multi-macrocyclic systems;
* molecules containing unusual combinations of heavy atoms or strongly electron-withdrawing peripheral fragments;
* structurally extreme compounds relative to the represented chemical families;
* prediction of `Final Energy`, especially across substantially different molecular sizes or structural classes;
* structures requiring interpretation beyond preliminary screening or prioritization.

The models are intended to support rapid screening and prioritization of candidate structures before more expensive reference-level quantum-chemical calculations. They should not be treated as universal replacements for ORCA/DFT calculations outside the represented chemical domain.

---

## Reproducibility Notes

This repository provides:

* processed datasets used in the reported analyses;
* fixed data-partition definitions;
* cross-validation split artifacts;
* object-level model predictions;
* model-selection tables and fixed-partition metrics;
* serialized evaluation and deployment model bundles;
* analysis notebooks;
* environment requirements.

The fixed-partition files, cross-validation artifacts, and stored output tables should be treated as the primary records supporting the reported numerical results.

When rerunning analyses, differences may arise from software versions, random-state handling, or changes in library implementations. For this reason, the supplied environment specification and split artifacts should be retained when reproducing the reported evaluation.

---

## Data and Code Availability

The processed datasets, fixed partition definitions, cross-validation splits, trained model artifacts, output tables, diagnostic results, and analysis notebooks supporting this study are available in this repository.

The full quantum-chemical source files and additional structural materials may be provided separately through the Supporting Information or upon reasonable request, depending on the final publication workflow.

---

## Citation

The associated manuscript is currently in preparation.

After publication, this section will be updated with the full bibliographic citation and DOI.

For use of the repository before publication, please cite the repository as:

```text
Betekhtin, A. A. Calixarenes_xTB_to_DFT: Machine Learning Correction of
xTB-Calculated Frontier Orbital Energies in Calixarene and Thiacalixarene
Systems. GitHub repository, 2026.
https://github.com/mshded/Calixarenes_xTB_to_DFT
```

---

## Repository Contact

For questions regarding the dataset, model artifacts, or analysis workflow, please contact the repository author through GitHub.
