#!/usr/bin/env python3
"""Retrospective training-only model-family selection for the within-group analyses.

The script is located in the ``scripts`` directory and automatically resolves
the parent directory as the project root. It checks whether restricting HOMO
and LUMO model-family selection to the fixed training partitions changes the
selected model families.

This check supplements the original complete-data repeated comparisons and
does not replace them. It does not repeat the grouped analysis, deployment
fits, figures, or the exploratory Final Energy analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
try:
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError:  # Allows the dependency-free smoke test.
    optuna = None
    TPESampler = None

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:  # Reported clearly before a scientific run.
    XGBRegressor = None


RANDOM_STATE = 42
TEST_SIZE = 0.20
SELECTION_REPEATS = 5

FEATURES = [
    "HOMO_xtb (eV)",
    "LUMO_xtb (eV)",
    "HOMO-LUMO xtb (eV)",
    "Dipole_xtb (Debye)",
    "Total energy xtb (Eh)",
    "Number of Atoms",
    "energy_xtb_per_atom",
    "dipole_xtb_per_atom",
    "abs_homo_xtb",
    "abs_lumo_xtb",
    "dihedral_1_xtb_sin",
    "dihedral_1_xtb_cos",
    "dihedral_1_xtb_exists",
    "dihedral_2_xtb_sin",
    "dihedral_2_xtb_cos",
    "dihedral_2_xtb_exists",
    "dihedral_3_xtb_sin",
    "dihedral_3_xtb_cos",
    "dihedral_3_xtb_exists",
    "dihedral_4_xtb_sin",
    "dihedral_4_xtb_cos",
    "dihedral_4_xtb_exists",
]

MODEL_NAMES = [
    "LinearRegression",
    "Ridge",
    "ElasticNet",
    "KernelRidge",
    "RandomForest",
    "XGBoost",
]

MODEL_TRIALS = {
    "LinearRegression": 1,
    "Ridge": 40,
    "ElasticNet": 40,
    "KernelRidge": 40,
    "RandomForest": 60,
    "XGBoost": 80,
}

SCALED_MODELS = {"LinearRegression", "Ridge", "ElasticNet", "KernelRidge"}

PROTOCOLS = {
    "full": {
        "data": "data/calix_database_full.csv",
        "split": "splits/main/within_group_full_split.csv",
        "old_output": "outputs/full_within_group",
        "old_cv": "splits/cv/full_within_group",
        "old_models": "models/evaluation/full_within_group",
        "corrected_output": "outputs/corrected_selection/full_within_group",
        "expected_n": 266,
        "expected_train": 211,
        "expected_test": 55,
    },
    "pruned": {
        "data": "data/calix_database_pruned.csv",
        "split": "splits/main/within_group_pruned_split.csv",
        "old_output": "outputs/pruning_sensitivity",
        "old_cv": "splits/cv/pruning_sensitivity",
        "old_models": "models/evaluation/pruning_sensitivity",
        "corrected_output": "outputs/corrected_selection/pruning_sensitivity",
        "expected_n": 263,
        "expected_train": 210,
        "expected_test": 53,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_engineer(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";", encoding="cp1251", decimal=",")
    numeric_cols = [
        "dihedral_1_xtb",
        "dihedral_2_xtb",
        "dihedral_3_xtb",
        "dihedral_4_xtb",
        "Number of Atoms",
        "Dipole_xtb (Debye)",
        "Total energy xtb (Eh)",
        "HOMO_orca (eV)",
        "LUMO_orca (eV)",
        "HOMO_xtb (eV)",
        "LUMO_xtb (eV)",
        "HOMO-LUMO xtb (eV)",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in [f"dihedral_{i}_xtb" for i in range(1, 5)]:
        data[f"{column}_exists"] = data[column].notna().astype(int)
        angle = np.deg2rad(data[column])
        data[f"{column}_sin"] = np.sin(angle).fillna(0.0)
        data[f"{column}_cos"] = np.cos(angle).fillna(1.0)

    data["delta_homo"] = data["HOMO_orca (eV)"] - data["HOMO_xtb (eV)"]
    data["delta_lumo"] = data["LUMO_orca (eV)"] - data["LUMO_xtb (eV)"]
    data["energy_xtb_per_atom"] = (
        data["Total energy xtb (Eh)"] / data["Number of Atoms"]
    )
    data["dipole_xtb_per_atom"] = (
        data["Dipole_xtb (Debye)"] / data["Number of Atoms"]
    )
    data["abs_homo_xtb"] = data["HOMO_xtb (eV)"].abs()
    data["abs_lumo_xtb"] = data["LUMO_xtb (eV)"].abs()
    data["sample_id"] = data["Folder"].astype(str) + "/" + data["File"].astype(str)
    data["group_id"] = data["Folder"].astype(str)

    required = FEATURES + [
        "sample_id",
        "group_id",
        "Folder",
        "File",
        "SMILES",
        "delta_homo",
        "delta_lumo",
        "HOMO_orca (eV)",
        "LUMO_orca (eV)",
    ]
    dataset = data[required].dropna().copy()
    if dataset["sample_id"].duplicated().any():
        raise RuntimeError("sample_id is not unique after preprocessing")
    return dataset


def load_frozen_split(
    dataset: pd.DataFrame,
    split_path: Path,
    expected_train: int,
    expected_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = pd.read_csv(split_path)
    required = {"sample_id", "subset"}
    if not required.issubset(split.columns):
        raise RuntimeError(f"Missing columns in {split_path}: {sorted(required - set(split.columns))}")
    if split["sample_id"].duplicated().any():
        raise RuntimeError("Frozen split contains duplicate sample_id values")

    dataset_ids = set(dataset["sample_id"])
    split_ids = set(split["sample_id"])
    if dataset_ids != split_ids:
        raise RuntimeError(
            "Frozen split and dataset do not contain exactly the same sample_id values"
        )

    train_ids = set(split.loc[split["subset"] == "train", "sample_id"])
    test_ids = set(split.loc[split["subset"] == "test", "sample_id"])
    if train_ids & test_ids:
        raise RuntimeError("Frozen train/test overlap detected")
    if len(train_ids) != expected_train or len(test_ids) != expected_test:
        raise RuntimeError(
            f"Unexpected split size: train={len(train_ids)}, test={len(test_ids)}; "
            f"expected {expected_train}/{expected_test}"
        )

    train = dataset[dataset["sample_id"].isin(train_ids)].copy()
    test = dataset[dataset["sample_id"].isin(test_ids)].copy()
    return train, test, split


def make_within_group_splits(
    groups: pd.Series,
    n_repeats: int,
    test_size: float,
    random_state: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    groups = groups.reset_index(drop=True).astype(str)
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    all_groups = set(groups)

    for repeat in range(1, n_repeats + 1):
        rng = np.random.RandomState(random_state + repeat * 1000)
        train_positions: list[int] = []
        valid_positions: list[int] = []
        for group in sorted(all_groups):
            positions = np.flatnonzero(groups.to_numpy() == group).astype(int)
            rng.shuffle(positions)
            if len(positions) == 1:
                # A singleton group cannot be split internally.  Keep it in
                # training so every represented group remains available to
                # every candidate model; it contributes no selection-CV row.
                train_positions.extend(positions.tolist())
                continue
            n_valid = max(1, min(len(positions) - 1, int(round(len(positions) * test_size))))
            valid_positions.extend(positions[:n_valid].tolist())
            train_positions.extend(positions[n_valid:].tolist())

        train_pos = np.asarray(sorted(train_positions), dtype=int)
        valid_pos = np.asarray(sorted(valid_positions), dtype=int)
        if set(train_pos) & set(valid_pos):
            raise RuntimeError("Internal train/validation overlap detected")
        if set(groups.iloc[train_pos]) != all_groups:
            raise RuntimeError("Not all represented groups occur in internal training")
        splits.append((repeat, train_pos, valid_pos))
    return splits


def suggest_model_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "LinearRegression":
        return {}
    if model_name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}
    if model_name == "ElasticNet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if model_name == "KernelRidge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }
    raise ValueError(model_name)


def build_model(model_name: str, params: dict[str, Any], seed: int):
    if model_name == "LinearRegression":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(max_iter=100000, **params)
    if model_name == "ElasticNet":
        return ElasticNet(max_iter=100000, **params)
    if model_name == "KernelRidge":
        return KernelRidge(kernel="rbf", **params)
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise ModuleNotFoundError(
                "xgboost is required for the scientific model-family selection run"
            )
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    raise ValueError(model_name)


def fit_predict(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_pred: pd.DataFrame,
    model_name: str,
    params: dict[str, Any],
    seed: int,
):
    if model_name in SCALED_MODELS:
        scaler_x = StandardScaler()
        X_fit_used = scaler_x.fit_transform(X_fit)
        X_pred_used = scaler_x.transform(X_pred)
        scaler_y = StandardScaler()
        y_fit_used = scaler_y.fit_transform(y_fit.to_frame()).ravel()
        model = build_model(model_name, params, seed)
        model.fit(X_fit_used, y_fit_used)
        pred_scaled = np.asarray(model.predict(X_pred_used)).reshape(-1, 1)
        pred = scaler_y.inverse_transform(pred_scaled).ravel()
        return model, scaler_x, scaler_y, pred

    model = build_model(model_name, params, seed)
    model.fit(X_fit, y_fit)
    pred = np.asarray(model.predict(X_pred), dtype=float).ravel()
    return model, None, None, pred


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    absolute_error = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(np.median(absolute_error)),
        "Q90_AE": float(np.quantile(absolute_error, 0.90)),
        "Q95_AE": float(np.quantile(absolute_error, 0.95)),
        "Max_AE": float(np.max(absolute_error)),
        "n_samples": int(len(y_true)),
    }


def tune_family(
    train: pd.DataFrame,
    target_delta: str,
    true_column: str,
    xtb_column: str,
    model_name: str,
    n_trials: int,
    seed: int,
    splits: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    X = train[FEATURES]
    y = train[target_delta]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, model_name)
        scores = []
        for _, fit_pos, valid_pos in splits:
            _, _, _, pred_delta = fit_predict(
                X.iloc[fit_pos],
                y.iloc[fit_pos],
                X.iloc[valid_pos],
                model_name,
                params,
                seed,
            )
            scores.append(mean_absolute_error(y.iloc[valid_pos], pred_delta))
        return float(np.mean(scores))

    if model_name == "LinearRegression":
        best_params = {}
        best_value = objective(None)
    else:
        if optuna is None or TPESampler is None:
            raise ModuleNotFoundError(
                "optuna is required for the scientific model-family selection run"
            )
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params.copy()
        best_value = float(study.best_value)

    fold_rows: list[dict[str, Any]] = []
    for repeat, fit_pos, valid_pos in splits:
        _, _, _, pred_delta = fit_predict(
            X.iloc[fit_pos],
            y.iloc[fit_pos],
            X.iloc[valid_pos],
            model_name,
            best_params,
            seed,
        )
        valid = train.iloc[valid_pos]
        pred_absolute = valid[xtb_column].to_numpy() + pred_delta
        true_absolute = valid[true_column].to_numpy()
        row: dict[str, Any] = {
            "selection_scope": "frozen_training_partition_only",
            "repeat": repeat,
            "model": model_name,
            "target": "HOMO" if target_delta == "delta_homo" else "LUMO",
            "best_params_json": json.dumps(best_params, sort_keys=True),
            "optimization_mean_MAE": best_value,
        }
        row.update(metrics(true_absolute, pred_absolute))
        fold_rows.append(row)
    return best_params, best_value, fold_rows


def verify_old_artifacts(
    root: Path,
    config: dict[str, Any],
    train_ids: set[str],
    test_ids: set[str],
) -> dict[str, Any]:
    old_output = root / config["old_output"]
    old_cv = root / config["old_cv"]
    selection_path = old_output / "metrics/final_model_selection.csv"
    final_inner_path = old_cv / "final_inner_shuffle_split_artifacts.csv"
    predictions_path = old_output / "predictions/test_results_hybrid_homo_lumo_energy.csv"
    artifacts_path = old_output / "metrics/final_model_artifacts.json"

    for path in [selection_path, final_inner_path, predictions_path, artifacts_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    old_selection = pd.read_csv(selection_path)
    old_winners = dict(zip(old_selection["task"], old_selection["selected_model"]))
    final_inner = pd.read_csv(final_inner_path)
    predictions = pd.read_csv(predictions_path)
    inner_ids = set(final_inner["sample_id"].astype(str))
    prediction_ids = set(predictions["sample_id"].astype(str))

    if not inner_ids.issubset(train_ids) or inner_ids & test_ids:
        raise RuntimeError("Existing final tuning artifacts include frozen-test records")
    if prediction_ids != test_ids:
        raise RuntimeError("Existing prediction rows do not exactly match the frozen test")

    return {
        "old_winners": old_winners,
        "final_inner_ids_are_training_only": True,
        "old_prediction_ids_equal_frozen_test": True,
        "old_selection_sha256": sha256(selection_path),
        "old_final_inner_splits_sha256": sha256(final_inner_path),
        "old_predictions_sha256": sha256(predictions_path),
        "old_final_artifacts_sha256": sha256(artifacts_path),
        "old_selection_path": str(selection_path.relative_to(root)),
        "old_final_inner_splits_path": str(final_inner_path.relative_to(root)),
        "old_predictions_path": str(predictions_path.relative_to(root)),
        "old_final_artifacts_path": str(artifacts_path.relative_to(root)),
    }


def refit_changed_targets(
    root: Path,
    config: dict[str, Any],
    output_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    winners: dict[str, str],
    best_params: dict[tuple[str, str], dict[str, Any]],
    changed_targets: list[str],
) -> None:
    old_metrics_path = root / config["old_output"] / "metrics/holdout_test_metrics.csv"
    corrected_metrics = pd.read_csv(old_metrics_path)

    target_spec = {
        "HOMO": ("delta_homo", "HOMO_xtb (eV)", "HOMO_orca (eV)", RANDOM_STATE + 70000),
        "LUMO": ("delta_lumo", "LUMO_xtb (eV)", "LUMO_orca (eV)", RANDOM_STATE + 71000),
    }
    for target in changed_targets:
        delta_col, xtb_col, true_col, seed = target_spec[target]
        model_name = winners[target]
        params = best_params[(target, model_name)]
        model, scaler_x, scaler_y, pred_delta = fit_predict(
            train[FEATURES],
            train[delta_col],
            test[FEATURES],
            model_name,
            params,
            seed,
        )
        pred_absolute = test[xtb_col].to_numpy() + pred_delta
        true_absolute = test[true_col].to_numpy()

        bundle = {
            "model_name": model_name,
            "target_name": target,
            "selection_scope": "frozen_training_partition_only",
            "feature_columns": FEATURES,
            "best_params": params,
            "fit_sample_ids": train["sample_id"].tolist(),
            "model": model,
            "scaler_X": scaler_x,
            "scaler_y": scaler_y,
        }
        joblib.dump(bundle, output_dir / f"final_{model_name.lower()}_{target.lower()}_bundle.pkl")

        prediction_table = test[["sample_id", "group_id", "Folder", "File", "SMILES"]].copy()
        prediction_table[f"Calculated_{target}"] = true_absolute
        prediction_table[f"Baseline_{target}_xTB"] = test[xtb_col].to_numpy()
        prediction_table[f"Predicted_delta_{target}"] = pred_delta
        prediction_table[f"Predicted_{target}"] = pred_absolute
        prediction_table[f"Abs_Error_{target}"] = np.abs(pred_absolute - true_absolute)
        prediction_table.to_csv(
            output_dir / f"corrected_{target.lower()}_test_predictions.csv",
            index=False,
        )

        new_row: dict[str, Any] = {
            "evaluation_type": "holdout_test",
            "split_type": "frozen_within_group_holdout",
            "model": model_name,
            "target": target,
            "unit": "eV",
        }
        new_row.update(metrics(true_absolute, pred_absolute))
        corrected_metrics = corrected_metrics[
            ~(
                (corrected_metrics["target"] == target)
                & (corrected_metrics["model"] != "xTB baseline")
            )
        ]
        corrected_metrics = pd.concat([corrected_metrics, pd.DataFrame([new_row])], ignore_index=True)

    corrected_metrics.to_csv(output_dir / "holdout_test_metrics_corrected.csv", index=False)


def run_protocol(
    root: Path,
    protocol: str,
    repeats: int,
    smoke_test: bool,
    refit_changed: bool,
) -> dict[str, Any]:
    config = PROTOCOLS[protocol]
    data_path = root / config["data"]
    split_path = root / config["split"]
    output_dir = root / config["corrected_output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_and_engineer(data_path)
    if len(dataset) != config["expected_n"]:
        raise RuntimeError(f"Unexpected processed dataset size: {len(dataset)}")
    train, test, frozen_split = load_frozen_split(
        dataset,
        split_path,
        config["expected_train"],
        config["expected_test"],
    )
    train_ids = set(train["sample_id"])
    test_ids = set(test["sample_id"])
    training_group_sizes = train.groupby("group_id").size().sort_values()
    singleton_training_groups = training_group_sizes[training_group_sizes == 1].index.tolist()

    selection_splits = make_within_group_splits(
        train["group_id"],
        n_repeats=repeats,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    split_rows = []
    for repeat, fit_pos, valid_pos in selection_splits:
        for role, positions in [("train", fit_pos), ("valid", valid_pos)]:
            part = train.iloc[positions][["sample_id", "group_id", "Folder", "File", "SMILES"]].copy()
            part.insert(0, "split_role", role)
            part.insert(0, "repeat", repeat)
            part.insert(0, "split_level", "training_only_within_group_selection")
            split_rows.append(part)
    selection_split_table = pd.concat(split_rows, ignore_index=True)
    if set(selection_split_table["sample_id"]) & test_ids:
        raise RuntimeError("Frozen-test leakage in model-selection splits")
    selection_split_table.to_csv(output_dir / "training_only_selection_split_artifacts.csv", index=False)

    target_specs = {
        "HOMO": ("delta_homo", "HOMO_orca (eV)", "HOMO_xtb (eV)", RANDOM_STATE + 10000),
        "LUMO": ("delta_lumo", "LUMO_orca (eV)", "LUMO_xtb (eV)", RANDOM_STATE + 11000),
    }
    models = ["LinearRegression"] if smoke_test else MODEL_NAMES
    all_fold_rows: list[dict[str, Any]] = []
    best_params: dict[tuple[str, str], dict[str, Any]] = {}

    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    for target, (delta_col, true_col, xtb_col, target_seed) in target_specs.items():
        for model_index, model_name in enumerate(models):
            trials = 1 if smoke_test else MODEL_TRIALS[model_name]
            seed = target_seed + model_index * 1000
            print(
                f"[{protocol}] {target} | {model_name} | "
                f"{trials} trial(s), {repeats} training-only split(s)"
            )
            params, _, fold_rows = tune_family(
                train,
                delta_col,
                true_col,
                xtb_col,
                model_name,
                trials,
                seed,
                selection_splits,
            )
            best_params[(target, model_name)] = params
            all_fold_rows.extend(fold_rows)

    fold_metrics = pd.DataFrame(all_fold_rows)
    fold_metrics.to_csv(output_dir / "training_only_selection_fold_metrics.csv", index=False)
    comparison = (
        fold_metrics.groupby(["target", "model"], as_index=False)
        .agg(
            selection_MAE_mean=("MAE", "mean"),
            selection_MAE_std=("MAE", "std"),
            selection_RMSE_mean=("RMSE", "mean"),
            selection_R2_mean=("R2", "mean"),
            selection_Q95_AE_mean=("Q95_AE", "mean"),
            n_selection_splits=("repeat", "count"),
        )
        .sort_values(
            ["target", "selection_MAE_mean", "selection_Q95_AE_mean", "selection_R2_mean"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
    comparison.to_csv(output_dir / "training_only_model_comparison.csv", index=False)

    winners_table = comparison.groupby("target", sort=False).head(1).copy()
    winners = dict(zip(winners_table["target"], winners_table["model"]))
    winners_table["selection_scope"] = "frozen_training_partition_only"
    winners_table["performance_estimate"] = False
    winners_table["note"] = (
        "Internal selection criterion only; use frozen holdout for performance estimation"
    )
    winners_table.to_csv(output_dir / "training_only_final_model_selection.csv", index=False)

    audit = verify_old_artifacts(root, config, train_ids, test_ids)
    old_winners = audit["old_winners"]
    target_reuse = {
        target: bool(winners.get(target) == old_winners.get(target))
        for target in ["HOMO", "LUMO"]
        if target in winners
    }
    changed_targets = [target for target, reusable in target_reuse.items() if not reusable]

    manifest = {
        "protocol": protocol,
        "correction": "training-only model-family selection",
        "random_state": RANDOM_STATE,
        "selection_repeats": repeats,
        "selection_test_size": TEST_SIZE,
        "singleton_training_groups_retained_in_every_internal_train": singleton_training_groups,
        "dataset_n": len(dataset),
        "frozen_train_n": len(train),
        "frozen_test_n": len(test),
        "dataset_sha256": sha256(data_path),
        "frozen_split_sha256": sha256(split_path),
        "frozen_test_used_in_selection": False,
        "new_winners": winners,
        "old_winners": old_winners,
        "reuse_existing_final_artifact_by_target": target_reuse,
        "changed_targets_requiring_refit": changed_targets,
        "old_artifact_audit": audit,
        "final_energy_status": (
            "not rerun; existing full/pruned Final Energy selection must remain exploratory "
            "and must not be described as an independent frozen-holdout model-selection result"
        ),
        "smoke_test": smoke_test,
    }
    with (output_dir / "corrected_protocol_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    if changed_targets and refit_changed and not smoke_test:
        refit_changed_targets(
            root,
            config,
            output_dir,
            train,
            test,
            winners,
            best_params,
            changed_targets,
        )
    elif changed_targets and not smoke_test:
        print(
            f"[{protocol}] Winner changed for {changed_targets}. "
            "Run again with --refit-changed to fit only those targets."
        )
    elif not smoke_test:
        print(
            f"[{protocol}] HOMO/LUMO winners are unchanged. Existing final train-only "
            "models and holdout predictions can be reused; do not rerun deployment fitting."
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=(
            "Project root containing data, models, outputs, and splits "
            "(default: parent of the scripts directory)"
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=["full", "pruned", "both"],
        default="both",
    )
    parser.add_argument(
        "--selection-repeats",
        type=int,
        default=SELECTION_REPEATS,
    )
    parser.add_argument(
        "--refit-changed",
        action="store_true",
        help="Fit/evaluate only targets whose training-only winning family changed",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use LinearRegression and one trial only; not valid for scientific results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    protocols = ["full", "pruned"] if args.protocol == "both" else [args.protocol]
    repeats = 2 if args.smoke_test else args.selection_repeats
    if repeats < 2:
        raise ValueError("selection-repeats must be at least 2")

    results = []
    for protocol in protocols:
        results.append(
            run_protocol(
                root=root,
                protocol=protocol,
                repeats=repeats,
                smoke_test=args.smoke_test,
                refit_changed=args.refit_changed,
            )
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
