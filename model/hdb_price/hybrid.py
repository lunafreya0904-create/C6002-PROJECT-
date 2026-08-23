from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from .data import PreparedData
from .utils import write_json

LOGGER = logging.getLogger(__name__)


class ResidualLocationEncoder:
    """Leakage-safe smoothed encodings for high-cardinality location fields."""

    def __init__(self, columns: list[str], smoothing: dict[str, float]) -> None:
        self.columns = columns
        self.smoothing = smoothing
        self.target_maps: dict[str, dict[str, float]] = {}
        self.count_maps: dict[str, dict[str, int]] = {}

    @staticmethod
    def _series(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series("UNKNOWN", index=frame.index, dtype="string")
        return frame[column].fillna("UNKNOWN").astype(str)

    @staticmethod
    def _mapping(keys: pd.Series, target: np.ndarray, smoothing: float) -> dict[str, float]:
        table = pd.DataFrame({"key": keys.to_numpy(), "target": target})
        statistics = table.groupby("key")["target"].agg(["sum", "count"])
        encoded = statistics["sum"] / (statistics["count"] + smoothing)
        return {str(key): float(value) for key, value in encoded.items()}

    def fit_transform(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
        folds: int,
        seed: int,
    ) -> np.ndarray:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        encoded_columns: list[np.ndarray] = []
        for column in self.columns:
            keys = self._series(frame, column).reset_index(drop=True)
            smoothing = float(self.smoothing[column])
            out_of_fold = np.zeros(len(frame), dtype=np.float32)
            for fit_indices, holdout_indices in splitter.split(frame):
                mapping = self._mapping(
                    keys.iloc[fit_indices],
                    target[fit_indices],
                    smoothing,
                )
                out_of_fold[holdout_indices] = (
                    keys.iloc[holdout_indices].map(mapping).fillna(0.0).to_numpy(dtype=np.float32)
                )
            self.target_maps[column] = self._mapping(keys, target, smoothing)
            counts = keys.value_counts()
            self.count_maps[column] = {str(key): int(value) for key, value in counts.items()}
            log_frequency = np.log1p(keys.map(self.count_maps[column]).fillna(0)).to_numpy(
                dtype=np.float32
            )
            encoded_columns.extend((out_of_fold, log_frequency))
        return np.column_stack(encoded_columns).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        encoded_columns: list[np.ndarray] = []
        for column in self.columns:
            keys = self._series(frame, column)
            target_encoded = keys.map(self.target_maps[column]).fillna(0.0).to_numpy(
                dtype=np.float32
            )
            log_frequency = np.log1p(keys.map(self.count_maps[column]).fillna(0)).to_numpy(
                dtype=np.float32
            )
            encoded_columns.extend((target_encoded, log_frequency))
        return np.column_stack(encoded_columns).astype(np.float32)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for column in self.columns:
            names.extend((f"location__{column}_residual_te", f"location__{column}_log_count"))
        return names


@dataclass
class HybridBundle:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    train_oof_prediction: np.ndarray
    validation_prediction: np.ndarray
    test_prediction: np.ndarray

    @property
    def train_conditions(self) -> np.ndarray:
        return np.column_stack((self.x_train, self.train_oof_prediction)).astype(np.float32)

    @property
    def validation_conditions(self) -> np.ndarray:
        return np.column_stack((self.x_validation, self.validation_prediction)).astype(np.float32)

    @property
    def test_conditions(self) -> np.ndarray:
        return np.column_stack((self.x_test, self.test_prediction)).astype(np.float32)


def _hybrid_xgboost(config: dict[str, Any], n_estimators: int) -> XGBRegressor:
    hybrid = config["hybrid"]
    return XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device="cpu",
        n_estimators=n_estimators,
        learning_rate=float(hybrid["learning_rate"]),
        max_depth=int(hybrid["max_depth"]),
        min_child_weight=float(hybrid["min_child_weight"]),
        subsample=float(hybrid["subsample"]),
        colsample_bytree=float(hybrid["colsample_bytree"]),
        reg_alpha=float(hybrid["reg_alpha"]),
        reg_lambda=float(hybrid["reg_lambda"]),
        random_state=int(config["project"]["seed"]),
        n_jobs=-1,
    )


def train_hybrid(config: dict[str, Any], data: PreparedData, ridge: Ridge) -> HybridBundle:
    """Fit Ridge trend plus XGBoost residuals with OOF predictions for CFM."""
    hybrid_config = config["hybrid"]
    seed = int(config["project"]["seed"])
    artifacts = Path(config["project"]["artifacts_dir"])

    ridge_train = ridge.predict(data.x_train).astype(np.float32)
    ridge_validation = ridge.predict(data.x_validation).astype(np.float32)
    ridge_test = ridge.predict(data.x_test).astype(np.float32)
    ridge_residual = data.y_train - ridge_train

    encoder = ResidualLocationEncoder(
        columns=list(hybrid_config["high_cardinality_features"]),
        smoothing={key: float(value) for key, value in hybrid_config["smoothing"].items()},
    )
    location_train = encoder.fit_transform(
        data.train_frame,
        ridge_residual,
        folds=int(hybrid_config["oof_folds"]),
        seed=seed,
    )
    location_validation = encoder.transform(data.validation_frame)
    location_test = encoder.transform(data.test_frame)
    x_train = np.column_stack((data.x_train, location_train)).astype(np.float32)
    x_validation = np.column_stack((data.x_validation, location_validation)).astype(np.float32)
    x_test = np.column_stack((data.x_test, location_test)).astype(np.float32)

    final_model = _hybrid_xgboost(config, int(hybrid_config["n_estimators"]))
    final_model.set_params(early_stopping_rounds=int(hybrid_config["early_stopping_rounds"]))
    final_model.fit(
        x_train,
        ridge_residual,
        eval_set=[(x_validation, data.y_validation - ridge_validation)],
        verbose=False,
    )
    best_iteration = int(final_model.best_iteration)
    validation_prediction = ridge_validation + final_model.predict(x_validation)
    test_prediction = ridge_test + final_model.predict(x_test)

    splitter = KFold(
        n_splits=int(hybrid_config["oof_folds"]),
        shuffle=True,
        random_state=seed,
    )
    oof_residual_prediction = np.zeros(len(x_train), dtype=np.float32)
    for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(x_train), start=1):
        LOGGER.info("Training hybrid OOF fold %d/%d", fold, splitter.n_splits)
        fold_model = _hybrid_xgboost(config, best_iteration + 1)
        fold_model.fit(x_train[fit_indices], ridge_residual[fit_indices], verbose=False)
        oof_residual_prediction[holdout_indices] = fold_model.predict(x_train[holdout_indices])
    train_oof_prediction = ridge_train + oof_residual_prediction

    joblib.dump(encoder, artifacts / "hybrid_location_encoder.joblib")
    final_model.save_model(artifacts / "hybrid_xgboost.json")
    np.savez_compressed(
        artifacts / "hybrid_predictions.npz",
        train_oof_prediction=train_oof_prediction,
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
    )
    metadata = {
        "architecture": "ridge_trend_plus_xgboost_residual",
        "best_iteration": best_iteration,
        "condition_dimensions": int(x_train.shape[1] + 1),
        "location_features": encoder.feature_names(),
        "oof_folds": int(hybrid_config["oof_folds"]),
    }
    write_json(artifacts / "hybrid_metadata.json", metadata)
    return HybridBundle(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        train_oof_prediction=train_oof_prediction,
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
    )


def load_hybrid_bundle(config: dict[str, Any], data: PreparedData) -> HybridBundle:
    artifacts = Path(config["project"]["artifacts_dir"])
    encoder: ResidualLocationEncoder = joblib.load(artifacts / "hybrid_location_encoder.joblib")
    location_train = encoder.transform(data.train_frame)
    location_validation = encoder.transform(data.validation_frame)
    location_test = encoder.transform(data.test_frame)
    predictions = np.load(artifacts / "hybrid_predictions.npz")
    return HybridBundle(
        x_train=np.column_stack((data.x_train, location_train)).astype(np.float32),
        x_validation=np.column_stack((data.x_validation, location_validation)).astype(np.float32),
        x_test=np.column_stack((data.x_test, location_test)).astype(np.float32),
        train_oof_prediction=predictions["train_oof_prediction"].astype(np.float32),
        validation_prediction=predictions["validation_prediction"].astype(np.float32),
        test_prediction=predictions["test_prediction"].astype(np.float32),
    )


def predict_hybrid(
    config: dict[str, Any],
    base_features: np.ndarray,
    frame: pd.DataFrame,
    ridge: Ridge,
) -> tuple[np.ndarray, np.ndarray]:
    artifacts = Path(config["project"]["artifacts_dir"])
    encoder: ResidualLocationEncoder = joblib.load(artifacts / "hybrid_location_encoder.joblib")
    x_hybrid = np.column_stack((base_features, encoder.transform(frame))).astype(np.float32)
    model = XGBRegressor()
    model.load_model(artifacts / "hybrid_xgboost.json")
    prediction = ridge.predict(base_features) + model.predict(x_hybrid)
    conditions = np.column_stack((x_hybrid, prediction)).astype(np.float32)
    return prediction.astype(np.float32), conditions

