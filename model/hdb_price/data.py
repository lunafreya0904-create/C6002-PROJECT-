from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TargetTransformer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "TargetTransformer":
        log_values = np.log1p(np.asarray(values, dtype=np.float64))
        std = float(log_values.std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Target standard deviation must be positive and finite")
        return cls(mean=float(log_values.mean()), std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.log1p(np.asarray(values, dtype=np.float64)) - self.mean) / self.std
        return result.astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        log_values = np.asarray(values, dtype=np.float64) * self.std + self.mean
        return np.maximum(np.expm1(log_values), 0.0).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "TargetTransformer":
        return cls(mean=float(values["mean"]), std=float(values["std"]))


@dataclass
class PreparedData:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    target_transformer: TargetTransformer
    feature_names: list[str]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "transaction_month_num" not in result.columns:
        raise ValueError("transaction_month_num is required for cyclic month features")
    angle = 2.0 * np.pi * (result["transaction_month_num"].astype(float) - 1.0) / 12.0
    result["month_sin"] = np.sin(angle)
    result["month_cos"] = np.cos(angle)
    return result


def model_features(config: dict[str, Any]) -> list[str]:
    data_config = config["data"]
    return (
        list(data_config["numeric_features"])
        + list(data_config["categorical_features"])
        + list(data_config["binary_features"])
    )


def validate_data(frame: pd.DataFrame, config: dict[str, Any], require_target: bool = True) -> dict[str, Any]:
    data_config = config["data"]
    required = set(model_features(config)) - {"month_sin", "month_cos"}
    required.add("transaction_month_num")
    if require_target:
        required.update({data_config["target"], data_config["split_column"]})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = set(model_features(config))
    leakage = selected.intersection(data_config["leakage_columns"])
    if leakage:
        raise ValueError(f"Leakage columns selected as predictors: {sorted(leakage)}")

    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_predictor_values": int(frame[list(required)].isna().sum().sum()),
    }
    if require_target:
        split_col = data_config["split_column"]
        expected = {
            data_config["train_label"],
            data_config["validation_label"],
            data_config["test_label"],
        }
        observed = set(frame[split_col].dropna().unique())
        if observed != expected:
            raise ValueError(f"Expected splits {sorted(expected)}, got {sorted(observed)}")
        if (frame[data_config["target"]] <= 0).any():
            raise ValueError("Target contains non-positive prices")
        summary["split_counts"] = {
            str(key): int(value) for key, value in frame[split_col].value_counts().items()
        }
        summary["target_min"] = float(frame[data_config["target"]].min())
        summary["target_max"] = float(frame[data_config["target"]].max())
    return summary


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    data_config = config["data"]
    numeric = list(data_config["numeric_features"])
    binary = list(data_config["binary_features"])
    categorical = list(data_config["categorical_features"])

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("binary", "passthrough", binary),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def load_and_prepare(config: dict[str, Any], persist: bool = True) -> PreparedData:
    frame = pd.read_csv(config["project"]["data_path"])
    validate_data(frame, config, require_target=True)
    frame = add_engineered_features(frame)

    data_config = config["data"]
    split_col = data_config["split_column"]
    train = frame.loc[frame[split_col] == data_config["train_label"]].copy()
    validation = frame.loc[frame[split_col] == data_config["validation_label"]].copy()
    test = frame.loc[frame[split_col] == data_config["test_label"]].copy()
    features = model_features(config)

    preprocessor = build_preprocessor(config)
    x_train = np.asarray(preprocessor.fit_transform(train[features]), dtype=np.float32)
    x_validation = np.asarray(preprocessor.transform(validation[features]), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(test[features]), dtype=np.float32)

    target_col = data_config["target"]
    target_transformer = TargetTransformer.fit(train[target_col].to_numpy())
    y_train = target_transformer.transform(train[target_col].to_numpy())
    y_validation = target_transformer.transform(validation[target_col].to_numpy())
    y_test = target_transformer.transform(test[target_col].to_numpy())
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]

    if not all(np.isfinite(values).all() for values in (x_train, x_validation, x_test)):
        raise ValueError("Preprocessed predictors contain NaN or Inf")
    if persist:
        artifacts = Path(config["project"]["artifacts_dir"])
        artifacts.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, artifacts / "preprocessor.joblib")
        joblib.dump(target_transformer, artifacts / "target_transformer.joblib")
        joblib.dump(feature_names, artifacts / "feature_names.joblib")

    return PreparedData(
        train_frame=train,
        validation_frame=validation,
        test_frame=test,
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        preprocessor=preprocessor,
        target_transformer=target_transformer,
        feature_names=feature_names,
    )


def transform_new_data(
    frame: pd.DataFrame,
    config: dict[str, Any],
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    validate_data(frame, config, require_target=False)
    enriched = add_engineered_features(frame)
    return np.asarray(preprocessor.transform(enriched[model_features(config)]), dtype=np.float32)

