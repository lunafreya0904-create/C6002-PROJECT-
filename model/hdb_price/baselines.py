from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from .data import PreparedData
from .utils import write_json

LOGGER = logging.getLogger(__name__)


def _xgboost_device(config: dict[str, Any]) -> str:
    requested = str(config["xgboost"].get("device", "auto"))
    if requested != "auto":
        return requested
    # The current Windows wheel uses a CUDA runtime newer than the installed
    # 572-series driver. Prefer the reliable CPU path instead of allowing a
    # silent CUDA-to-CPU fallback with misleading metadata.
    if platform.system() == "Windows":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _original_price_mae(data: PreparedData, truth: np.ndarray, prediction: np.ndarray) -> float:
    actual = data.target_transformer.inverse(truth)
    predicted = data.target_transformer.inverse(prediction)
    return float(mean_absolute_error(actual, predicted))


def train_baselines(config: dict[str, Any], data: PreparedData) -> dict[str, Any]:
    """Fit the median, Ridge and tuned XGBoost baseline models."""
    seed = int(config["project"]["seed"])
    artifacts = Path(config["project"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)

    train_prices = data.train_frame[config["data"]["target"]].to_numpy()
    median_price = float(np.median(train_prices))

    ridge_results: list[dict[str, float]] = []
    best_ridge: Ridge | None = None
    best_ridge_mae = float("inf")
    for alpha in config["ridge"]["alphas"]:
        model = Ridge(alpha=float(alpha))
        model.fit(data.x_train, data.y_train)
        val_prediction = model.predict(data.x_validation)
        val_mae = _original_price_mae(data, data.y_validation, val_prediction)
        ridge_results.append({"alpha": float(alpha), "validation_mae": val_mae})
        if val_mae < best_ridge_mae:
            best_ridge = model
            best_ridge_mae = val_mae
    if best_ridge is None:
        raise RuntimeError("No Ridge model was trained")
    joblib.dump(best_ridge, artifacts / "ridge.joblib")

    xgb_config = config["xgboost"]
    distributions = {
        "learning_rate": xgb_config["learning_rates"],
        "max_depth": xgb_config["max_depths"],
        "min_child_weight": xgb_config["min_child_weights"],
        "subsample": xgb_config["subsamples"],
        "colsample_bytree": xgb_config["colsample_bytrees"],
        "reg_alpha": xgb_config["reg_alphas"],
        "reg_lambda": xgb_config["reg_lambdas"],
    }
    candidates = list(
        ParameterSampler(
            distributions,
            n_iter=int(xgb_config["tune_trials"]),
            random_state=seed,
        )
    )
    device = _xgboost_device(config)
    LOGGER.info("Training %d XGBoost candidates on %s", len(candidates), device)
    xgb_results: list[dict[str, Any]] = []
    best_xgb: XGBRegressor | None = None
    best_xgb_mae = float("inf")
    for index, parameters in enumerate(candidates, start=1):
        common = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "device": device,
            "n_estimators": int(xgb_config["n_estimators"]),
            "early_stopping_rounds": int(xgb_config["early_stopping_rounds"]),
            "random_state": seed,
            "n_jobs": -1,
        }
        model = XGBRegressor(**common, **parameters)
        try:
            model.fit(
                data.x_train,
                data.y_train,
                eval_set=[(data.x_validation, data.y_validation)],
                verbose=False,
            )
        except Exception as error:
            if device != "cuda":
                raise
            LOGGER.warning("CUDA XGBoost failed (%s); retrying candidate on CPU", error)
            device = "cpu"
            common["device"] = "cpu"
            model = XGBRegressor(**common, **parameters)
            model.fit(
                data.x_train,
                data.y_train,
                eval_set=[(data.x_validation, data.y_validation)],
                verbose=False,
            )
        val_prediction = model.predict(data.x_validation)
        val_mae = _original_price_mae(data, data.y_validation, val_prediction)
        result = {
            "trial": index,
            "validation_mae": val_mae,
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            **parameters,
        }
        xgb_results.append(result)
        LOGGER.info("XGBoost %d/%d validation MAE: %.2f", index, len(candidates), val_mae)
        if val_mae < best_xgb_mae:
            best_xgb = model
            best_xgb_mae = val_mae
    if best_xgb is None:
        raise RuntimeError("No XGBoost model was trained")
    best_xgb.save_model(artifacts / "xgboost.json")

    from .hybrid import train_hybrid

    hybrid_bundle = train_hybrid(config, data, best_ridge)
    hybrid_validation_mae = _original_price_mae(
        data,
        data.y_validation,
        hybrid_bundle.validation_prediction,
    )

    metadata = {
        "median_price": median_price,
        "ridge_validation_results": ridge_results,
        "best_ridge_validation_mae": best_ridge_mae,
        "xgboost_device": device,
        "xgboost_validation_results": xgb_results,
        "best_xgboost_validation_mae": best_xgb_mae,
        "hybrid_validation_mae": hybrid_validation_mae,
    }
    write_json(artifacts / "baselines_metadata.json", metadata)
    LOGGER.info(
        "Best Ridge MAE %.2f; XGBoost MAE %.2f; hybrid MAE %.2f",
        best_ridge_mae,
        best_xgb_mae,
        hybrid_validation_mae,
    )
    return metadata


def load_baselines(artifacts_dir: str | Path) -> tuple[Ridge, XGBRegressor, dict[str, Any]]:
    artifacts = Path(artifacts_dir)
    ridge = joblib.load(artifacts / "ridge.joblib")
    xgboost = XGBRegressor()
    xgboost.load_model(artifacts / "xgboost.json")
    from .utils import read_json

    metadata = read_json(artifacts / "baselines_metadata.json")
    return ridge, xgboost, metadata
