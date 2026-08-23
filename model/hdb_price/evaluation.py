from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .baselines import load_baselines
from .data import PreparedData, load_and_prepare
from .flow import load_flow_model, resolve_device, sample_flow
from .hybrid import load_hybrid_bundle, predict_hybrid
from .utils import write_json

LOGGER = logging.getLogger(__name__)


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return interpretable point-regression accuracy metrics."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "r2": float(r2_score(actual, predicted)),
    }


def ensemble_crps(actual: np.ndarray, samples: np.ndarray) -> float:
    """Return ensemble CRPS averaged over observations."""
    actual = np.asarray(actual, dtype=np.float64)
    sorted_samples = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    sample_count = sorted_samples.shape[1]
    first_term = np.mean(np.abs(sorted_samples - actual[:, None]), axis=1)
    coefficients = 2.0 * np.arange(1, sample_count + 1) - sample_count - 1.0
    second_term = np.sum(sorted_samples * coefficients[None, :], axis=1) / sample_count**2
    return float(np.mean(first_term - second_term))


def interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Return the fraction of targets contained in a prediction interval."""
    actual = np.asarray(actual)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def conformal_adjustment(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    scores = np.maximum.reduce((lower - actual, actual - upper, np.zeros_like(actual)))
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, quantile_level, method="higher"))


def _inverse_prediction(data: PreparedData, prediction: np.ndarray) -> np.ndarray:
    return data.target_transformer.inverse(np.asarray(prediction))


def evaluate_models(config: dict[str, Any], data: PreparedData | None = None) -> dict[str, Any]:
    """Evaluate fitted models and return metric values only.

    Besides persisting ``metrics.json`` and the calibration parameters required
    by explicit inference, this function has no result-generation side effects.
    """
    if data is None:
        data = load_and_prepare(config, persist=False)
    artifacts = Path(config["project"]["artifacts_dir"])
    reports = Path(config["project"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    ridge, xgboost, baseline_metadata = load_baselines(artifacts)

    actual = data.test_frame[config["data"]["target"]].to_numpy(dtype=np.float64)
    median_prediction = np.full_like(actual, float(baseline_metadata["median_price"]))
    ridge_prediction = _inverse_prediction(data, ridge.predict(data.x_test))
    xgboost_prediction = _inverse_prediction(data, xgboost.predict(data.x_test))
    hybrid = load_hybrid_bundle(config, data)
    hybrid_prediction = _inverse_prediction(data, hybrid.test_prediction)

    device = resolve_device(str(config["flow"]["device"]))
    flow_model, residual_transformer, flow_target_transformer = load_flow_model(
        artifacts / "flow_best.pt", device
    )
    flow_config = config["flow"]
    validation_samples = sample_flow(
        flow_model,
        hybrid.validation_conditions,
        hybrid.validation_prediction,
        residual_transformer,
        flow_target_transformer,
        n_samples=int(flow_config["validation_samples"]),
        steps=int(flow_config["sampling_steps"]),
        batch_rows=int(flow_config["prediction_batch_rows"]),
        device=device,
        seed=int(config["project"]["seed"]) + 100,
    )
    test_samples = sample_flow(
        flow_model,
        hybrid.test_conditions,
        hybrid.test_prediction,
        residual_transformer,
        flow_target_transformer,
        n_samples=int(flow_config["test_samples"]),
        steps=int(flow_config["sampling_steps"]),
        batch_rows=int(flow_config["prediction_batch_rows"]),
        device=device,
        seed=int(config["project"]["seed"]) + 200,
    )

    validation_actual = data.validation_frame[config["data"]["target"]].to_numpy(dtype=np.float64)
    validation_raw_median = np.median(validation_samples, axis=1)
    median_log_correction = float(
        np.median(np.log1p(validation_actual) - np.log1p(validation_raw_median))
    )
    validation_samples = np.expm1(np.log1p(validation_samples) + median_log_correction)
    test_samples = np.expm1(np.log1p(test_samples) + median_log_correction)

    alpha = float(config["evaluation"]["interval_alpha"])
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0
    validation_lower = np.quantile(validation_samples, lower_quantile, axis=1)
    validation_upper = np.quantile(validation_samples, upper_quantile, axis=1)
    adjustment = conformal_adjustment(
        validation_actual,
        validation_lower,
        validation_upper,
        alpha,
    )

    cfm_lower = np.quantile(test_samples, lower_quantile, axis=1)
    cfm_median = np.median(test_samples, axis=1)
    cfm_upper = np.quantile(test_samples, upper_quantile, axis=1)
    calibrated_lower = np.maximum(cfm_lower - adjustment, 0.0)
    calibrated_upper = cfm_upper + adjustment

    metrics = {
        "point_metrics": {
            "training_median": point_metrics(actual, median_prediction),
            "ridge": point_metrics(actual, ridge_prediction),
            "xgboost": point_metrics(actual, xgboost_prediction),
            "ridge_xgboost_residual_hybrid": point_metrics(actual, hybrid_prediction),
            "conditional_flow_matching_median": point_metrics(actual, cfm_median),
        },
        "probabilistic_metrics": {
            "conditional_flow_matching": {
                "crps": ensemble_crps(actual, test_samples),
                "raw_80_percent_interval_coverage": interval_coverage(
                    actual,
                    cfm_lower,
                    cfm_upper,
                ),
                "conformal_80_percent_interval_coverage": interval_coverage(
                    actual,
                    calibrated_lower,
                    calibrated_upper,
                ),
                "validation_median_log_correction": median_log_correction,
            }
        },
    }
    write_json(reports / "metrics.json", metrics)
    write_json(
        artifacts / "calibration.json",
        {
            "alpha": alpha,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "conformal_adjustment_sgd": adjustment,
            "median_log_correction": median_log_correction,
            "calibration_split": config["data"]["validation_label"],
        },
    )
    LOGGER.info(
        "Evaluation complete; Ridge MAE %.2f, XGBoost MAE %.2f, hybrid MAE %.2f, CFM MAE %.2f",
        metrics["point_metrics"]["ridge"]["mae"],
        metrics["point_metrics"]["xgboost"]["mae"],
        metrics["point_metrics"]["ridge_xgboost_residual_hybrid"]["mae"],
        metrics["point_metrics"]["conditional_flow_matching_median"]["mae"],
    )
    return metrics


def predict_new_data(
    config: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Run explicit inference and write transaction-level predictions."""
    frame = pd.read_csv(input_path)
    artifacts = Path(config["project"]["artifacts_dir"])
    preprocessor = joblib.load(artifacts / "preprocessor.joblib")
    from .data import transform_new_data

    base_features = transform_new_data(frame, config, preprocessor)
    ridge, xgboost, _ = load_baselines(artifacts)
    hybrid_prediction, conditions = predict_hybrid(
        config,
        base_features,
        frame,
        ridge,
    )
    device = resolve_device(str(config["flow"]["device"]))
    model, residual_transformer, transformer = load_flow_model(
        artifacts / "flow_best.pt", device
    )
    flow_config = config["flow"]
    samples = sample_flow(
        model,
        conditions,
        hybrid_prediction,
        residual_transformer,
        transformer,
        n_samples=int(flow_config["test_samples"]),
        steps=int(flow_config["sampling_steps"]),
        batch_rows=int(flow_config["prediction_batch_rows"]),
        device=device,
        seed=int(config["project"]["seed"]) + 300,
    )
    from .utils import read_json

    calibration_path = artifacts / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError("Run the evaluate command before predict to create calibration.json")
    calibration = read_json(calibration_path)
    samples = np.expm1(
        np.log1p(samples) + float(calibration.get("median_log_correction", 0.0))
    )
    lower = np.quantile(samples, float(calibration["lower_quantile"]), axis=1)
    upper = np.quantile(samples, float(calibration["upper_quantile"]), axis=1)
    adjustment = float(calibration["conformal_adjustment_sgd"])

    result = frame.copy()
    result["ridge_prediction"] = transformer.inverse(ridge.predict(base_features))
    result["xgboost_prediction"] = transformer.inverse(xgboost.predict(base_features))
    result["ridge_xgboost_hybrid"] = transformer.inverse(hybrid_prediction)
    result["cfm_q10"] = lower
    result["cfm_median"] = np.median(samples, axis=1)
    result["cfm_q90"] = upper
    result["calibrated_lower"] = np.maximum(lower - adjustment, 0.0)
    result["calibrated_upper"] = upper + adjustment
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
