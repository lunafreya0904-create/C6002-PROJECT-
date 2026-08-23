from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .baselines import train_baselines
from .config import ensure_output_dirs, load_config
from .data import load_and_prepare, validate_data
from .evaluation import evaluate_models, predict_new_data
from .flow import train_flow
from .utils import configure_logging, set_global_seed, write_json

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CA6002 HDB price modelling")
    parser.add_argument(
        "command",
        choices=("validate-data", "train-baselines", "train-flow", "evaluate", "run-all", "predict"),
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", help="Input CSV for predict")
    parser.add_argument("--output", default="reports/new_predictions.csv")
    return parser


def _initialize(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_output_dirs(config)
    configure_logging()
    set_global_seed(int(config["project"]["seed"]))
    return config


def main() -> None:
    arguments = _parser().parse_args()
    config = _initialize(arguments.config)
    command = arguments.command

    if command == "validate-data":
        frame = pd.read_csv(config["project"]["data_path"])
        summary = validate_data(frame, config)
        write_json(Path(config["project"]["reports_dir"]) / "data_validation.json", summary)
        LOGGER.info("Data validation passed: %s", summary)
        return

    if command == "predict":
        if not arguments.input:
            raise SystemExit("--input is required for predict")
        output = Path(arguments.output)
        if not output.is_absolute():
            output = Path(config["project"]["root_dir"]) / output
        predict_new_data(config, arguments.input, output)
        LOGGER.info("Predictions written to %s", output)
        return

    data = load_and_prepare(config, persist=True)
    if command == "train-baselines":
        train_baselines(config, data)
    elif command == "train-flow":
        train_flow(config, data)
    elif command == "evaluate":
        metrics = evaluate_models(config, data)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    elif command == "run-all":
        train_baselines(config, data)
        train_flow(config, data)
        metrics = evaluate_models(config, data)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
