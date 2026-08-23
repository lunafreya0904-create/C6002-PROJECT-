from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from hdb_price.config import load_config
from hdb_price.data import TargetTransformer, add_engineered_features, validate_data


def _minimal_frame() -> pd.DataFrame:
    rows = []
    for index, split in enumerate(("TRAIN", "VALIDATION", "TEST"), start=1):
        rows.append(
            {
                "record_id": index,
                "transaction_month_num": index,
                "floor_area_sqm": 90.0 + index,
                "storey_mid": 8.0,
                "remaining_lease_months": 800.0,
                "relative_floor": 0.5,
                "time_index": float(index),
                "max_floor_lvl": 16,
                "total_dwelling_units": 100,
                "town": "BEDOK",
                "flat_type": "4 ROOM",
                "flat_model": "MODEL A",
                "commercial_flag": 0,
                "market_hawker_flag": 0,
                "multistorey_carpark_flag": 1,
                "precinct_pavilion_flag": 0,
                "resale_price": 500_000.0 + index,
                "split_group": split,
            }
        )
    return pd.DataFrame(rows)


def test_engineered_month_features_are_cyclic() -> None:
    frame = pd.DataFrame({"transaction_month_num": [1, 4, 7, 10]})
    result = add_engineered_features(frame)
    radius = result["month_sin"] ** 2 + result["month_cos"] ** 2
    assert np.allclose(radius, 1.0)


def test_target_transform_round_trip() -> None:
    values = np.array([300_000.0, 500_000.0, 900_000.0])
    transformer = TargetTransformer.fit(values)
    restored = transformer.inverse(transformer.transform(values))
    assert np.allclose(restored, values, rtol=1e-5)


def test_data_validation_and_leakage_guard() -> None:
    config = load_config("configs/default.yaml")
    summary = validate_data(_minimal_frame(), config)
    assert summary["rows"] == 3

    leaked = deepcopy(config)
    leaked["data"]["numeric_features"].append("price_per_sqm")
    frame = _minimal_frame()
    frame["price_per_sqm"] = frame["resale_price"] / frame["floor_area_sqm"]
    with pytest.raises(ValueError, match="Leakage"):
        validate_data(frame, leaked)

