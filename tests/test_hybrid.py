from __future__ import annotations

import numpy as np
import pandas as pd

from hdb_price.hybrid import ResidualLocationEncoder


def test_location_encoder_is_finite_and_handles_unknown_values() -> None:
    frame = pd.DataFrame(
        {
            "address_key": ["A", "A", "B", "B", "C", "C"],
            "street_name": ["S1", "S1", "S1", "S2", "S2", "S3"],
            "block": ["1", "1", "2", "2", "3", "3"],
        }
    )
    target = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    encoder = ResidualLocationEncoder(
        columns=["address_key", "street_name", "block"],
        smoothing={"address_key": 10.0, "street_name": 20.0, "block": 20.0},
    )
    training = encoder.fit_transform(frame, target, folds=2, seed=42)
    unknown = encoder.transform(
        pd.DataFrame({"address_key": ["NEW"], "street_name": ["NEW"], "block": ["9"]})
    )
    assert training.shape == (6, 6)
    assert unknown.shape == (1, 6)
    assert np.isfinite(training).all()
    assert np.isfinite(unknown).all()

