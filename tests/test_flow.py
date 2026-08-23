from __future__ import annotations

import numpy as np
import torch

from hdb_price.data import TargetTransformer
from hdb_price.flow import (
    ConditionalVelocityNetwork,
    ResidualStandardizer,
    _flow_loss,
    sample_flow,
)


def test_flow_forward_and_gradients_are_finite() -> None:
    model = ConditionalVelocityNetwork(
        condition_dim=5,
        hidden_dims=[16, 16],
        time_embedding_dim=8,
        dropout=0.0,
    )
    condition = torch.randn(12, 5)
    target = torch.randn(12, 1)
    time_value = torch.rand(12, 1)
    noise = torch.randn(12, 1)
    loss = _flow_loss(model, condition, target, time_value, noise)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_sampler_shape_and_reproducibility() -> None:
    model = ConditionalVelocityNetwork(
        condition_dim=3,
        hidden_dims=[8],
        time_embedding_dim=8,
        dropout=0.0,
    ).eval()
    transformer = TargetTransformer.fit(np.array([300_000.0, 500_000.0, 700_000.0]))
    residual_transformer = ResidualStandardizer.fit(np.array([-0.1, 0.0, 0.1]))
    conditions = np.zeros((4, 3), dtype=np.float32)
    base_prediction = np.zeros(4, dtype=np.float32)
    first = sample_flow(
        model,
        conditions,
        base_prediction,
        residual_transformer,
        transformer,
        n_samples=5,
        steps=3,
        batch_rows=2,
        device=torch.device("cpu"),
        seed=42,
        show_progress=False,
    )
    second = sample_flow(
        model,
        conditions,
        base_prediction,
        residual_transformer,
        transformer,
        n_samples=5,
        steps=3,
        batch_rows=2,
        device=torch.device("cpu"),
        seed=42,
        show_progress=False,
    )
    assert first.shape == (4, 5)
    assert np.isfinite(first).all()
    assert np.all(first >= 0)
    assert np.allclose(first, second)
