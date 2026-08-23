from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import PreparedData, TargetTransformer
from .utils import write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class ResidualStandardizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "ResidualStandardizer":
        array = np.asarray(values, dtype=np.float64)
        std = float(array.std())
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Residual standard deviation must be positive and finite")
        return cls(mean=float(array.mean()), std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values) - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "ResidualStandardizer":
        return cls(mean=float(values["mean"]), std=float(values["std"]))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2 != 0:
            raise ValueError("Time embedding dimension must be even and at least four")
        self.dimension = dimension

    def forward(self, time_value: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000.0) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=time_value.device, dtype=time_value.dtype) * -scale
        )
        angles = time_value * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class ConditionalVelocityNetwork(nn.Module):
    """Velocity field v(y_t, t, condition) for one-dimensional price flow."""

    def __init__(
        self,
        condition_dim: int,
        hidden_dims: list[int],
        time_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.hidden_dims = list(hidden_dims)
        self.time_embedding_dim = time_embedding_dim
        self.dropout = dropout
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)

        layers: list[nn.Module] = []
        input_dim = condition_dim + time_embedding_dim + 1
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        price_state: torch.Tensor,
        time_value: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if price_state.ndim == 1:
            price_state = price_state.unsqueeze(-1)
        if time_value.ndim == 1:
            time_value = time_value.unsqueeze(-1)
        embedded_time = self.time_embedding(time_value)
        return self.network(torch.cat((price_state, embedded_time, condition), dim=-1))


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return device


def _flow_loss(
    model: ConditionalVelocityNetwork,
    condition: torch.Tensor,
    target: torch.Tensor,
    time_value: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    interpolated = (1.0 - time_value) * noise + time_value * target
    target_velocity = target - noise
    predicted_velocity = model(interpolated, time_value, condition)
    loss = torch.mean((predicted_velocity - target_velocity) ** 2)
    if not torch.isfinite(loss):
        raise FloatingPointError("Flow loss became NaN or Inf")
    return loss


def train_flow(config: dict[str, Any], data: PreparedData) -> dict[str, Any]:
    flow_config = config["flow"]
    seed = int(config["project"]["seed"])
    torch.manual_seed(seed)
    device = resolve_device(str(flow_config["device"]))
    LOGGER.info("Training Conditional Flow Matching on %s", device)

    from .hybrid import load_hybrid_bundle

    hybrid = load_hybrid_bundle(config, data)
    train_residual = data.y_train - hybrid.train_oof_prediction
    validation_residual = data.y_validation - hybrid.validation_prediction
    residual_transformer = ResidualStandardizer.fit(train_residual)
    flow_train_target = residual_transformer.transform(train_residual)
    flow_validation_target = residual_transformer.transform(validation_residual)

    model_kwargs = {
        "condition_dim": int(hybrid.train_conditions.shape[1]),
        "hidden_dims": [int(value) for value in flow_config["hidden_dims"]],
        "time_embedding_dim": int(flow_config["time_embedding_dim"]),
        "dropout": float(flow_config["dropout"]),
    }
    model = ConditionalVelocityNetwork(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(flow_config["learning_rate"]),
        weight_decay=float(flow_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(flow_config["lr_factor"]),
        patience=int(flow_config["lr_patience"]),
        min_lr=float(flow_config["min_learning_rate"]),
    )
    amp_enabled = bool(flow_config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ExponentialMovingAverage(model, float(flow_config["ema_decay"]))

    x_train = torch.from_numpy(hybrid.train_conditions)
    y_train = torch.from_numpy(flow_train_target).unsqueeze(-1)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(flow_config["batch_size"]),
        shuffle=True,
        num_workers=int(flow_config["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )

    x_val = torch.from_numpy(hybrid.validation_conditions)
    y_val = torch.from_numpy(flow_validation_target).unsqueeze(-1)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    fixed_time = torch.rand((len(x_val), 1), generator=validation_generator)
    fixed_noise = torch.randn((len(x_val), 1), generator=validation_generator)
    validation_loader = DataLoader(
        TensorDataset(x_val, y_val, fixed_time, fixed_noise),
        batch_size=int(flow_config["batch_size"]),
        shuffle=False,
        num_workers=int(flow_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    artifacts = Path(config["project"]["artifacts_dir"])
    reports = Path(config["project"]["reports_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, int(flow_config["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for condition, target in train_loader:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            time_value = torch.rand_like(target)
            noise = torch.randn_like(target)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = _flow_loss(model, condition, target, time_value, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(flow_config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            train_loss_sum += float(loss.detach()) * len(condition)
            train_examples += len(condition)

        model.eval()
        validation_loss_sum = 0.0
        validation_examples = 0
        with torch.no_grad():
            for condition, target, time_value, noise in validation_loader:
                condition = condition.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                time_value = time_value.to(device, non_blocking=True)
                noise = noise.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = _flow_loss(model, condition, target, time_value, noise)
                validation_loss_sum += float(loss) * len(condition)
                validation_examples += len(condition)

        train_loss = train_loss_sum / train_examples
        validation_loss = validation_loss_sum / validation_examples
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
        }
        history.append(record)
        LOGGER.info(
            "Flow epoch %03d | train %.6f | validation %.6f | lr %.2e",
            epoch,
            train_loss,
            validation_loss,
            learning_rate,
        )

        last_checkpoint = {
            "epoch": epoch,
            "model_kwargs": model_kwargs,
            "model_state": model.state_dict(),
            "ema_state": ema.shadow,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_validation_loss": best_loss,
            "target_transformer": data.target_transformer.to_dict(),
            "residual_transformer": residual_transformer.to_dict(),
            "flow_target": "standardized_log_price_residual_after_ridge_xgboost",
        }
        torch.save(last_checkpoint, artifacts / "flow_last.pt")
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_checkpoint = dict(last_checkpoint)
            best_checkpoint["model_state"] = copy.deepcopy(ema.shadow)
            best_checkpoint["best_validation_loss"] = best_loss
            torch.save(best_checkpoint, artifacts / "flow_best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(flow_config["early_stopping_patience"]):
            LOGGER.info("Early stopping at epoch %d", epoch)
            break

    metadata = {
        "device": str(device),
        "amp_enabled": amp_enabled,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": time.time() - start_time,
        "model_kwargs": model_kwargs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "residual_mean": residual_transformer.mean,
        "residual_std": residual_transformer.std,
    }
    write_json(reports / "flow_history.json", history)
    write_json(artifacts / "flow_metadata.json", metadata)
    return metadata


def load_flow_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ConditionalVelocityNetwork, ResidualStandardizer, TargetTransformer]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConditionalVelocityNetwork(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    residual_transformer = ResidualStandardizer.from_dict(checkpoint["residual_transformer"])
    target_transformer = TargetTransformer.from_dict(checkpoint["target_transformer"])
    return model, residual_transformer, target_transformer


@torch.inference_mode()
def sample_flow(
    model: ConditionalVelocityNetwork,
    conditions: np.ndarray,
    base_prediction: np.ndarray,
    residual_transformer: ResidualStandardizer,
    target_transformer: TargetTransformer,
    n_samples: int,
    steps: int,
    batch_rows: int,
    device: torch.device,
    seed: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate conditional price samples with a fixed-step Heun solver."""
    if n_samples < 1 or steps < 1 or batch_rows < 1:
        raise ValueError("n_samples, steps and batch_rows must be positive")
    output = np.empty((len(conditions), n_samples), dtype=np.float32)
    generator = torch.Generator(device=device).manual_seed(seed)
    iterator = range(0, len(conditions), batch_rows)
    if show_progress:
        iterator = tqdm(iterator, total=math.ceil(len(conditions) / batch_rows), desc="Sampling")
    dt = 1.0 / steps
    for start in iterator:
        stop = min(start + batch_rows, len(conditions))
        condition = torch.from_numpy(conditions[start:stop]).to(device)
        condition = condition.repeat_interleave(n_samples, dim=0)
        price_state = torch.randn(
            (len(condition), 1), device=device, generator=generator, dtype=condition.dtype
        )
        for step_index in range(steps):
            current_time = step_index * dt
            time_value = torch.full_like(price_state, current_time)
            velocity_start = model(price_state, time_value, condition)
            predicted_state = price_state + dt * velocity_start
            next_time = torch.full_like(price_state, current_time + dt)
            velocity_end = model(predicted_state, next_time, condition)
            price_state = price_state + 0.5 * dt * (velocity_start + velocity_end)
        standardized_residual = price_state.view(stop - start, n_samples).cpu().numpy()
        residual = residual_transformer.inverse(standardized_residual)
        final_standardized_log_price = base_prediction[start:stop, None] + residual
        prices = target_transformer.inverse(final_standardized_log_price)
        if not np.isfinite(prices).all():
            raise FloatingPointError("Flow sampler produced NaN or Inf prices")
        output[start:stop] = prices
    return output
