# CA6002 HDB Resale Price Modelling

This project compares deterministic resale-price baselines with an integrated residual architecture:

```text
Ridge time trend -> XGBoost nonlinear residual -> Conditional Flow Matching residual distribution
```

## Interpretation

- Ridge and direct XGBoost remain transparent baselines.
- The hybrid point model combines Ridge extrapolation with XGBoost residual correction.
- CFM learns the remaining hybrid-model error distribution instead of relearning the full price.
- Address, street and block information use cross-fitted smoothed residual encodings to avoid target leakage.
- The sample median is the statistical market benchmark.
- The 10th and 90th percentiles form an 80% market range.
- `possible_premium` and `possible_discount` are statistical flags, not official valuations or causal claims.

The source CSV remains unchanged. `price_per_sqm` is never used as an input because it is calculated from the target.

## Environment

The tested local target is Windows, Python 3.12, and an NVIDIA RTX 5070 Laptop GPU. The CUDA 12.8 PyTorch wheel is used because it is compatible with the installed 572-series driver.
On this driver, CFM uses the GPU while the current Windows XGBoost wheel uses its reliable CPU path.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cu128.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

For a CPU-only environment, install the CPU PyTorch wheel first and then `requirements.txt`.

## Commands

```powershell
.\.venv\Scripts\python.exe -m hdb_price.cli validate-data
.\.venv\Scripts\python.exe -m hdb_price.cli train-baselines
.\.venv\Scripts\python.exe -m hdb_price.cli train-flow
.\.venv\Scripts\python.exe -m hdb_price.cli evaluate
```

Run the complete pipeline:

```powershell
.\.venv\Scripts\python.exe -m hdb_price.cli run-all
```

Quick end-to-end integration check:

```powershell
.\.venv\Scripts\python.exe -m hdb_price.cli run-all --config configs/smoke.yaml
```

Predict a new CSV containing the configured model features:

```powershell
.\.venv\Scripts\python.exe -m hdb_price.cli predict --input new_flats.csv --output reports/new_predictions.csv
```

`examples/new_flats.csv` provides the required inference schema.

## Outputs

- `artifacts/`: preprocessing objects, baseline models, leakage-safe location encoder, hybrid XGBoost and residual-CFM checkpoints.
- `reports/metrics.json`: the only evaluation result; contains point and probabilistic metrics.
- `reports/flow_history.json`: CFM training loss history.
- `reports/data_validation.json`: source-data validation summary.

The `evaluate` command prints the same metric dictionary stored in
`reports/metrics.json`. It does not create plots or transaction-level prediction
files. The separate `predict` command writes a CSV only when explicitly invoked.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
