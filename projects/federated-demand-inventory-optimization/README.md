# Federated Demand Learning for Inventory Optimization

A decision-focused benchmark for asking a more useful question than "does federated learning predict well?":

> **Does decentralized model training preserve enough information to support good downstream inventory decisions?**

Three heterogeneous sites retain their local demand histories. A shared linear forecasting model is trained with FedAvg, then each site's forecast is passed into a one-period order-up-to decision with safety stock optimized on training data. The evaluation reports **operational cost and service level**, not forecast loss alone.

## Why this project exists

Federated learning is often demonstrated as a model-training architecture without an explicit Operations Research consequence. In Industrial Engineering, the relevant test is downstream:

```text
local demand histories
        ↓
centralized / local / federated forecasting
        ↓
site-specific safety-stock optimization
        ↓
held-out order-up-to decisions
        ↓
holding cost / shortage cost / service level
```

This benchmark therefore keeps prediction and decision quality separate.

## Compared learning regimes

1. **centralized** — pooled training data; information-advantaged reference.
2. **fedavg** — sites train local gradient steps and share model weights, not raw demand rows.
3. **local_only** — each site trains its own model with no cross-site information sharing.

All three models use the same feature family so the comparison focuses on the data-sharing/training regime rather than architecture changes.

## Demand features

For period t:

- intercept;
- demand at t-1;
- demand at t-7;
- weekly sine/cosine calendar encoding;
- normalized time trend.

The synthetic fixture deliberately creates heterogeneous site levels, seasonality, trend, and noise.

## Decision layer

Given forecast `f_t` and safety stock `s`, the target stock level is:

```text
S_t = f_t + s
```

The per-period cost is:

```text
holding_cost  * max(S_t - demand_t, 0)
+ shortage_cost * max(demand_t - S_t, 0)
```

Each method/site combination selects safety stock using **training data only** from a fixed grid, then evaluates the frozen policy on the held-out future block.

## Reference result

With the default seed and fixture, the bundled benchmark currently produces approximately:

| Method | Mean decision cost | Mean service level |
|---|---:|---:|
| centralized | 11.22 | 0.63 |
| fedavg | 12.20 | 0.61 |
| local_only | 14.12 | 0.44 |

These numbers are fixture-specific. They are not presented as a general ranking of federated, centralized, or local learning.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m federated_decision.benchmark
pytest -q
```

## Repository structure

```text
src/federated_decision/
  __init__.py
  benchmark.py

tests/
  test_benchmark.py

docs/
  experiment_design.md
```

## Scope and limitations

- This is a reproducible synthetic multi-site benchmark, not a privacy guarantee.
- FedAvg weight exchange is simulated in-process; there is no networking or secure aggregation layer.
- Differential privacy is not implemented.
- The decision layer is intentionally transparent and one-period; multi-echelon lead times and coupled network constraints are future extensions.
- Centralized training is an information-advantaged reference, not a deployment recommendation.

## Next research extensions

- multi-echelon inventory with lead times and transshipment;
- heterogeneous local model architectures and personalization;
- secure aggregation / differential privacy with explicit utility loss;
- decision-focused federated objectives that optimize downstream cost directly;
- federated scenario generation for stochastic programming;
- distribution shift and site-entry/site-exit stress tests.
