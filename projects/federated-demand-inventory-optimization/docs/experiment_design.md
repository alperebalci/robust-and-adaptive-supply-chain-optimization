# Experiment Design

## Objective

Evaluate federated learning as an input to an Operations Research decision, not as an isolated prediction task.

## Data boundary

Each site owns one chronological demand series. The federated method exchanges model parameters only. The centralized reference pools rows and is therefore information-advantaged. The local-only baseline uses no cross-site information.

The current benchmark is a simulation of federated training; it does not provide cryptographic privacy, secure aggregation, or differential privacy.

## Train/test protocol

- First 190 periods: training and safety-stock selection.
- Remaining future periods: held-out decision evaluation.
- No test-period demand is used for model fitting or safety-stock optimization.

## Decision protocol

For each method and site:

1. fit the forecasting model on permitted training data;
2. generate training forecasts;
3. select safety stock on the training block only;
4. freeze model and safety stock;
5. generate held-out forecasts;
6. evaluate holding cost, shortage cost, and service level.

## Metrics

Primary downstream metrics:

- mean inventory decision cost;
- service level;
- mean holding;
- mean shortage;
- selected safety stock.

Forecast error can be added as a diagnostic, but it is not the primary objective.

## Interpretation

A federated model can be statistically worse than the centralized reference yet still be operationally adequate if downstream decision cost remains close. Conversely, a small prediction-error improvement may have little decision value.

Any future privacy mechanism should therefore report a three-way trade-off:

```text
privacy / communication burden
        ↔
forecast utility
        ↔
downstream decision utility
```
