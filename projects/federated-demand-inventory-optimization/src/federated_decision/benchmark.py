from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SiteSeries:
    name: str
    demand: np.ndarray


@dataclass(frozen=True)
class PolicyMetrics:
    mean_cost: float
    service_level: float
    mean_holding: float
    mean_shortage: float
    safety_stock: float


def generate_sites(n_periods: int = 260, seed: int = 7) -> list[SiteSeries]:
    """Generate reproducible heterogeneous demand series for three sites."""
    if n_periods < 40:
        raise ValueError("n_periods must be at least 40")
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods, dtype=float)
    configs = [
        ("north", 45.0, 8.0, 0.10, 4.0),
        ("central", 62.0, 12.0, 0.06, 6.0),
        ("south", 36.0, 6.0, 0.14, 3.5),
    ]
    sites: list[SiteSeries] = []
    for idx, (name, level, weekly, trend, noise) in enumerate(configs):
        phase = idx * 0.8
        seasonal = weekly * np.sin(2.0 * np.pi * t / 7.0 + phase)
        slower = 0.35 * weekly * np.sin(2.0 * np.pi * t / 28.0 + 0.3 * idx)
        signal = level + trend * t + seasonal + slower
        shocks = rng.normal(0.0, noise, size=n_periods)
        demand = np.maximum(0.0, signal + shocks)
        sites.append(SiteSeries(name=name, demand=demand.astype(float)))
    return sites


def supervised_rows(
    demand: np.ndarray,
    start: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Lag/calendar features; target and lags are scaled by 100 for stable GD."""
    if len(demand) <= start:
        raise ValueError("demand series is too short")
    X, y = [], []
    for t in range(start, len(demand)):
        X.append(
            [
                1.0,
                demand[t - 1] / 100.0,
                demand[t - 7] / 100.0,
                np.sin(2.0 * np.pi * t / 7.0),
                np.cos(2.0 * np.pi * t / 7.0),
                t / max(len(demand) - 1, 1),
            ]
        )
        y.append(demand[t] / 100.0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def fit_linear_gd(
    X: np.ndarray,
    y: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    learning_rate: float = 0.05,
    epochs: int = 300,
    l2: float = 1e-3,
) -> np.ndarray:
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("invalid supervised data shapes")
    w = (
        np.zeros(X.shape[1], dtype=float)
        if initial is None
        else initial.astype(float).copy()
    )
    for _ in range(epochs):
        error = X @ w - y
        grad = (X.T @ error) / len(X)
        grad[1:] += l2 * w[1:]
        w -= learning_rate * grad
    return w


def fedavg(
    site_data: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    rounds: int = 30,
    local_epochs: int = 8,
    learning_rate: float = 0.04,
    l2: float = 1e-3,
) -> np.ndarray:
    if not site_data:
        raise ValueError("site_data cannot be empty")
    d = site_data[0][0].shape[1]
    global_w = np.zeros(d, dtype=float)
    total = sum(len(y) for _, y in site_data)
    for _ in range(rounds):
        local_weights = []
        sizes = []
        for X, y in site_data:
            local_w = fit_linear_gd(
                X,
                y,
                initial=global_w,
                learning_rate=learning_rate,
                epochs=local_epochs,
                l2=l2,
            )
            local_weights.append(local_w)
            sizes.append(len(y))
        global_w = sum(w * (n / total) for w, n in zip(local_weights, sizes))
    return global_w


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, (X @ weights) * 100.0)


def one_period_cost(
    forecast: np.ndarray,
    actual: np.ndarray,
    safety_stock: float,
    *,
    holding_cost: float = 1.0,
    shortage_cost: float = 5.0,
) -> tuple[float, float, float, float]:
    target = forecast + safety_stock
    leftover = np.maximum(target - actual, 0.0)
    shortage = np.maximum(actual - target, 0.0)
    cost = holding_cost * leftover + shortage_cost * shortage
    service = np.mean(shortage <= 1e-9)
    return (
        float(np.mean(cost)),
        float(service),
        float(np.mean(leftover)),
        float(np.mean(shortage)),
    )


def optimize_safety_stock(
    train_forecast: np.ndarray,
    train_actual: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    holding_cost: float = 1.0,
    shortage_cost: float = 5.0,
) -> float:
    if grid is None:
        grid = np.linspace(0.0, 35.0, 71)
    best_s, best_cost = 0.0, float("inf")
    for s in grid:
        cost, _, _, _ = one_period_cost(
            train_forecast,
            train_actual,
            float(s),
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
        )
        if cost < best_cost - 1e-12:
            best_s, best_cost = float(s), cost
    return best_s


def evaluate_policy(
    train_forecast: np.ndarray,
    train_actual: np.ndarray,
    test_forecast: np.ndarray,
    test_actual: np.ndarray,
) -> PolicyMetrics:
    safety = optimize_safety_stock(train_forecast, train_actual)
    cost, service, holding, shortage = one_period_cost(
        test_forecast,
        test_actual,
        safety,
    )
    return PolicyMetrics(cost, service, holding, shortage, safety)


def run_benchmark(
    *,
    n_periods: int = 260,
    train_end: int = 190,
    seed: int = 7,
) -> dict[str, dict[str, PolicyMetrics]]:
    sites = generate_sites(n_periods=n_periods, seed=seed)
    train_sets: list[tuple[np.ndarray, np.ndarray]] = []
    test_sets: list[tuple[np.ndarray, np.ndarray]] = []

    for site in sites:
        train_demand = site.demand[:train_end]
        X_train, y_train = supervised_rows(train_demand)
        X_full, y_full = supervised_rows(site.demand)
        test_start_row = train_end - 7
        X_test, y_test = X_full[test_start_row:], y_full[test_start_row:]
        train_sets.append((X_train, y_train))
        test_sets.append((X_test, y_test))

    pooled_X = np.vstack([X for X, _ in train_sets])
    pooled_y = np.concatenate([y for _, y in train_sets])
    central_w = fit_linear_gd(
        pooled_X,
        pooled_y,
        epochs=900,
        learning_rate=0.04,
    )
    fed_w = fedavg(train_sets)
    local_ws = [
        fit_linear_gd(X, y, epochs=900, learning_rate=0.04)
        for X, y in train_sets
    ]

    results: dict[str, dict[str, PolicyMetrics]] = {
        "centralized": {},
        "fedavg": {},
        "local_only": {},
    }
    for i, site in enumerate(sites):
        X_train, y_train_scaled = train_sets[i]
        X_test, y_test_scaled = test_sets[i]
        y_train = y_train_scaled * 100.0
        y_test = y_test_scaled * 100.0
        for method, weights in [
            ("centralized", central_w),
            ("fedavg", fed_w),
            ("local_only", local_ws[i]),
        ]:
            train_pred = predict(X_train, weights)
            test_pred = predict(X_test, weights)
            results[method][site.name] = evaluate_policy(
                train_pred,
                y_train,
                test_pred,
                y_test,
            )
    return results


def summarize(
    results: dict[str, dict[str, PolicyMetrics]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method, site_metrics in results.items():
        values = list(site_metrics.values())
        summary[method] = {
            "mean_cost": float(np.mean([m.mean_cost for m in values])),
            "mean_service_level": float(
                np.mean([m.service_level for m in values])
            ),
            "mean_safety_stock": float(
                np.mean([m.safety_stock for m in values])
            ),
        }
    return summary


if __name__ == "__main__":
    from pprint import pprint

    pprint(summarize(run_benchmark()))
