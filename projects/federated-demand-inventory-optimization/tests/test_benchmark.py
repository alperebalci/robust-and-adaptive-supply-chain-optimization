import numpy as np

from federated_decision.benchmark import (
    fedavg,
    generate_sites,
    optimize_safety_stock,
    run_benchmark,
    summarize,
    supervised_rows,
)


def test_generated_sites_are_deterministic() -> None:
    a = generate_sites(seed=11)
    b = generate_sites(seed=11)
    assert [s.name for s in a] == [s.name for s in b]
    for left, right in zip(a, b):
        assert np.allclose(left.demand, right.demand)


def test_fedavg_returns_finite_weights() -> None:
    sites = generate_sites(n_periods=80, seed=3)
    data = [supervised_rows(s.demand[:60]) for s in sites]
    weights = fedavg(data, rounds=3, local_epochs=2)
    assert weights.shape == (6,)
    assert np.isfinite(weights).all()


def test_safety_stock_optimization_is_feasible() -> None:
    forecast = np.array([10.0, 12.0, 14.0])
    actual = np.array([11.0, 15.0, 13.0])
    safety = optimize_safety_stock(
        forecast,
        actual,
        grid=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    assert 0.0 <= safety <= 3.0


def test_end_to_end_benchmark_has_all_methods_and_sites() -> None:
    summary = summarize(
        run_benchmark(n_periods=120, train_end=85, seed=5)
    )
    assert set(summary) == {"centralized", "fedavg", "local_only"}
    for metrics in summary.values():
        assert metrics["mean_cost"] >= 0.0
        assert 0.0 <= metrics["mean_service_level"] <= 1.0
