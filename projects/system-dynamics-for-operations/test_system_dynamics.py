import numpy as np
from system_dynamics import simulate_inventory_feedback, policy_sweep


def test_constant_demand_is_equilibrium_under_initialized_pipeline():
    r = simulate_inventory_feedback(np.full(30, 10.0), lead_time=2, forecast_alpha=0.5, order_smoothing=0.7, target_inventory_cover=2)
    assert np.allclose(r.orders, 10.0)
    assert np.allclose(r.backlog, 0.0)
    assert r.bullwhip_ratio == 0.0


def test_stock_flow_mass_balance():
    d = np.array([10, 12, 8, 15, 9, 11], dtype=float)
    initial_inventory = 20.0
    r = simulate_inventory_feedback(d, lead_time=2, initial_inventory=initial_inventory)
    assert abs(initial_inventory + r.receipts.sum() - r.shipments.sum() - r.inventory[-1]) < 1e-9


def test_smoothing_reduces_order_variance_for_step_change():
    d = np.r_[np.full(30, 10.0), np.full(30, 18.0), np.full(30, 10.0)]
    fast = simulate_inventory_feedback(d, forecast_alpha=1.0, order_smoothing=1.0)
    smooth = simulate_inventory_feedback(d, forecast_alpha=0.3, order_smoothing=0.3)
    assert np.var(smooth.orders) < np.var(fast.orders)


def test_policy_sweep_returns_ranked_policies():
    d = np.r_[np.full(20, 10.0), np.full(20, 14.0)]
    rows = policy_sweep(d, alphas=(0.2, 0.8), smoothing=(0.2, 0.8))
    assert len(rows) == 4
    assert rows[0]["score"] <= rows[-1]["score"]
