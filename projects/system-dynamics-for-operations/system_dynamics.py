from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DynamicsResult:
    demand: np.ndarray
    orders: np.ndarray
    receipts: np.ndarray
    shipments: np.ndarray
    inventory: np.ndarray
    backlog: np.ndarray
    forecast: np.ndarray

    @property
    def bullwhip_ratio(self) -> float:
        vd = float(np.var(self.demand, ddof=1)) if len(self.demand) > 1 else 0.0
        vo = float(np.var(self.orders, ddof=1)) if len(self.orders) > 1 else 0.0
        if vd < 1e-12:
            return 0.0 if vo < 1e-12 else float("inf")
        return vo / vd


def simulate_inventory_feedback(
    demand,
    lead_time: int = 2,
    forecast_alpha: float = 0.4,
    order_smoothing: float = 0.6,
    target_inventory_cover: float = 2.0,
    initial_inventory: float | None = None,
    capacity: float | None = None,
) -> DynamicsResult:
    """Discrete stock-flow model with forecast, inventory and supply-line feedback."""
    d = np.asarray(demand, dtype=float)
    if len(d) == 0 or np.any(d < 0):
        raise ValueError("demand must be a non-empty non-negative series")
    if lead_time < 1 or not 0 < forecast_alpha <= 1 or not 0 < order_smoothing <= 1:
        raise ValueError("invalid lead time or smoothing parameters")
    f = float(d[0])
    inv = float(target_inventory_cover * f if initial_inventory is None else initial_inventory)
    backlog = 0.0
    pipeline = [f] * lead_time
    previous_order = f

    orders = np.zeros(len(d)); receipts = np.zeros(len(d)); shipments = np.zeros(len(d))
    inventory = np.zeros(len(d)); backlogs = np.zeros(len(d)); forecasts = np.zeros(len(d))

    for t, observed in enumerate(d):
        receipt = pipeline.pop(0)
        inv += receipt
        receipts[t] = receipt

        required = backlog + observed
        shipped = min(inv, required)
        inv -= shipped
        backlog = required - shipped
        shipments[t] = shipped

        f = forecast_alpha * observed + (1 - forecast_alpha) * f
        target_inventory = target_inventory_cover * f
        desired_inventory_position = target_inventory + lead_time * f
        inventory_position = inv + sum(pipeline) - backlog
        raw_order = max(0.0, desired_inventory_position - inventory_position)
        order = (1 - order_smoothing) * previous_order + order_smoothing * raw_order
        if capacity is not None:
            order = min(order, capacity)
        order = max(0.0, order)
        pipeline.append(order)
        previous_order = order

        orders[t] = order; inventory[t] = inv; backlogs[t] = backlog; forecasts[t] = f

    return DynamicsResult(d, orders, receipts, shipments, inventory, backlogs, forecasts)


def policy_score(result: DynamicsResult, holding_cost=1.0, backlog_cost=5.0, order_variance_cost=0.1) -> float:
    return float(
        holding_cost * result.inventory.mean()
        + backlog_cost * result.backlog.mean()
        + order_variance_cost * np.var(result.orders)
    )


def policy_sweep(demand, alphas=(0.2, 0.4, 0.7, 1.0), smoothing=(0.2, 0.5, 0.8, 1.0), **kwargs):
    rows = []
    for a in alphas:
        for s in smoothing:
            result = simulate_inventory_feedback(demand, forecast_alpha=a, order_smoothing=s, **kwargs)
            rows.append({"forecast_alpha": a, "order_smoothing": s, "score": policy_score(result), "bullwhip_ratio": result.bullwhip_ratio})
    return sorted(rows, key=lambda r: r["score"])
