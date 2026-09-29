"""Exact LP formulations for a transparent robust-satisficing special case."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


@dataclass(frozen=True)
class SourcingProblem:
    """Historical unit-cost samples and supplier allocation caps."""

    cost_samples: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]

    @classmethod
    def from_arrays(
        cls,
        cost_samples: ArrayLike,
        upper_bounds: ArrayLike | None = None,
    ) -> SourcingProblem:
        costs = np.asarray(cost_samples, dtype=float)
        if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[1] < 1:
            raise ValueError("cost_samples must be a non-empty 2D array")
        if np.any(~np.isfinite(costs)):
            raise ValueError("cost_samples must be finite")

        n = costs.shape[1]
        bounds = np.ones(n, dtype=float) if upper_bounds is None else np.asarray(upper_bounds, dtype=float)
        if bounds.ndim != 1 or bounds.shape != (n,):
            raise ValueError("upper_bounds must have one entry per supplier")
        if np.any(~np.isfinite(bounds)) or np.any(bounds < 0.0) or np.any(bounds > 1.0):
            raise ValueError("upper_bounds must lie in [0, 1]")
        if bounds.sum() < 1.0 - 1e-12:
            raise ValueError("supplier upper bounds cannot cover total demand")

        return cls(cost_samples=costs, upper_bounds=bounds)

    @property
    def n_suppliers(self) -> int:
        return int(self.cost_samples.shape[1])

    @property
    def empirical_mean(self) -> NDArray[np.float64]:
        return self.cost_samples.mean(axis=0)


@dataclass(frozen=True)
class SourcingSolution:
    """A feasible sourcing allocation and its in-sample diagnostics."""

    allocation: NDArray[np.float64]
    empirical_cost: float
    fragility: float
    objective: float
    status: int
    message: str


def _base_bounds(problem: SourcingProblem) -> list[tuple[float, float]]:
    return [(0.0, float(upper)) for upper in problem.upper_bounds]


def empirical_solution(problem: SourcingProblem) -> SourcingSolution:
    """Minimize empirical mean sourcing cost."""

    result = linprog(
        c=problem.empirical_mean,
        A_eq=np.ones((1, problem.n_suppliers)),
        b_eq=np.array([1.0]),
        bounds=_base_bounds(problem),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"empirical LP failed: {result.message}")

    x = np.asarray(result.x, dtype=float)
    cost = float(problem.empirical_mean @ x)
    return SourcingSolution(
        allocation=x,
        empirical_cost=cost,
        fragility=float(np.max(x)),
        objective=float(result.fun),
        status=int(result.status),
        message=str(result.message),
    )


def robust_satisficing_solution(problem: SourcingProblem, target: float) -> SourcingSolution:
    """Minimize Wasserstein fragility subject to an acceptable empirical-cost target.

    For loss f(x, z)=z^T x, W1 with L1 ground norm, support R^n, x>=0,
    the robust-satisficing model reduces exactly to:

        min k
        s.t. mean(z)^T x <= target
             x_i <= k
             sum_i x_i = 1
             0 <= x_i <= upper_i.
    """

    if not np.isfinite(target):
        raise ValueError("target must be finite")

    empirical = empirical_solution(problem)
    if target < empirical.empirical_cost - 1e-10:
        raise ValueError(
            f"target {target:.8g} is below empirical optimum {empirical.empirical_cost:.8g}"
        )

    n = problem.n_suppliers
    c = np.concatenate([np.zeros(n), np.array([1.0])])

    rows = [np.concatenate([problem.empirical_mean, np.array([0.0])])]
    rhs = [float(target)]
    for i in range(n):
        row = np.zeros(n + 1)
        row[i] = 1.0
        row[-1] = -1.0
        rows.append(row)
        rhs.append(0.0)

    result = linprog(
        c=c,
        A_ub=np.stack(rows),
        b_ub=np.asarray(rhs),
        A_eq=np.concatenate([np.ones(n), np.array([0.0])]).reshape(1, -1),
        b_eq=np.array([1.0]),
        bounds=_base_bounds(problem) + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"robust satisficing LP failed: {result.message}")

    x = np.asarray(result.x[:n], dtype=float)
    k = float(result.x[-1])
    return SourcingSolution(
        allocation=x,
        empirical_cost=float(problem.empirical_mean @ x),
        fragility=k,
        objective=float(result.fun),
        status=int(result.status),
        message=str(result.message),
    )


def wasserstein_dro_solution(problem: SourcingProblem, radius: float) -> SourcingSolution:
    """Solve the matching W1-DRO special case: min mean_cost + radius * ||x||_inf."""

    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")

    n = problem.n_suppliers
    c = np.concatenate([problem.empirical_mean, np.array([radius])])

    rows = []
    rhs = []
    for i in range(n):
        row = np.zeros(n + 1)
        row[i] = 1.0
        row[-1] = -1.0
        rows.append(row)
        rhs.append(0.0)

    result = linprog(
        c=c,
        A_ub=np.stack(rows),
        b_ub=np.asarray(rhs),
        A_eq=np.concatenate([np.ones(n), np.array([0.0])]).reshape(1, -1),
        b_eq=np.array([1.0]),
        bounds=_base_bounds(problem) + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Wasserstein DRO LP failed: {result.message}")

    x = np.asarray(result.x[:n], dtype=float)
    k = float(result.x[-1])
    return SourcingSolution(
        allocation=x,
        empirical_cost=float(problem.empirical_mean @ x),
        fragility=k,
        objective=float(result.fun),
        status=int(result.status),
        message=str(result.message),
    )


def evaluate_solution(solution: SourcingSolution, cost_samples: ArrayLike) -> dict[str, float]:
    """Evaluate a fixed allocation on held-out or shifted cost samples."""

    costs = np.asarray(cost_samples, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != solution.allocation.size:
        raise ValueError("evaluation samples must match the allocation dimension")
    realized = costs @ solution.allocation
    return {
        "mean_cost": float(np.mean(realized)),
        "p90_cost": float(np.quantile(realized, 0.90)),
        "std_cost": float(np.std(realized, ddof=1)) if realized.size > 1 else 0.0,
    }
