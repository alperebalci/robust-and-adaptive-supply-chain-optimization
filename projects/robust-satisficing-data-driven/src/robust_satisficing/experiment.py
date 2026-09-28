"""Synthetic train/shift evaluation for robust satisficing."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .model import (
    SourcingProblem,
    SourcingSolution,
    empirical_solution,
    evaluate_solution,
    robust_satisficing_solution,
    wasserstein_dro_solution,
)


def generate_cost_samples(
    seed: int,
    samples: int,
    *,
    stress: bool = False,
) -> np.ndarray:
    """Generate correlated supplier unit costs with an optional supplier-specific shift."""

    rng = np.random.default_rng(seed)
    mean = np.array([10.0, 10.35, 10.75, 11.05], dtype=float)
    if stress:
        mean = mean + np.array([2.0, 0.7, 0.15, 0.0])

    covariance = np.full((4, 4), 0.12, dtype=float)
    np.fill_diagonal(covariance, 0.45)
    return rng.multivariate_normal(mean, covariance, size=samples)


def _serialize(solution: SourcingSolution) -> dict[str, object]:
    payload = asdict(solution)
    payload["allocation"] = solution.allocation.tolist()
    return payload


def run_experiment(seed: int = 11) -> dict[str, object]:
    train = generate_cost_samples(seed, 100, stress=False)
    nominal_test = generate_cost_samples(seed + 1000, 5000, stress=False)
    stress_test = generate_cost_samples(seed + 2000, 5000, stress=True)

    problem = SourcingProblem.from_arrays(train, upper_bounds=[0.8, 0.8, 0.8, 0.8])
    empirical = empirical_solution(problem)

    targets = {
        "rs_1pct": empirical.empirical_cost + 0.01 * abs(empirical.empirical_cost),
        "rs_3pct": empirical.empirical_cost + 0.03 * abs(empirical.empirical_cost),
        "rs_6pct": empirical.empirical_cost + 0.06 * abs(empirical.empirical_cost),
    }
    policies = {"empirical": empirical}
    policies.update(
        {name: robust_satisficing_solution(problem, target) for name, target in targets.items()}
    )
    policies["dro_radius_0_4"] = wasserstein_dro_solution(problem, radius=0.4)

    return {
        "empirical_optimum": empirical.empirical_cost,
        "targets": targets,
        "policies": {
            name: {
                "solution": _serialize(solution),
                "nominal_test": evaluate_solution(solution, nominal_test),
                "stress_test": evaluate_solution(solution, stress_test),
            }
            for name, solution in policies.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
