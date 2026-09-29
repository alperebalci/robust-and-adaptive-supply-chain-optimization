"""Validation-only target calibration for robust satisficing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import SourcingProblem, SourcingSolution, evaluate_solution, robust_satisficing_solution


@dataclass(frozen=True)
class TargetSelection:
    target: float
    solution: SourcingSolution
    validation_mean: float
    validation_p90: float
    score: float


def select_target_by_validation(
    problem: SourcingProblem,
    validation_costs: ArrayLike,
    targets: ArrayLike,
    *,
    fragility_penalty: float = 0.0,
) -> TargetSelection:
    """Select an acceptable target using validation data only.

    Candidate policies are fit on the training problem. Validation observations are
    used only to select among already-declared targets. Lower validation p90 cost is
    preferred, optionally with an explicit fragility penalty.
    """

    validation = np.asarray(validation_costs, dtype=float)
    candidate_targets = np.asarray(targets, dtype=float)
    if validation.ndim != 2 or validation.shape[1] != problem.n_suppliers:
        raise ValueError("validation_costs have invalid shape")
    if candidate_targets.ndim != 1 or candidate_targets.size < 1:
        raise ValueError("targets must be a non-empty vector")
    if fragility_penalty < 0.0 or not np.isfinite(fragility_penalty):
        raise ValueError("fragility_penalty must be finite and non-negative")

    best: TargetSelection | None = None
    for target in candidate_targets:
        solution = robust_satisficing_solution(problem, float(target))
        metrics = evaluate_solution(solution, validation)
        score = metrics["p90_cost"] + fragility_penalty * solution.fragility
        candidate = TargetSelection(
            target=float(target),
            solution=solution,
            validation_mean=metrics["mean_cost"],
            validation_p90=metrics["p90_cost"],
            score=float(score),
        )
        if best is None or candidate.score < best.score - 1e-12:
            best = candidate

    assert best is not None
    return best
