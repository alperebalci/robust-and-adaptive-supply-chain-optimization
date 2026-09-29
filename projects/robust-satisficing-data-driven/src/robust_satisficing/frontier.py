"""Target-fragility frontier utilities for robust satisficing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import (\n    SourcingProblem,\n    SourcingSolution,\n    empirical_solution,\n    robust_satisficing_solution,\n)


@dataclass(frozen=True)
class FrontierPoint:
    target: float
    empirical_cost: float
    fragility: float
    allocation: np.ndarray


def target_fragility_frontier(
    problem:SourcingProblem,
    *,
    points:int=11,
    max_relative_slack:float=0.10,
) -> list[FrontierPoint]:
    """Trace the efficient target/fragility trade-off above the empirical optimum."""
    if points < 2:
        raise ValueError("points must be at least two")
    if max_relative_slack < 0:
        raise ValueError("max_relative_slack must be non-negative")
    base=empirical_solution(problem).empirical_cost
    scale=max(abs(base),1.0)
    targets=np.linspace(base,base+max_relative_slack*scale,points)
    out=[]
    previous=float("inf")
    for target in targets:
        solution: SourcingSolution=robust_satisficing_solution(problem,float(target))
        if solution.fragility > previous + 1e-8:
            raise RuntimeError("fragility frontier is not monotone")
        previous=solution.fragility
        out.append(
            FrontierPoint(
                target=float(target),
                empirical_cost=solution.empirical_cost,
                fragility=solution.fragility,
                allocation=solution.allocation.copy(),
            )
        )
    return out
