import numpy as np

from robust_satisficing.frontier import target_fragility_frontier
from robust_satisficing.model import SourcingProblem


def test_target_fragility_frontier_is_monotone() -> None:
    rng=np.random.default_rng(8)
    samples=rng.normal([10.0,10.4,10.8],[0.3,0.4,0.5],size=(200,3))
    problem=SourcingProblem.from_arrays(samples,[0.8,0.8,0.8])
    frontier=target_fragility_frontier(problem,points=7,max_relative_slack=0.08)
    fragility=np.array([p.fragility for p in frontier])
    assert np.all(np.diff(fragility) <= 1e-8)
