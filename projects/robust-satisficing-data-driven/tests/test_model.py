import numpy as np
import pytest

from robust_satisficing import (
    SourcingProblem,
    empirical_solution,
    evaluate_solution,
    robust_satisficing_solution,
    select_target_by_validation,
    wasserstein_dro_solution,
)


@pytest.fixture
def problem() -> SourcingProblem:
    rng = np.random.default_rng(4)
    mean = np.array([10.0, 10.4, 10.8, 11.2])
    covariance = np.full((4, 4), 0.1)
    np.fill_diagonal(covariance, 0.4)
    samples = rng.multivariate_normal(mean, covariance, size=120)
    return SourcingProblem.from_arrays(samples, upper_bounds=[0.8, 0.8, 0.8, 0.8])


def test_empirical_solution_is_feasible(problem: SourcingProblem) -> None:
    result = empirical_solution(problem)
    assert np.isclose(result.allocation.sum(), 1.0)
    assert np.all(result.allocation >= -1e-10)
    assert np.all(result.allocation <= problem.upper_bounds + 1e-10)


def test_target_below_empirical_optimum_is_rejected(problem: SourcingProblem) -> None:
    z0 = empirical_solution(problem).empirical_cost
    with pytest.raises(ValueError, match="below empirical optimum"):
        robust_satisficing_solution(problem, z0 - 1e-4)


def test_rs_hits_target_and_fragility_matches_max_share(problem: SourcingProblem) -> None:
    z0 = empirical_solution(problem).empirical_cost
    target = z0 + 0.25
    result = robust_satisficing_solution(problem, target)

    assert result.empirical_cost <= target + 1e-8
    assert np.isclose(result.fragility, np.max(result.allocation), atol=1e-8)


def test_more_cost_leeway_cannot_increase_optimal_fragility(problem: SourcingProblem) -> None:
    z0 = empirical_solution(problem).empirical_cost
    tight = robust_satisficing_solution(problem, z0 + 0.05)
    loose = robust_satisficing_solution(problem, z0 + 0.45)

    assert loose.fragility <= tight.fragility + 1e-9


def test_zero_radius_dro_matches_empirical_optimum(problem: SourcingProblem) -> None:
    empirical = empirical_solution(problem)
    dro = wasserstein_dro_solution(problem, radius=0.0)

    assert np.isclose(dro.empirical_cost, empirical.empirical_cost, atol=1e-8)


def test_evaluation_uses_fixed_allocation(problem: SourcingProblem) -> None:
    solution = empirical_solution(problem)
    metrics = evaluate_solution(solution, problem.cost_samples[:20])

    assert set(metrics) == {"mean_cost", "p90_cost", "std_cost"}
    assert np.isfinite(list(metrics.values())).all()



def test_target_selection_uses_declared_validation_candidates(problem: SourcingProblem) -> None:
    rng = np.random.default_rng(77)
    validation = rng.multivariate_normal(
        problem.empirical_mean + np.array([0.5, 0.2, 0.0, 0.0]),
        np.eye(problem.n_suppliers) * 0.2,
        size=300,
    )
    z0 = empirical_solution(problem).empirical_cost
    targets = np.array([z0 + 0.05, z0 + 0.20, z0 + 0.40])
    selected = select_target_by_validation(problem, validation, targets)

    assert selected.target in targets
    assert selected.solution.empirical_cost <= selected.target + 1e-8
    assert np.isfinite(selected.validation_p90)
