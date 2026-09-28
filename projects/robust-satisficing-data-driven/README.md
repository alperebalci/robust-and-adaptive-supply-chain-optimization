# Robust Satisficing for Data-Driven Sourcing

A compact research implementation of **robust satisficing** for a transparent supply-allocation problem.

The model follows the target-oriented framework of Long, Sim, and Zhou, *Robust Satisficing*, Operations Research 71(1), 61–82. Instead of choosing a Wasserstein ambiguity radius first, the decision maker specifies an acceptable empirical objective target and the optimization model minimizes the fragility of meeting that target as the data-generating distribution moves away from the empirical distribution.

## Why this is not ordinary DRO

For a loss function `f(x,z)`, empirical distribution `P_hat`, probability distance `Delta`, and acceptable target `tau`, robust satisficing seeks the smallest proportionality factor `k` such that

```text
E_P[f(x,z)] - tau <= k * Delta(P, P_hat)
for every distribution P.
```

The optimized value `k` is the fragility measure in this project.

A Wasserstein DRO model instead fixes a radius `r` and minimizes the worst-case objective over distributions within that radius. The two formulations therefore expose different control parameters:

- DRO: choose uncertainty radius `r`;
- robust satisficing: choose acceptable target `tau`, then minimize fragility `k`.

## Exact special case implemented here

The decision is a vector of sourcing shares

```text
x_i >= 0
sum_i x_i = 1
x_i <= supplier_cap_i.
```

A historical scenario `z` contains supplier unit costs, and the realized sourcing loss is

```text
f(x,z) = z^T x.
```

The project uses 1-Wasserstein distance with an L1 ground norm and unrestricted cost-vector support `R^n`. For this affine loss, the dual norm is L-infinity and the robust-satisficing formulation reduces exactly to the LP

```text
minimize      k
subject to    mean(z)^T x <= tau
              x_i <= k                 for every supplier i
              sum_i x_i = 1
              0 <= x_i <= supplier_cap_i.
```

Because `x >= 0`, the optimal fragility is

```text
k = ||x||_inf = max_i x_i.
```

The model therefore makes the target/robustness trade-off visible: additional leeway in empirical cost can be exchanged for a less concentrated sourcing policy.

## Matching Wasserstein-DRO baseline

Under the same assumptions, the corresponding Wasserstein-DRO special case is

```text
minimize    mean(z)^T x + r * ||x||_inf
```

and is solved with the same LP machinery. This gives a controlled comparison between:

- empirical optimization;
- target-driven robust satisficing;
- radius-driven Wasserstein DRO.

## Synthetic benchmark

The benchmark creates correlated historical supplier costs and evaluates frozen policies on:

1. an independent nominal sample;
2. a shifted sample in which the cheapest historical supplier experiences the largest cost increase.

Run:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
python -m robust_satisficing.experiment
```

The synthetic stress result is an experiment, not evidence that robust satisficing dominates DRO or empirical optimization in general.

## Validation contract

The tests verify that:

- supplier allocations sum to one and satisfy supplier caps;
- a target below the empirical optimum is rejected;
- the robust-satisficing solution satisfies the chosen target;
- optimized fragility equals the maximum sourcing share in this special case;
- relaxing the acceptable target cannot increase optimal fragility;
- zero-radius DRO reproduces the empirical optimum value;
- holdout evaluation never re-optimizes the frozen sourcing vector.

## Scope boundary

This v0.1 intentionally implements one exact, auditable special case. It does not claim to reproduce every reformulation in the paper.

Natural extensions are:

- bounded/polyhedral supports;
- piecewise-linear losses;
- discrete/combinatorial sourcing decisions;
- lot sizing with recourse;
- target selection by validation;
- finite-sample confidence studies;
- direct comparison with the umbrella repository's existing Wasserstein-DRO supply-network models.

## Reference

Daniel Zhuoyu Long, Melvyn Sim, and Minglong Zhou. "Robust Satisficing." *Operations Research* 71(1):61–82. DOI: 10.1287/opre.2021.2238.

## License

This project inherits the umbrella repository's MIT license.
