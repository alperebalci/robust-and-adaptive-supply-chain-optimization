# Research Notes

## Methodological placement

Robust satisficing is target-driven rather than ambiguity-radius-driven. The acceptable target `tau` must be at least as large as the empirical optimum for this minimization model. The optimization then seeks the smallest fragility coefficient `k` that controls target deterioration as the distribution moves away from the empirical distribution.

This is not a relaxation of Wasserstein DRO. It produces a different parameterization of robustness and can generate a different solution path.

## Why the sourcing model is useful

For `f(x,z)=z^T x` with nonnegative sourcing shares and an L1 Wasserstein ground metric, the fragility term becomes `||x||_inf`. This yields a small LP whose behavior is easy to verify independently:

- low target leeway tends to preserve concentration in historically cheap suppliers;
- larger target leeway permits diversification;
- the optimal fragility path is monotone non-increasing as the target is relaxed.

These are structural properties of this special case, not empirical claims about all robust-satisficing models.

## Literature correction

The principal paper implemented here is by Daniel Zhuoyu Long, Melvyn Sim, and Minglong Zhou, published online in 2022 and in *Operations Research* 71(1). It should not be attributed to Bandi and Bertsimas.
