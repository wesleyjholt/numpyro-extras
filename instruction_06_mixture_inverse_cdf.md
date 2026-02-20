# Agent Task 06: Mixture -> Inverse-CDF Callable (Root-Finding Backend)

## Objective
Implement a reusable, JAX-compatible backend that computes the quantile function (inverse CDF) for a **1D continuous mixture distribution** using robust root finding (`optimistix.Bisection`).

This agent delivers the numerical truth source for later interpolation and transform layers.

## Precondition
Task 01 test specification must be completed first, and the Task 06 implementation must satisfy those tests.

## Scope
- In scope:
  - Mixture CDF callable `F(x)`.
  - Mixture log-density callable `log f(x)`.
  - Inverse-CDF callable `Q(u)` via root finding.
  - Bracketing strategy that works near tails.
  - Batch/vectorized operation with `vmap`.
- Out of scope:
  - Interpolation over quantile knots.
  - NumPyro `Transform` subclasses.
  - Sampling APIs beyond validation utilities.

## Inputs (Contract)
Define a config and data contract the caller must provide:

1. `weights: Array[K]`
   - Non-negative, finite.
   - Sum to `1` within tolerance (`atol=1e-6`).
2. `component_distribution`
   - Must expose `cdf(x)` and `log_prob(x)`.
   - Parameters broadcast to `K` mixture components.
3. `solver_cfg`
   - `rtol: float` (default `1e-6`)
   - `atol: float` (default `1e-6`)
   - `max_steps: int` (default `128`)
4. `bracket_cfg`
   - `x_init_low: float` (default `-8.0`)
   - `x_init_high: float` (default `8.0`)
   - `expansion_factor: float` (default `2.0`)
   - `max_expansions: int` (default `32`)
5. `u`
   - Scalar or array in open interval `(0, 1)`.
   - Caller may pass clipped values in `[eps, 1-eps]`; still guard internally.

## Outputs (Contract)
Return a small backend object (dataclass or lightweight class) with:

1. `cdf(x) -> u`
2. `log_prob(x) -> log_pdf`
3. `icdf(u) -> x`
4. `icdf_with_status(u) -> (x, converged, n_steps)` (optional, recommended for diagnostics)
5. `validate() -> dict` (recommended; monotonicity and roundtrip quick checks)

All callables must be pure functions over explicit inputs (no hidden mutable state).

## File Targets
- Add a new module:
  - `src/numpyro_extras/mixture_quantile.py`
- Export new public symbols in:
  - `src/numpyro_extras/__init__.py` (if present)
  - or relevant package export file used in this repo.

## Implementation Requirements

### 1) Mixture CDF and log-density
- Implement:
  - `mixture_cdf(x) = sum_k w_k * F_k(x)`
  - `mixture_log_prob(x) = logsumexp(log w_k + log f_k(x))`
- Use numerically stable `logsumexp` for log-density.
- Ensure shape logic handles scalar and batched `x`.

### 2) Root function for inverse-CDF
- Define `g(x; u) = mixture_cdf(x) - u`.
- For each `u`, solve `g(x; u)=0` with bisection.
- Use `optimistix.Bisection` with provided tolerances.

### 3) Bracket discovery strategy
- Start with `[x_init_low, x_init_high]`.
- If sign condition fails (`g(low)` and `g(high)` same sign), expand:
  - `low *= expansion_factor` toward `-inf`
  - `high *= expansion_factor` toward `+inf`
  - Repeat up to `max_expansions`.
- If still unbracketed:
  - Return explicit failure status in diagnostics path.
  - Default path may raise informative `ValueError` (preferred) unless caller opts for soft-failure.

### 4) Endpoint behavior
- Internally clamp `u` to `[eps, 1-eps]`, default `eps=1e-10`.
- Preserve caller-visible determinism: document clamp value.

### 5) JAX compatibility
- Ensure functions are JIT-safe:
  - no Python-side data-dependent control flow in core kernels.
  - use `lax.while_loop`/`lax.cond` where needed.
- Provide vectorized wrappers with `jax.vmap` for batched `u`.

## Edge Cases to Handle
1. Degenerate weights (some zeros).
2. Highly imbalanced mixtures.
3. Very small/large scales in components.
4. `u` extremely near `0` or `1`.
5. Broadcasted batch dimensions.

## Validation and Acceptance Tests
Add tests in `tests/test_mixture_quantile.py`:

1. **Single-component sanity**
   - Mixture with one component should match component `icdf` closely.
2. **Roundtrip**
   - `cdf(icdf(u))` close to `u` on dense grid.
3. **Monotonic quantile**
   - increasing `u` yields nondecreasing `x`.
4. **Tail stability**
   - test at `u in {1e-8, 1e-6, 1-1e-6, 1-1e-8}` finite outputs.
5. **Batched input**
   - batched `u` returns expected shape and convergence.

Use concrete tolerances:
- interior: `abs(cdf(icdf(u))-u) < 5e-5`
- extreme tails: `< 2e-4`

## Deliverables Checklist
- [ ] `mixture_quantile.py` with documented public API.
- [ ] Unit tests passing for backend only.
- [ ] Short module docstring summarizing assumptions.
- [ ] Notes on any unresolved numerical limitations.

## Handoff to Next Agent
Provide these artifacts exactly:
1. Callable `icdf(u)` for scalar and batched `u`.
2. Callable `cdf(x)`.
3. Optional diagnostic API with convergence metadata.
4. Recommended default config values used in tests.

The next agent should not need to inspect solver internals.
