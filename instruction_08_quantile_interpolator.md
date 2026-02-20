# Agent Task 08: Knot Set -> Quantile Interpolator with Sigmoid Tails

## Objective
Build a standalone interpolator object that provides fast, stable bidirectional maps:
- `x(u)` approximating inverse CDF (quantile)
- `u(x)` approximating CDF

Use interior interpolation over knots and ECDF-style sigmoid tails (`tanh`/`arctanh`) for smooth, stable endpoint extrapolation.

## Precondition
Task 03 test specification must be completed first, and Task 08 implementation must satisfy those tests.

## Scope
- In scope:
  - Interior interpolation for monotone data.
  - Left/right tail continuation formulas.
  - Bidirectional evaluation (`u -> x` and `x -> u`).
  - Derivative helpers for Jacobian computation.
- Out of scope:
  - True mixture root solving (already done in Agent 06).
  - NumPyro transform subclassing.

## Required Upstream Dependency
From Agent 07:
1. `QuantileKnotSet` containing `u_knots`, `x_knots`, `du_dx_left`, `du_dx_right`.

## Inputs (Contract)
1. `knot_set: QuantileKnotSet`
2. `interp_cfg`
   - `interior_method: Literal["akima", "linear", "pchip_like"]` (default `"akima"`)
   - `clip_u_eps: float` (default `1e-10`)
   - `safe_arctanh_eps: float` (default `1e-7`)
3. `tail_cfg`
   - `enforce_c1_stitch: bool` (default `True`)
   - `min_tail_scale: float` (default `1e-8`)

## Outputs (Contract)
Create an immutable object `QuantileInterpolator1D` exposing:

1. `icdf(u) -> x`  (full domain `u in (0,1)`)
2. `cdf(x) -> u`   (full real line)
3. `dudx(x) -> du/dx`
4. `dxdu(u) -> dx/du`
5. `log_abs_dxdu(u) -> log|dx/du|`
6. `stitch_points() -> dict` (returns endpoint knot data)

All outputs must be JAX-compatible and vectorizable.

## File Targets
- Add module:
  - `src/numpyro_extras/quantile_interpolator.py`
- Add tests:
  - `tests/test_quantile_interpolator.py`

## Implementation Requirements

### 1) Interior interpolation
- Construct interior map from knots:
  - `x(u)` using `u_knots -> x_knots`.
  - `u(x)` using `x_knots -> u_knots`.
- Choose method supporting stable monotone behavior. If chosen method can overshoot, add clipping/guardrails.

### 2) Tail formulas (must mirror ECDF scaler concept)
Define:
- Left knot `(x0, u0)`, right knot `(xN, uN)`.
- Endpoint slopes in CDF space: `m0 = du/dx|x0`, `mN = du/dx|xN`.

CDF tails (`u(x)`):
1. Left:
   - `u = u0 + u0 * tanh((m0/u0) * (x - x0))`
2. Right:
   - `u = uN + (1-uN) * tanh((mN/(1-uN)) * (x - xN))`

Inverse tails (`x(u)`), exact inverse of above:
1. Left:
   - `x = x0 + (u0/m0) * arctanh((u-u0)/u0)`
2. Right:
   - `x = xN + ((1-uN)/mN) * arctanh((u-uN)/(1-uN))`

Requirements:
- Preserve continuity at `x0`/`xN` and `u0`/`uN`.
- Ensure safe `arctanh` input clipping to `(-1+eps, 1-eps)`.

### 3) Piecewise stitching
- For `cdf(x)`:
  - left tail for `x < x0`
  - interior interpolator for `x0 <= x <= xN`
  - right tail for `x > xN`
- For `icdf(u)`:
  - left inverse-tail for `u < u0`
  - interior interpolator for `u0 <= u <= uN`
  - right inverse-tail for `u > uN`

### 4) Derivatives
- Implement `dudx(x)` and `dxdu(u)`:
  - Prefer analytic derivatives in tails.
  - For interior, use either interpolator derivative if available or `jax.grad`.
- Ensure positivity (monotonic bijection) up to tolerance.

### 5) Numerical protections
- Clip `u` to `[clip_u_eps, 1-clip_u_eps]` before unstable operations.
- Guard denominators involving `u0`, `1-uN`, and slopes with floors.
- Provide clear error if knot_set invalid (non-monotone or out-of-domain).

## Edge Cases to Handle
1. Very small `u0` or very large `uN`.
2. Near-zero endpoint slopes.
3. Dense knots with tiny spacing.
4. Inputs outside nominal domain due to upstream noise.

## Validation and Acceptance Tests
In `tests/test_quantile_interpolator.py`:

1. **Roundtrip on dense grid**
   - `cdf(icdf(u))` close to `u`.
2. **Reverse roundtrip**
   - `icdf(cdf(x))` close to `x` for representative `x`.
3. **Monotonicity**
   - `icdf` increasing in `u`; `cdf` increasing in `x`.
4. **Stitch continuity**
   - value and slope continuity at left/right stitch points.
5. **Tail finiteness**
   - no NaN/Inf near extreme probabilities.

Suggested tolerances:
- interior roundtrip error `< 1e-4`
- near tails `< 5e-4`

## Deliverables Checklist
- [ ] `QuantileInterpolator1D` API implemented.
- [ ] Piecewise bidirectional mapping with sigmoid tails.
- [ ] Derivative and log-derivative helpers.
- [ ] Comprehensive unit tests.

## Handoff to Next Agent
Deliver:
1. `QuantileInterpolator1D` object interface.
2. Stable defaults for interpolation and safety epsilons.
3. Known approximation error envelope from tests.

Next agent should only wrap this object as NumPyro transforms.
