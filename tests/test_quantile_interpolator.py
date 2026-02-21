"""Tests for QuantileInterpolator1D with sigmoid tails (Task 08 gates).

Required API names (strict):
1. ``numpyro_extras.quantile_interpolator.QuantileInterpolator1D``
2. ``numpyro_extras.mixture_knots.QuantileKnotSet`` (input schema)

Pass gates for Task 08 implementation:
1. Dense roundtrip consistency in interior and tails.
2. Monotone/bijective behavior with positive derivatives.
3. C1 continuity at stitch points for piecewise interior/tail transitions.
4. Tail stability for extreme probabilities and extreme x values.
5. ``jit`` and ``vmap`` compatibility for cdf/icdf/derivatives.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Task 08 suggested acceptance targets:
# - interior roundtrip < 1e-4
# - near tails < 5e-4
# - stitch value continuity < 1e-6
# - stitch slope continuity < 1e-4 (finite-difference proxy)
INTERIOR_TOL = 1e-4
TAIL_TOL = 5e-4
STITCH_VALUE_TOL = 1e-6
STITCH_SLOPE_TOL = 1e-4
POSITIVE_DERIVATIVE_FLOOR = 1e-12


def _logit(u):
    u = jnp.asarray(u, dtype=jnp.float32)
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    return jnp.log(u) - jnp.log1p(-u)


def _require_quantile_interpolator_module():
    try:
        module = importlib.import_module("numpyro_extras.quantile_interpolator")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.quantile_interpolator` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )
    return module


def _require_quantile_knotset_class():
    try:
        module = importlib.import_module("numpyro_extras.mixture_knots")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.mixture_knots` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )
    if not hasattr(module, "QuantileKnotSet"):
        pytest.fail("Expected `numpyro_extras.mixture_knots.QuantileKnotSet` to exist.")
    return module.QuantileKnotSet


def _require_interpolator_class():
    module = _require_quantile_interpolator_module()
    if not hasattr(module, "QuantileInterpolator1D"):
        pytest.fail("Expected `numpyro_extras.quantile_interpolator.QuantileInterpolator1D` to exist.")
    return module.QuantileInterpolator1D


def _make_knot_set(*, u_knots, x_knots, du_dx_left, du_dx_right, name):
    QuantileKnotSet = _require_quantile_knotset_class()
    return QuantileKnotSet(
        u_knots=jnp.asarray(u_knots, dtype=jnp.float32),
        x_knots=jnp.asarray(x_knots, dtype=jnp.float32),
        du_dx_left=jnp.asarray(du_dx_left, dtype=jnp.float32),
        du_dx_right=jnp.asarray(du_dx_right, dtype=jnp.float32),
        meta={"fixture_name": name},
    )


def _build_interpolator(knot_set, *, interp_cfg=None, tail_cfg=None):
    QuantileInterpolator1D = _require_interpolator_class()
    interp_cfg_base = {
        "interior_method": "akima",
        "clip_u_eps": 1e-10,
        "safe_arctanh_eps": 1e-7,
    }
    tail_cfg_base = {
        "enforce_c1_stitch": True,
        "min_tail_scale": 1e-8,
    }
    if interp_cfg:
        interp_cfg_base.update(interp_cfg)
    if tail_cfg:
        tail_cfg_base.update(tail_cfg)

    try:
        return QuantileInterpolator1D(
            knot_set,
            interp_cfg=interp_cfg_base,
            tail_cfg=tail_cfg_base,
        )
    except TypeError as exc:
        pytest.fail(
            "Expected constructor `QuantileInterpolator1D(knot_set, interp_cfg, tail_cfg)` "
            f"to be supported. Received TypeError: {exc}"
        )


def _extract_stitch_points(interpolator):
    stitch = interpolator.stitch_points()
    assert isinstance(stitch, dict), "`stitch_points()` must return a dict."
    required = ("u0", "uN", "x0", "xN", "du_dx_left", "du_dx_right")
    missing = [key for key in required if key not in stitch]
    assert not missing, f"`stitch_points()` missing expected keys: {missing}"
    return (
        float(np.asarray(stitch["u0"])),
        float(np.asarray(stitch["uN"])),
        float(np.asarray(stitch["x0"])),
        float(np.asarray(stitch["xN"])),
        float(np.asarray(stitch["du_dx_left"])),
        float(np.asarray(stitch["du_dx_right"])),
    )


@pytest.fixture(scope="module")
def smooth_interior_knot_set():
    u_knots = jnp.linspace(1e-3, 1.0 - 1e-3, 129, dtype=jnp.float32)
    x_knots = _logit(u_knots) + 0.08 * (u_knots - 0.5) ** 3
    du_dx_left = (u_knots[1] - u_knots[0]) / (x_knots[1] - x_knots[0])
    du_dx_right = (u_knots[-1] - u_knots[-2]) / (x_knots[-1] - x_knots[-2])
    return _make_knot_set(
        u_knots=u_knots,
        x_knots=x_knots,
        du_dx_left=du_dx_left,
        du_dx_right=du_dx_right,
        name="smooth_interior",
    )


@pytest.fixture(scope="module")
def tail_stress_knot_set():
    base = jax.nn.sigmoid(jnp.linspace(-18.0, 18.0, 121, dtype=jnp.float32))
    tail_dense = jnp.array(
        [
            1e-10,
            3e-10,
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1.0 - 1e-6,
            1.0 - 1e-7,
            1.0 - 1e-8,
            1.0 - 1e-9,
            1.0 - 3e-10,
            1.0 - 1e-10,
        ],
        dtype=jnp.float32,
    )
    u_knots = jnp.unique(jnp.sort(jnp.concatenate([base, tail_dense], axis=0)))
    x_knots = _logit(u_knots) + 0.05 * jnp.tanh(2.0 * (u_knots - 0.5))
    du_dx_left = (u_knots[1] - u_knots[0]) / (x_knots[1] - x_knots[0])
    du_dx_right = (u_knots[-1] - u_knots[-2]) / (x_knots[-1] - x_knots[-2])
    return _make_knot_set(
        u_knots=u_knots,
        x_knots=x_knots,
        du_dx_left=du_dx_left,
        du_dx_right=du_dx_right,
        name="tail_stress",
    )


@pytest.fixture(scope="module")
def near_degenerate_slope_knot_set():
    u_low = jnp.linspace(2e-4, 0.2, 45, dtype=jnp.float32)
    u_dense = 0.5 + jnp.linspace(-2e-5, 2e-5, 9, dtype=jnp.float32)
    u_high = jnp.linspace(0.8, 1.0 - 2e-4, 45, dtype=jnp.float32)
    u_knots = jnp.unique(jnp.sort(jnp.concatenate([u_low, u_dense, u_high], axis=0)))

    stretch = 2.0e5
    x_knots = stretch * _logit(u_knots)
    du_dx_left = (u_knots[1] - u_knots[0]) / (x_knots[1] - x_knots[0])
    du_dx_right = (u_knots[-1] - u_knots[-2]) / (x_knots[-1] - x_knots[-2])
    return _make_knot_set(
        u_knots=u_knots,
        x_knots=x_knots,
        du_dx_left=du_dx_left,
        du_dx_right=du_dx_right,
        name="near_degenerate",
    )


def test_quantile_interpolator_api_contract(smooth_interior_knot_set):
    interpolator = _build_interpolator(smooth_interior_knot_set)

    for name in ("icdf", "cdf", "dudx", "dxdu", "log_abs_dxdu", "stitch_points"):
        assert hasattr(interpolator, name), f"Interpolator is missing required method `{name}`."
        assert callable(getattr(interpolator, name)), f"`{name}` must be callable."

    u_scalar = jnp.array(0.3, dtype=jnp.float32)
    x_scalar = interpolator.icdf(u_scalar)
    cdf_scalar = interpolator.cdf(x_scalar)
    dudx_scalar = interpolator.dudx(x_scalar)
    dxdu_scalar = interpolator.dxdu(u_scalar)
    ladj_scalar = interpolator.log_abs_dxdu(u_scalar)

    assert jnp.shape(x_scalar) == ()
    assert jnp.shape(cdf_scalar) == ()
    assert jnp.shape(dudx_scalar) == ()
    assert jnp.shape(dxdu_scalar) == ()
    assert jnp.shape(ladj_scalar) == ()

    u_batch = jnp.linspace(0.1, 0.9, 17, dtype=jnp.float32)
    x_batch = interpolator.icdf(u_batch)
    assert x_batch.shape == u_batch.shape
    assert interpolator.cdf(x_batch).shape == x_batch.shape
    assert interpolator.dudx(x_batch).shape == x_batch.shape
    assert interpolator.dxdu(u_batch).shape == u_batch.shape
    assert interpolator.log_abs_dxdu(u_batch).shape == u_batch.shape

    _extract_stitch_points(interpolator)


def test_roundtrip_cdf_icdf_dense_grid(smooth_interior_knot_set):
    interpolator = _build_interpolator(smooth_interior_knot_set)
    u0, uN, _, _, _, _ = _extract_stitch_points(interpolator)

    u_dense = jnp.linspace(u0 + 1e-4, uN - 1e-4, 1024, dtype=jnp.float32)
    recovered = interpolator.cdf(interpolator.icdf(u_dense))
    max_error = float(jnp.max(jnp.abs(recovered - u_dense)))
    assert max_error < INTERIOR_TOL, (
        f"Interior roundtrip cdf(icdf(u)) error exceeded tolerance: {max_error} >= {INTERIOR_TOL}"
    )


def test_reverse_roundtrip_icdf_cdf_interior(smooth_interior_knot_set):
    interpolator = _build_interpolator(smooth_interior_knot_set)
    _, _, x0, xN, _, _ = _extract_stitch_points(interpolator)

    x_dense = jnp.linspace(x0 + 1e-3, xN - 1e-3, 1024, dtype=jnp.float32)
    recovered = interpolator.icdf(interpolator.cdf(x_dense))
    max_error = float(jnp.max(jnp.abs(recovered - x_dense)))
    assert max_error < INTERIOR_TOL, (
        f"Interior reverse-roundtrip icdf(cdf(x)) error exceeded tolerance: {max_error} >= {INTERIOR_TOL}"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["smooth_interior_knot_set", "tail_stress_knot_set", "near_degenerate_slope_knot_set"],
)
def test_monotonicity_of_cdf_and_icdf(request, fixture_name):
    knot_set = request.getfixturevalue(fixture_name)
    interpolator = _build_interpolator(knot_set)
    u0, uN, x0, xN, _, _ = _extract_stitch_points(interpolator)

    u_grid = jnp.linspace(max(1e-9, 0.5 * u0), min(1.0 - 1e-9, 1.0 - 0.5 * (1.0 - uN)), 1024)
    x_grid = jnp.linspace(x0 - 3.0, xN + 3.0, 1024)

    x_from_u = interpolator.icdf(u_grid)
    u_from_x = interpolator.cdf(x_grid)
    assert jnp.all(jnp.diff(x_from_u) >= -1e-7), "icdf(u) must be monotone increasing."
    assert jnp.all(jnp.diff(u_from_x) >= -1e-7), "cdf(x) must be monotone increasing."

    dudx = interpolator.dudx(x_grid)
    dxdu = interpolator.dxdu(u_grid)
    assert jnp.all(jnp.isfinite(dudx))
    assert jnp.all(jnp.isfinite(dxdu))
    assert jnp.all(dudx > POSITIVE_DERIVATIVE_FLOOR), "dudx(x) must remain strictly positive."
    assert jnp.all(dxdu > POSITIVE_DERIVATIVE_FLOOR), "dxdu(u) must remain strictly positive."


def test_stitch_value_continuity_left_and_right(smooth_interior_knot_set):
    interpolator = _build_interpolator(smooth_interior_knot_set)
    u0, uN, x0, xN, _, _ = _extract_stitch_points(interpolator)

    # Expected tail formulas at stitch boundaries (Task 08):
    # left:  u = u0 + u0 * tanh((m0/u0) * (x - x0)), x = x0 + (u0/m0) * arctanh((u-u0)/u0)
    # right: u = uN + (1-uN) * tanh((mN/(1-uN)) * (x - xN)),
    #        x = xN + ((1-uN)/mN) * arctanh((u-uN)/(1-uN))
    # At x=x0/xN and u=u0/uN, these evaluate exactly to u0/uN and x0/xN.
    eps_x = 1e-6
    eps_u = 1e-6

    np.testing.assert_allclose(interpolator.cdf(jnp.asarray(x0)), u0, atol=STITCH_VALUE_TOL, rtol=STITCH_VALUE_TOL)
    np.testing.assert_allclose(interpolator.cdf(jnp.asarray(xN)), uN, atol=STITCH_VALUE_TOL, rtol=STITCH_VALUE_TOL)
    np.testing.assert_allclose(interpolator.icdf(jnp.asarray(u0)), x0, atol=STITCH_VALUE_TOL, rtol=STITCH_VALUE_TOL)
    np.testing.assert_allclose(interpolator.icdf(jnp.asarray(uN)), xN, atol=STITCH_VALUE_TOL, rtol=STITCH_VALUE_TOL)

    assert abs(float(interpolator.cdf(x0 - eps_x)) - u0) < STITCH_VALUE_TOL
    assert abs(float(interpolator.cdf(x0 + eps_x)) - u0) < STITCH_VALUE_TOL
    assert abs(float(interpolator.cdf(xN - eps_x)) - uN) < STITCH_VALUE_TOL
    assert abs(float(interpolator.cdf(xN + eps_x)) - uN) < STITCH_VALUE_TOL
    assert abs(float(interpolator.icdf(u0 - eps_u)) - x0) < STITCH_VALUE_TOL
    assert abs(float(interpolator.icdf(u0 + eps_u)) - x0) < STITCH_VALUE_TOL
    assert abs(float(interpolator.icdf(uN - eps_u)) - xN) < STITCH_VALUE_TOL
    assert abs(float(interpolator.icdf(uN + eps_u)) - xN) < STITCH_VALUE_TOL


def test_stitch_slope_continuity_left_and_right(smooth_interior_knot_set):
    interpolator = _build_interpolator(smooth_interior_knot_set)
    _, _, x0, xN, m0, mN = _extract_stitch_points(interpolator)

    h = 1e-3
    left_slope_minus = float((interpolator.cdf(x0) - interpolator.cdf(x0 - h)) / h)
    left_slope_plus = float((interpolator.cdf(x0 + h) - interpolator.cdf(x0)) / h)
    right_slope_minus = float((interpolator.cdf(xN) - interpolator.cdf(xN - h)) / h)
    right_slope_plus = float((interpolator.cdf(xN + h) - interpolator.cdf(xN)) / h)

    assert abs(left_slope_minus - left_slope_plus) < STITCH_SLOPE_TOL
    assert abs(right_slope_minus - right_slope_plus) < STITCH_SLOPE_TOL
    assert abs(left_slope_minus - m0) < STITCH_SLOPE_TOL
    assert abs(left_slope_plus - m0) < STITCH_SLOPE_TOL
    assert abs(right_slope_minus - mN) < STITCH_SLOPE_TOL
    assert abs(right_slope_plus - mN) < STITCH_SLOPE_TOL

    x_window_left = x0 + jnp.linspace(-4.0 * h, 4.0 * h, 25, dtype=jnp.float32)
    x_window_right = xN + jnp.linspace(-4.0 * h, 4.0 * h, 25, dtype=jnp.float32)
    slope_window_left = jnp.diff(interpolator.cdf(x_window_left)) / jnp.diff(x_window_left)
    slope_window_right = jnp.diff(interpolator.cdf(x_window_right)) / jnp.diff(x_window_right)
    assert float(jnp.max(slope_window_left) - jnp.min(slope_window_left)) < 5e-4
    assert float(jnp.max(slope_window_right) - jnp.min(slope_window_right)) < 5e-4


@pytest.mark.parametrize("fixture_name", ["tail_stress_knot_set", "near_degenerate_slope_knot_set"])
def test_tail_inputs_remain_finite_and_valid(request, fixture_name):
    knot_set = request.getfixturevalue(fixture_name)
    interpolator = _build_interpolator(knot_set)
    u0, uN, x0, xN, _, _ = _extract_stitch_points(interpolator)

    # Includes exact 0 and 1 to verify arctanh-safety clipping behavior.
    u_extreme = jnp.array(
        [0.0, 1e-15, 1e-12, 1e-9, u0, 0.5, uN, 1.0 - 1e-9, 1.0 - 1e-12, 1.0 - 1e-15, 1.0],
        dtype=jnp.float32,
    )
    x_from_u = interpolator.icdf(u_extreme)
    u_recovered = interpolator.cdf(x_from_u)

    assert jnp.all(jnp.isfinite(x_from_u)), "Extreme-u icdf outputs must be finite."
    assert jnp.all(jnp.isfinite(u_recovered)), "cdf(icdf(u)) for extreme-u inputs must be finite."
    assert jnp.all((u_recovered > 0.0) & (u_recovered < 1.0))

    x_extreme = jnp.array([x0 - 8.0, x0 - 4.0, x0, xN, xN + 4.0, xN + 8.0], dtype=jnp.float32)
    u_from_x = interpolator.cdf(x_extreme)
    assert jnp.all(jnp.isfinite(u_from_x)), "Extreme-x cdf outputs must be finite."
    assert jnp.all((u_from_x > 0.0) & (u_from_x < 1.0))

    clipped_reference = jnp.clip(u_extreme, 1e-10, 1.0 - 1e-10)
    tail_roundtrip_err = float(jnp.max(jnp.abs(u_recovered - clipped_reference)))
    assert tail_roundtrip_err < TAIL_TOL, (
        "Tail roundtrip error exceeded tolerance after clip-aware comparison: "
        f"{tail_roundtrip_err} >= {TAIL_TOL}"
    )


def test_jit_and_vmap_compatibility(smooth_interior_knot_set):
    interpolator = _build_interpolator(smooth_interior_knot_set)

    u = jnp.linspace(1e-5, 1.0 - 1e-5, 256, dtype=jnp.float32)
    x = interpolator.icdf(u)

    cdf_jit = jax.jit(lambda value: interpolator.cdf(value))
    icdf_jit = jax.jit(lambda value: interpolator.icdf(value))
    dudx_jit = jax.jit(lambda value: interpolator.dudx(value))
    dxdu_jit = jax.jit(lambda value: interpolator.dxdu(value))
    ladj_jit = jax.jit(lambda value: interpolator.log_abs_dxdu(value))

    np.testing.assert_allclose(cdf_jit(x), interpolator.cdf(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(icdf_jit(u), interpolator.icdf(u), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dudx_jit(x), interpolator.dudx(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dxdu_jit(u), interpolator.dxdu(u), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ladj_jit(u), interpolator.log_abs_dxdu(u), rtol=1e-6, atol=1e-6)

    u_batch = jnp.stack([u, 0.8 * u + 0.1], axis=0)
    x_batch = jax.vmap(interpolator.icdf)(u_batch)
    recovered_batch = jax.vmap(interpolator.cdf)(x_batch)
    assert x_batch.shape == u_batch.shape
    assert recovered_batch.shape == u_batch.shape
    max_error = float(jnp.max(jnp.abs(recovered_batch - u_batch)))
    assert max_error < INTERIOR_TOL, f"vmap roundtrip error exceeded tolerance: {max_error} >= {INTERIOR_TOL}"
