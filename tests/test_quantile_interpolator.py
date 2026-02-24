"""Step 03 manifest for Task 08 quantile interpolator + sigmoid tails.

Required module:
- `numpyro_extras.quantile_interpolator`
  (expected file: `src/numpyro_extras/quantile_interpolator.py`)

Required public API:
- `QuantileInterpolator1D`
- methods:
  `icdf(u)`, `cdf(x)`, `dudx(x)`, `dxdu(u)`, `log_abs_dxdu(u)`, `stitch_points()`

Behavioral pass gates:
- dense roundtrip checks (`cdf(icdf(u))` and `icdf(cdf(x))`)
- monotonicity and positive-derivative checks
- C1 continuity at stitch points with sigmoid tails
- finite/valid tail behavior for extreme inputs
- JAX `jit` + `vmap` compatibility (no tracer->NumPy fallback)
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


MODULE_NAME = "numpyro_extras.quantile_interpolator"
INTERPOLATOR_NAME = "QuantileInterpolator1D"


@dataclasses.dataclass(frozen=True)
class _SyntheticKnotSet:
    u_knots: jnp.ndarray
    x_knots: jnp.ndarray
    du_dx_left: float
    du_dx_right: float
    meta: Mapping[str, object]


def _import_quantile_interpolator_module() -> ModuleType:
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.quantile_interpolator` "
                "(expected at `src/numpyro_extras/quantile_interpolator.py`)."
            )
        raise


def _get_interpolator_type():
    module = _import_quantile_interpolator_module()
    interpolator_type = getattr(module, INTERPOLATOR_NAME, None)
    if interpolator_type is None:
        pytest.fail(
            "Missing `QuantileInterpolator1D` in "
            "`numpyro_extras.quantile_interpolator`."
        )
    if not callable(interpolator_type):
        pytest.fail("`QuantileInterpolator1D` must be callable.")
    return interpolator_type


def _make_interp_cfg(**overrides):
    cfg = dict(
        interior_method="akima",
        clip_u_eps=1e-10,
        safe_arctanh_eps=1e-7,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _make_tail_cfg(**overrides):
    cfg = dict(
        enforce_c1_stitch=True,
        min_tail_scale=1e-8,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _build_interpolator(knot_set, *, interp_cfg=None, tail_cfg=None):
    interpolator_type = _get_interpolator_type()
    interp_cfg = interp_cfg or _make_interp_cfg()
    tail_cfg = tail_cfg or _make_tail_cfg()
    try:
        return interpolator_type(
            knot_set=knot_set,
            interp_cfg=interp_cfg,
            tail_cfg=tail_cfg,
        )
    except TypeError as exc:
        pytest.fail(
            "Expected constructor signature "
            "`QuantileInterpolator1D(knot_set=..., interp_cfg=..., tail_cfg=...)`. "
            f"Received TypeError: {exc}"
        )


def _logit(u):
    u = jnp.asarray(u, dtype=jnp.float64)
    return jnp.log(u) - jnp.log1p(-u)


def _smooth_quantile_fn(u):
    u = jnp.asarray(u, dtype=jnp.float64)
    return 0.95 * _logit(u) + 0.18 * (u - 0.5) + 0.03 * jnp.sin(8.0 * u)


def _smooth_dxdu_fn(u):
    u = jnp.asarray(u, dtype=jnp.float64)
    return 0.95 / (u * (1.0 - u)) + 0.18 + 0.24 * jnp.cos(8.0 * u)


def _tail_stress_quantile_fn(u):
    u = jnp.asarray(u, dtype=jnp.float64)
    return 0.70 * _logit(u) + 0.06 * (u - 0.5) + 0.01 * jnp.sin(80.0 * u)


def _tail_stress_dxdu_fn(u):
    u = jnp.asarray(u, dtype=jnp.float64)
    return 0.70 / (u * (1.0 - u)) + 0.06 + 0.8 * jnp.cos(80.0 * u)


def _make_knot_set(
    *,
    u_knots: jnp.ndarray,
    quantile_fn,
    dxdu_fn,
    du_dx_left: float | None = None,
    du_dx_right: float | None = None,
    meta: Mapping[str, object] | None = None,
) -> _SyntheticKnotSet:
    u_knots = jnp.asarray(u_knots, dtype=jnp.float64)
    x_knots = jnp.asarray(quantile_fn(u_knots), dtype=jnp.float64)
    if du_dx_left is None:
        du_dx_left = float(1.0 / dxdu_fn(u_knots[0]))
    if du_dx_right is None:
        du_dx_right = float(1.0 / dxdu_fn(u_knots[-1]))
    return _SyntheticKnotSet(
        u_knots=u_knots,
        x_knots=x_knots,
        du_dx_left=float(du_dx_left),
        du_dx_right=float(du_dx_right),
        meta=meta or {},
    )


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _stitch_point_values(interpolator) -> dict[str, float]:
    stitch = interpolator.stitch_points()
    assert isinstance(stitch, Mapping), "`stitch_points()` must return a mapping."
    required = {"u0", "uN", "x0", "xN"}
    missing = required.difference(stitch.keys())
    assert not missing, (
        "`stitch_points()` missing required keys: "
        f"{sorted(missing)}."
    )
    return {k: float(stitch[k]) for k in required}


@pytest.fixture
def smooth_knot_set() -> _SyntheticKnotSet:
    u_knots = jnp.linspace(2e-4, 1.0 - 2e-4, 257, dtype=jnp.float64)
    return _make_knot_set(
        u_knots=u_knots,
        quantile_fn=_smooth_quantile_fn,
        dxdu_fn=_smooth_dxdu_fn,
        meta={"fixture": "smooth_interior_case"},
    )


@pytest.fixture
def tail_stress_knot_set() -> _SyntheticKnotSet:
    u_min = 2e-10
    u_max = 1.0 - 2e-10
    z = jnp.linspace(_logit(u_min), _logit(u_max), 321, dtype=jnp.float64)
    u_knots = jax.nn.sigmoid(z)
    return _make_knot_set(
        u_knots=u_knots,
        quantile_fn=_tail_stress_quantile_fn,
        dxdu_fn=_tail_stress_dxdu_fn,
        meta={"fixture": "tail_stress_case"},
    )


@pytest.fixture
def near_degenerate_slope_knot_set() -> _SyntheticKnotSet:
    z = jnp.linspace(-8.5, 8.5, 193, dtype=jnp.float64)
    u_knots = jax.nn.sigmoid(z)
    x_knots = _smooth_quantile_fn(u_knots)
    # Tiny edge spacing creates large adjacent-knot secants. A correct C1 stitch
    # must match the interior interpolator derivative, not these edge secants.
    x_knots = x_knots.at[1].set(x_knots[0] + 1e-9)
    x_knots = x_knots.at[-2].set(x_knots[-1] - 1e-9)
    return _SyntheticKnotSet(
        u_knots=u_knots,
        x_knots=x_knots,
        du_dx_left=1e-14,
        du_dx_right=1e-14,
        meta={"fixture": "near_degenerate_slope_case"},
    )


def test_quantile_interpolator_api_contract(smooth_knot_set):
    interpolator_type = _get_interpolator_type()
    signature = inspect.signature(interpolator_type)
    assert "knot_set" in signature.parameters, (
        "`QuantileInterpolator1D` must accept `knot_set` as an explicit argument."
    )

    interpolator = _build_interpolator(smooth_knot_set)
    for method_name in (
        "icdf",
        "cdf",
        "dudx",
        "dxdu",
        "log_abs_dxdu",
        "stitch_points",
    ):
        method = getattr(interpolator, method_name, None)
        assert callable(method), f"Missing callable method `{method_name}`."

    x_scalar = jnp.asarray(0.15, dtype=jnp.float64)
    u_scalar = jnp.asarray(0.73, dtype=jnp.float64)
    assert jnp.shape(interpolator.cdf(x_scalar)) == (), "`cdf` must accept scalar x."
    assert jnp.shape(interpolator.icdf(u_scalar)) == (), "`icdf` must accept scalar u."
    assert jnp.shape(interpolator.dudx(x_scalar)) == (), "`dudx` must accept scalar x."
    assert jnp.shape(interpolator.dxdu(u_scalar)) == (), "`dxdu` must accept scalar u."
    assert jnp.shape(interpolator.log_abs_dxdu(u_scalar)) == (), (
        "`log_abs_dxdu` must accept scalar u."
    )

    x_batch = jnp.array([-2.5, 0.0, 3.4], dtype=jnp.float64)
    u_batch = jnp.array([0.1, 0.4, 0.9], dtype=jnp.float64)
    assert interpolator.cdf(x_batch).shape == x_batch.shape, "`cdf` must preserve shape."
    assert interpolator.icdf(u_batch).shape == u_batch.shape, "`icdf` must preserve shape."
    assert interpolator.dudx(x_batch).shape == x_batch.shape, "`dudx` must preserve shape."
    assert interpolator.dxdu(u_batch).shape == u_batch.shape, "`dxdu` must preserve shape."
    assert interpolator.log_abs_dxdu(u_batch).shape == u_batch.shape, (
        "`log_abs_dxdu` must preserve shape."
    )

    stitch = interpolator.stitch_points()
    assert isinstance(stitch, Mapping), "`stitch_points()` must return a mapping."
    for key in ("u0", "uN", "x0", "xN"):
        assert key in stitch, f"`stitch_points()` missing `{key}`."


def test_roundtrip_cdf_icdf_dense_grid(smooth_knot_set):
    interpolator = _build_interpolator(smooth_knot_set)
    stitch = _stitch_point_values(interpolator)

    u_grid = jnp.linspace(1e-10, 1.0 - 1e-10, 3001, dtype=jnp.float64)
    x_grid = interpolator.icdf(u_grid)
    u_roundtrip = interpolator.cdf(x_grid)
    err = jnp.abs(u_roundtrip - u_grid)

    interior_mask = (u_grid > stitch["u0"] + 1e-4) & (u_grid < stitch["uN"] - 1e-4)
    interior_max_err = float(jnp.max(err[interior_mask]))
    tail_max_err = float(jnp.max(err[~interior_mask]))

    assert interior_max_err < 1e-4, (
        f"Interior roundtrip tolerance violated: max_err={interior_max_err:.3e}"
    )
    assert tail_max_err < 5e-4, (
        f"Tail roundtrip tolerance violated: max_err={tail_max_err:.3e}"
    )


def test_reverse_roundtrip_icdf_cdf_interior(smooth_knot_set):
    interpolator = _build_interpolator(smooth_knot_set)
    stitch = _stitch_point_values(interpolator)

    x0 = stitch["x0"]
    xN = stitch["xN"]
    span = xN - x0
    x_grid = jnp.linspace(x0 + 1e-3 * span, xN - 1e-3 * span, 2049, dtype=jnp.float64)
    x_roundtrip = interpolator.icdf(interpolator.cdf(x_grid))
    max_err = float(jnp.max(jnp.abs(x_roundtrip - x_grid)))

    assert max_err < 1e-4, (
        f"Reverse interior roundtrip tolerance violated: max_err={max_err:.3e}"
    )


def test_monotonicity_of_cdf_and_icdf(smooth_knot_set):
    interpolator = _build_interpolator(smooth_knot_set)
    stitch = _stitch_point_values(interpolator)

    u_grid = jnp.linspace(1e-10, 1.0 - 1e-10, 3001, dtype=jnp.float64)
    x_from_u = interpolator.icdf(u_grid)
    assert jnp.all(jnp.diff(x_from_u) >= -1e-10), "`icdf(u)` must be nondecreasing."

    x_grid = jnp.linspace(float(x_from_u[0]) - 3.0, float(x_from_u[-1]) + 3.0, 3001)
    u_from_x = interpolator.cdf(x_grid)
    assert jnp.all(jnp.diff(u_from_x) >= -1e-10), "`cdf(x)` must be nondecreasing."

    u_interior = jnp.linspace(stitch["u0"] + 3e-4, stitch["uN"] - 3e-4, 1024)
    x_interior = interpolator.icdf(u_interior)
    dxdu = interpolator.dxdu(u_interior)
    dudx = interpolator.dudx(x_interior)

    assert jnp.all(jnp.isfinite(dxdu)), "`dxdu` must be finite."
    assert jnp.all(jnp.isfinite(dudx)), "`dudx` must be finite."
    assert float(jnp.min(dxdu)) > 0.0, "`dxdu` must stay positive."
    assert float(jnp.min(dudx)) > 0.0, "`dudx` must stay positive."

    reciprocal_err = float(jnp.max(jnp.abs(dxdu * dudx - 1.0)))
    assert reciprocal_err < 5e-3, (
        f"`dxdu * dudx` should be near 1 in interior; max_err={reciprocal_err:.3e}"
    )


def test_stitch_value_continuity_left_and_right(smooth_knot_set):
    interpolator = _build_interpolator(smooth_knot_set)
    stitch = _stitch_point_values(interpolator)

    x0 = stitch["x0"]
    xN = stitch["xN"]
    u0 = stitch["u0"]
    uN = stitch["uN"]

    # Required stitch-point formulas:
    # left:  u = u0 + u0 * tanh((m0/u0) * (x - x0))
    # right: u = uN + (1-uN) * tanh((mN/(1-uN)) * (x - xN))
    # exact continuity requires cdf(x0)=u0 and cdf(xN)=uN.
    assert float(interpolator.cdf(x0)) == pytest.approx(u0, abs=1e-7, rel=1e-7)
    assert float(interpolator.cdf(xN)) == pytest.approx(uN, abs=1e-7, rel=1e-7)

    # Inverse-tail formulas at stitch points:
    # left:  x = x0 + (u0/m0) * arctanh((u-u0)/u0)
    # right: x = xN + ((1-uN)/mN) * arctanh((u-uN)/(1-uN))
    # exact continuity requires icdf(u0)=x0 and icdf(uN)=xN.
    assert float(interpolator.icdf(u0)) == pytest.approx(x0, abs=1e-7, rel=1e-7)
    assert float(interpolator.icdf(uN)) == pytest.approx(xN, abs=1e-7, rel=1e-7)

    eps_x = 1e-8 * max(1.0, abs(x0), abs(xN))
    assert abs(float(interpolator.cdf(x0 - eps_x)) - u0) < 1e-6
    assert abs(float(interpolator.cdf(x0 + eps_x)) - u0) < 1e-6
    assert abs(float(interpolator.cdf(xN - eps_x)) - uN) < 1e-6
    assert abs(float(interpolator.cdf(xN + eps_x)) - uN) < 1e-6


def test_stitch_slope_continuity_left_and_right(near_degenerate_slope_knot_set):
    interpolator = _build_interpolator(
        near_degenerate_slope_knot_set,
        interp_cfg=_make_interp_cfg(interior_method="akima"),
        tail_cfg=_make_tail_cfg(enforce_c1_stitch=True, min_tail_scale=1e-10),
    )
    stitch = _stitch_point_values(interpolator)

    x0 = stitch["x0"]
    xN = stitch["xN"]
    h = 1e-6 * max(1.0, abs(x0), abs(xN))

    left_tail_slope = float(interpolator.dudx(x0 - h))
    left_interior_slope = float(interpolator.dudx(x0 + h))
    right_interior_slope = float(interpolator.dudx(xN - h))
    right_tail_slope = float(interpolator.dudx(xN + h))

    assert _relative_error(left_tail_slope, left_interior_slope) < 1e-4, (
        "Left stitch slope discontinuity exceeds C1 tolerance."
    )
    assert _relative_error(right_tail_slope, right_interior_slope) < 1e-4, (
        "Right stitch slope discontinuity exceeds C1 tolerance."
    )

    # Finite-difference slope proxy around stitch should not spike.
    left_tail_fd = float((interpolator.cdf(x0 - h) - interpolator.cdf(x0 - 2.0 * h)) / h)
    left_interior_fd = float(
        (interpolator.cdf(x0 + 2.0 * h) - interpolator.cdf(x0 + h)) / h
    )
    right_interior_fd = float(
        (interpolator.cdf(xN - h) - interpolator.cdf(xN - 2.0 * h)) / h
    )
    right_tail_fd = float((interpolator.cdf(xN + 2.0 * h) - interpolator.cdf(xN + h)) / h)
    assert _relative_error(left_tail_fd, left_interior_fd) < 1e-4
    assert _relative_error(right_tail_fd, right_interior_fd) < 1e-4

    # Regression pin: adjacent-knot secants are intentionally pathological here.
    # C1 stitching must use interior derivative, not adjacent finite-difference
    # endpoint secants.
    u_knots = near_degenerate_slope_knot_set.u_knots
    x_knots = near_degenerate_slope_knot_set.x_knots
    secant_left = float((u_knots[1] - u_knots[0]) / (x_knots[1] - x_knots[0]))
    secant_right = float((u_knots[-1] - u_knots[-2]) / (x_knots[-1] - x_knots[-2]))
    assert _relative_error(left_interior_slope, secant_left) > 1e-2, (
        "Left boundary slope appears tied to adjacent-knot secant, "
        "not interior derivative."
    )
    assert _relative_error(right_interior_slope, secant_right) > 1e-2, (
        "Right boundary slope appears tied to adjacent-knot secant, "
        "not interior derivative."
    )


def test_tail_inputs_remain_finite_and_valid(tail_stress_knot_set):
    interpolator = _build_interpolator(
        tail_stress_knot_set,
        interp_cfg=_make_interp_cfg(clip_u_eps=1e-12, safe_arctanh_eps=1e-8),
    )
    stitch = _stitch_point_values(interpolator)

    u_extreme = jnp.array(
        [-1e-6, 0.0, 1e-15, 1e-12, 0.5, 1.0 - 1e-12, 1.0, 1.0 + 1e-6],
        dtype=jnp.float64,
    )
    x_extreme = interpolator.icdf(u_extreme)
    assert jnp.all(jnp.isfinite(x_extreme)), (
        "`icdf` must remain finite for extreme/perturbed u values."
    )

    probe_x = jnp.array(
        [
            stitch["x0"] - 12.0,
            stitch["x0"] - 6.0,
            stitch["x0"],
            0.0,
            stitch["xN"],
            stitch["xN"] + 6.0,
            stitch["xN"] + 12.0,
        ],
        dtype=jnp.float64,
    )
    u_probe = interpolator.cdf(probe_x)
    assert jnp.all(jnp.isfinite(u_probe)), "`cdf` must remain finite in tails."
    assert jnp.all((u_probe >= 0.0) & (u_probe <= 1.0)), (
        "`cdf` outputs must stay within [0, 1]."
    )

    moderate_tail_x = jnp.array(
        [stitch["x0"] - 4.0, stitch["xN"] + 4.0],
        dtype=jnp.float64,
    )
    moderate_tail_u = interpolator.cdf(moderate_tail_x)
    assert jnp.all((moderate_tail_u > 0.0) & (moderate_tail_u < 1.0)), (
        "Tail CDF should stay strictly inside (0, 1) away from asymptotic extremes."
    )


def test_jit_and_vmap_compatibility(smooth_knot_set):
    interpolator = _build_interpolator(smooth_knot_set)
    u = jnp.linspace(1e-8, 1.0 - 1e-8, 1024, dtype=jnp.float64)

    icdf_jit = jax.jit(interpolator.icdf)
    cdf_jit = jax.jit(interpolator.cdf)
    dudx_jit = jax.jit(interpolator.dudx)
    dxdu_jit = jax.jit(interpolator.dxdu)
    log_abs_dxdu_jit = jax.jit(interpolator.log_abs_dxdu)

    x = icdf_jit(u)
    u_roundtrip = cdf_jit(x)
    dudx_vals = dudx_jit(x)
    dxdu_vals = dxdu_jit(u)
    log_abs_dxdu_vals = log_abs_dxdu_jit(u)

    assert jnp.all(jnp.isfinite(x))
    assert jnp.all(jnp.isfinite(u_roundtrip))
    assert jnp.all(jnp.isfinite(dudx_vals))
    assert jnp.all(jnp.isfinite(dxdu_vals))
    assert jnp.all(jnp.isfinite(log_abs_dxdu_vals))
    assert float(jnp.max(jnp.abs(u_roundtrip - u))) < 1e-4

    vmapped_icdf = jax.vmap(lambda ui: interpolator.icdf(ui))(u)
    vmapped_cdf = jax.vmap(lambda xi: interpolator.cdf(xi))(x)
    assert jnp.allclose(vmapped_icdf, x, atol=1e-9, rtol=1e-9)
    assert jnp.allclose(vmapped_cdf, u_roundtrip, atol=1e-9, rtol=1e-9)

    # If runtime path falls back to NumPy host ops, tracer conversion will fail.
    cdf_jaxpr = jax.make_jaxpr(interpolator.cdf)(x)
    icdf_jaxpr = jax.make_jaxpr(interpolator.icdf)(u)
    assert len(cdf_jaxpr.jaxpr.eqns) > 0
    assert len(icdf_jaxpr.jaxpr.eqns) > 0
