"""Step 02 manifest for Task 07 knot generation.

Required module:
- `numpyro_extras.quantile_knots` (expected file: `src/numpyro_extras/quantile_knots.py`)

Required public API:
- `QuantileKnotSet` immutable dataclass
- `build_quantile_knot_set(quantile_backend, knot_cfg, slope_cfg)`

Behavioral pass gates:
- deterministic knot generation for `uniform_u`, `logit_u`, and `hybrid` grids
- strict `u_knots` ordering and monotone-clean `x_knots`
- finite positive endpoint slope seeds
- metadata quality/diagnostics contract and invalid-config validation
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


MODULE_NAME = "numpyro_extras.quantile_knots"
KNOTSET_TYPE_NAME = "QuantileKnotSet"
KNOT_BUILDER_NAME = "build_quantile_knot_set"

REQUIRED_KNOTSET_FIELDS = {
    "u_knots",
    "x_knots",
    "du_dx_left",
    "du_dx_right",
    "meta",
}

REQUIRED_META_KEYS = {
    "grid_type",
    "num_knots",
    "cleanup_count",
    "min_du",
    "max_du",
    "min_dx",
    "max_dx",
    "non_finite_count",
    "small_num_knots_warning",
    "endpoint_slope_role",
}


def _import_quantile_knots_module() -> ModuleType:
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.quantile_knots` "
                "(expected at `src/numpyro_extras/quantile_knots.py`)."
            )
        raise


def _get_required_api():
    module = _import_quantile_knots_module()
    knotset_type = getattr(module, KNOTSET_TYPE_NAME, None)
    if knotset_type is None:
        pytest.fail("Missing `QuantileKnotSet` in `numpyro_extras.quantile_knots`.")

    builder = getattr(module, KNOT_BUILDER_NAME, None)
    if builder is None:
        pytest.fail(
            "Missing `build_quantile_knot_set` in `numpyro_extras.quantile_knots`."
        )
    if not callable(builder):
        pytest.fail("`build_quantile_knot_set` must be callable.")
    return knotset_type, builder


def _make_knot_cfg(**overrides):
    cfg = dict(
        num_knots=128,
        u_min=1e-6,
        u_max=1.0 - 1e-6,
        grid_type="logit_u",
        tail_density=0.35,
        min_delta_u=1e-10,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _make_slope_cfg(**overrides):
    cfg = dict(
        method="finite_diff_u",
        delta_u=5e-5,
        min_positive_slope=1e-8,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _extract_field(obj, field_name: str):
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if isinstance(obj, Mapping) and field_name in obj:
        return obj[field_name]
    pytest.fail(f"Missing required field `{field_name}` in QuantileKnotSet return value.")


def _build_knot_set(backend, knot_cfg=None, slope_cfg=None):
    _, builder = _get_required_api()
    knot_cfg = knot_cfg or _make_knot_cfg()
    slope_cfg = slope_cfg or _make_slope_cfg()
    try:
        return builder(
            quantile_backend=backend,
            knot_cfg=knot_cfg,
            slope_cfg=slope_cfg,
        )
    except TypeError as exc:
        pytest.fail(
            "Expected `build_quantile_knot_set(quantile_backend=..., "
            "knot_cfg=..., slope_cfg=...)` signature."
            f" Received TypeError: {exc}"
        )


def _assert_basic_knot_validity(knot_set, *, u_min: float, u_max: float):
    u_knots = jnp.asarray(_extract_field(knot_set, "u_knots"), dtype=jnp.float64)
    x_knots = jnp.asarray(_extract_field(knot_set, "x_knots"), dtype=jnp.float64)
    du_dx_left = float(_extract_field(knot_set, "du_dx_left"))
    du_dx_right = float(_extract_field(knot_set, "du_dx_right"))

    assert u_knots.ndim == 1, "`u_knots` must be rank-1."
    assert x_knots.ndim == 1, "`x_knots` must be rank-1."
    assert u_knots.shape == x_knots.shape, (
        "`u_knots` and `x_knots` must have identical lengths."
    )
    assert u_knots.shape[0] >= 3, "Need at least 3 knots after cleanup."

    assert jnp.all(jnp.isfinite(u_knots)), "`u_knots` must be finite."
    assert jnp.all(jnp.isfinite(x_knots)), "`x_knots` must be finite."
    assert jnp.isfinite(du_dx_left), "`du_dx_left` must be finite."
    assert jnp.isfinite(du_dx_right), "`du_dx_right` must be finite."
    assert du_dx_left > 0.0, "`du_dx_left` must be strictly positive."
    assert du_dx_right > 0.0, "`du_dx_right` must be strictly positive."

    assert jnp.all(jnp.diff(u_knots) > 0.0), (
        "`u_knots` must be strictly increasing: jnp.all(jnp.diff(u_knots) > 0) failed."
    )
    assert jnp.all(jnp.diff(x_knots) >= 0.0), (
        "`x_knots` must be nondecreasing after cleanup: "
        "jnp.all(jnp.diff(x_knots) >= 0) failed."
    )

    assert abs(float(u_knots[0]) - u_min) <= 2e-5, (
        "Left knot does not track configured `u_min` within tolerance."
    )
    assert abs(float(u_knots[-1]) - u_max) <= 2e-5, (
        "Right knot does not track configured `u_max` within tolerance."
    )


def _tail_to_center_du_ratio(u_knots: jnp.ndarray) -> float:
    du = jnp.diff(u_knots)
    n = int(du.shape[0])
    window = max(3, n // 12)
    center_start = (n - window) // 2
    center = du[center_start : center_start + window]
    tails = jnp.concatenate([du[:window], du[-window:]])
    return float(jnp.mean(tails) / jnp.mean(center))


class _SmoothBackend:
    def icdf(self, u):
        u = jnp.asarray(u, dtype=jnp.float64)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        return 0.65 * jnp.log(u / (1.0 - u)) + 0.20 * u + 0.05 * jnp.sin(7.0 * u)


class _NoisyBackend:
    def icdf(self, u):
        u = jnp.asarray(u, dtype=jnp.float64)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        return 0.35 * jnp.log(u / (1.0 - u)) + 0.10 * u + 0.02 * jnp.sin(180.0 * u)


class _PlateauBackend:
    def icdf(self, u):
        u = jnp.asarray(u, dtype=jnp.float64)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        middle = (u - 0.08) / (0.92 - 0.08)
        return jnp.where(u < 0.08, 0.0, jnp.where(u > 0.92, 1.0, middle))


class _NonFiniteBackend:
    def icdf(self, u):
        u = jnp.asarray(u, dtype=jnp.float64)
        u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        base = 0.45 * jnp.log(u / (1.0 - u)) + 0.15 * u
        bad_mask = ((u > 0.23) & (u < 0.29)) | ((u > 0.71) & (u < 0.76))
        return jnp.where(bad_mask, jnp.nan, base)


def test_quantile_knotset_schema_contract():
    knotset_type, builder = _get_required_api()

    assert dataclasses.is_dataclass(knotset_type), (
        "`QuantileKnotSet` must be declared as a dataclass."
    )
    dataclass_params = getattr(knotset_type, "__dataclass_params__", None)
    assert dataclass_params is not None and dataclass_params.frozen, (
        "`QuantileKnotSet` must be immutable (`@dataclass(frozen=True)`)."
    )

    signature = inspect.signature(builder)
    for param_name in ("quantile_backend", "knot_cfg", "slope_cfg"):
        assert param_name in signature.parameters, (
            "`build_quantile_knot_set` must accept "
            f"`{param_name}` as an explicit argument."
        )

    field_names = {field.name for field in dataclasses.fields(knotset_type)}
    missing_fields = REQUIRED_KNOTSET_FIELDS.difference(field_names)
    assert not missing_fields, (
        "QuantileKnotSet missing required fields: "
        f"{sorted(missing_fields)}."
    )

    knot_cfg = _make_knot_cfg()
    knot_set = _build_knot_set(_SmoothBackend(), knot_cfg=knot_cfg)
    assert isinstance(knot_set, knotset_type), (
        "Builder must return a `QuantileKnotSet` instance."
    )
    _assert_basic_knot_validity(
        knot_set,
        u_min=knot_cfg.u_min,
        u_max=knot_cfg.u_max,
    )


def test_grid_modes_generate_valid_knots():
    backend = _SmoothBackend()
    ratios = {}

    for grid_type in ("uniform_u", "logit_u", "hybrid"):
        knot_cfg = _make_knot_cfg(
            grid_type=grid_type,
            num_knots=192,
            u_min=5e-6,
            u_max=1.0 - 5e-6,
        )
        knot_set = _build_knot_set(backend, knot_cfg=knot_cfg)
        _assert_basic_knot_validity(
            knot_set,
            u_min=knot_cfg.u_min,
            u_max=knot_cfg.u_max,
        )
        ratios[grid_type] = _tail_to_center_du_ratio(
            jnp.asarray(_extract_field(knot_set, "u_knots"), dtype=jnp.float64)
        )

    assert ratios["uniform_u"] > 0.8, (
        "uniform_u should have near-uniform spacing (tail/center ratio too small)."
    )
    assert ratios["logit_u"] < ratios["uniform_u"], (
        "logit_u must allocate denser tails than uniform_u."
    )
    assert ratios["hybrid"] < ratios["uniform_u"], (
        "hybrid must allocate denser tails than uniform_u."
    )


def test_knots_are_strictly_ordered_and_monotone():
    knot_cfg = _make_knot_cfg(grid_type="logit_u", num_knots=224)
    knot_set = _build_knot_set(_NoisyBackend(), knot_cfg=knot_cfg)
    _assert_basic_knot_validity(
        knot_set,
        u_min=knot_cfg.u_min,
        u_max=knot_cfg.u_max,
    )

    meta = _extract_field(knot_set, "meta")
    assert isinstance(meta, Mapping), "`meta` must be a mapping."
    assert "cleanup_count" in meta, (
        "`meta` must contain `cleanup_count` to report monotonic repairs."
    )
    assert int(meta["cleanup_count"]) > 0, (
        "Expected noisy backend to trigger monotonic cleanup; "
        "`meta['cleanup_count']` must be > 0."
    )


def test_endpoint_slope_estimation_is_positive_and_finite():
    min_positive = 1e-4
    slope_cfg = _make_slope_cfg(min_positive_slope=min_positive, delta_u=1e-5)
    knot_set = _build_knot_set(_PlateauBackend(), slope_cfg=slope_cfg)

    du_dx_left = float(_extract_field(knot_set, "du_dx_left"))
    du_dx_right = float(_extract_field(knot_set, "du_dx_right"))
    assert jnp.isfinite(du_dx_left), "`du_dx_left` must be finite for duplicate edge x."
    assert jnp.isfinite(du_dx_right), "`du_dx_right` must be finite for duplicate edge x."
    assert du_dx_left > 0.0 and du_dx_right > 0.0, (
        "Endpoint slopes must remain positive even when edge `dx` is tiny."
    )
    assert du_dx_left >= min_positive and du_dx_right >= min_positive, (
        "`min_positive_slope` floor was not respected at endpoints."
    )

    meta = _extract_field(knot_set, "meta")
    assert isinstance(meta, Mapping), "`meta` must be a mapping."
    assert "endpoint_slope_role" in meta, (
        "`meta` must document endpoint slope role "
        "(seed-only for non-C1 fallback)."
    )
    role = str(meta["endpoint_slope_role"]).lower()
    assert "seed" in role and "non_c1" in role, (
        "endpoint slope metadata must state these are seed slopes for non-C1 fallback."
    )


def test_metadata_contains_quality_metrics():
    knot_cfg = _make_knot_cfg(grid_type="hybrid", num_knots=160)
    knot_set = _build_knot_set(_SmoothBackend(), knot_cfg=knot_cfg)

    meta = _extract_field(knot_set, "meta")
    assert isinstance(meta, Mapping), "`meta` must be a mapping."
    missing_meta_keys = REQUIRED_META_KEYS.difference(meta.keys())
    assert not missing_meta_keys, (
        "Missing required metadata keys: "
        f"{sorted(missing_meta_keys)}."
    )

    u_knots = jnp.asarray(_extract_field(knot_set, "u_knots"), dtype=jnp.float64)
    x_knots = jnp.asarray(_extract_field(knot_set, "x_knots"), dtype=jnp.float64)
    du = jnp.diff(u_knots)
    dx = jnp.diff(x_knots)

    assert str(meta["grid_type"]) == knot_cfg.grid_type, (
        "`meta['grid_type']` must match requested grid_type."
    )
    assert int(meta["num_knots"]) == int(u_knots.shape[0]), (
        "`meta['num_knots']` must equal final knot count."
    )
    assert int(meta["non_finite_count"]) >= 0, "`non_finite_count` must be non-negative."
    assert int(meta["cleanup_count"]) >= 0, "`cleanup_count` must be non-negative."
    assert float(meta["min_du"]) == pytest.approx(float(jnp.min(du))), (
        "`meta['min_du']` inconsistent with generated knot spacing."
    )
    assert float(meta["max_du"]) == pytest.approx(float(jnp.max(du))), (
        "`meta['max_du']` inconsistent with generated knot spacing."
    )
    assert float(meta["min_dx"]) == pytest.approx(float(jnp.min(dx))), (
        "`meta['min_dx']` inconsistent with generated knot spacing."
    )
    assert float(meta["max_dx"]) == pytest.approx(float(jnp.max(dx))), (
        "`meta['max_dx']` inconsistent with generated knot spacing."
    )
    assert float(meta["max_du"]) >= float(meta["min_du"]), "`max_du` must be >= `min_du`."
    assert float(meta["max_dx"]) >= float(meta["min_dx"]), "`max_dx` must be >= `min_dx`."
    assert not bool(meta["small_num_knots_warning"]), (
        "Warning flag should be False for num_knots >= 16."
    )

    small_cfg = _make_knot_cfg(num_knots=12)
    small_knot_set = _build_knot_set(_SmoothBackend(), knot_cfg=small_cfg)
    small_meta = _extract_field(small_knot_set, "meta")
    assert bool(small_meta["small_num_knots_warning"]), (
        "Expected `small_num_knots_warning=True` when num_knots < 16."
    )


def test_nonfinite_quantiles_are_reported_and_handled():
    knot_cfg = _make_knot_cfg(num_knots=200, grid_type="logit_u")
    knot_set_1 = _build_knot_set(_NonFiniteBackend(), knot_cfg=knot_cfg)
    knot_set_2 = _build_knot_set(_NonFiniteBackend(), knot_cfg=knot_cfg)

    _assert_basic_knot_validity(
        knot_set_1,
        u_min=knot_cfg.u_min,
        u_max=knot_cfg.u_max,
    )
    meta_1 = _extract_field(knot_set_1, "meta")
    assert int(meta_1["non_finite_count"]) > 0, (
        "Expected non-finite quantiles to be counted in metadata."
    )

    u1 = jnp.asarray(_extract_field(knot_set_1, "u_knots"), dtype=jnp.float64)
    x1 = jnp.asarray(_extract_field(knot_set_1, "x_knots"), dtype=jnp.float64)
    u2 = jnp.asarray(_extract_field(knot_set_2, "u_knots"), dtype=jnp.float64)
    x2 = jnp.asarray(_extract_field(knot_set_2, "x_knots"), dtype=jnp.float64)
    assert u1.shape == u2.shape and x1.shape == x2.shape, (
        "Knot generation must be deterministic under fixed backend and config."
    )
    assert jnp.allclose(u1, u2, atol=0.0, rtol=0.0), (
        "u-knots changed across identical deterministic runs."
    )
    assert jnp.allclose(x1, x2, atol=0.0, rtol=0.0), (
        "x-knots changed across identical deterministic runs."
    )


def test_invalid_configs_raise_useful_errors():
    backend = _SmoothBackend()

    with pytest.raises((ValueError, TypeError), match=r"(?i)grid|grid_type|uniform_u|logit_u|hybrid"):
        _build_knot_set(backend, knot_cfg=_make_knot_cfg(grid_type="bad-grid"))

    with pytest.raises((ValueError, TypeError), match=r"(?i)u_min|u_max|bounds|order"):
        _build_knot_set(backend, knot_cfg=_make_knot_cfg(u_min=0.8, u_max=0.2))

    with pytest.raises((ValueError, TypeError), match=r"(?i)num_knots|knot|at least|>=|positive"):
        _build_knot_set(backend, knot_cfg=_make_knot_cfg(num_knots=1))
