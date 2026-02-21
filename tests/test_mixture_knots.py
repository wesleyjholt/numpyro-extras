"""Tests for interpolation-knot generation from a quantile backend (Task 07 gates).

Required API names (strict):
1. ``numpyro_extras.mixture_knots.QuantileKnotSet``
2. ``numpyro_extras.mixture_knots.build_quantile_knot_set(...)``

Expected builder contract:
- ``build_quantile_knot_set(quantile_backend, knot_cfg, slope_cfg)``
- ``knot_cfg`` and ``slope_cfg`` are config objects (dict-like is accepted).

Required ``QuantileKnotSet`` fields:
1. ``u_knots``
2. ``x_knots``
3. ``du_dx_left``
4. ``du_dx_right``
5. ``meta``

Required metadata keys (strict):
1. ``grid_type``
2. ``point_count``
3. ``cleanup_count``
4. ``monotonicity_violations``
5. ``non_finite_count``
6. ``min_du``
7. ``max_du``
8. ``min_dx``
9. ``max_dx``
10. ``small_n_warning``
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


REQUIRED_META_KEYS = (
    "grid_type",
    "point_count",
    "cleanup_count",
    "monotonicity_violations",
    "non_finite_count",
    "min_du",
    "max_du",
    "min_dx",
    "max_dx",
    "small_n_warning",
)


def _require_knots_module():
    try:
        module = importlib.import_module("numpyro_extras.mixture_knots")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.mixture_knots` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )
    return module


def _require_api():
    module = _require_knots_module()
    if not hasattr(module, "QuantileKnotSet"):
        pytest.fail("Expected `numpyro_extras.mixture_knots.QuantileKnotSet` to exist.")
    if not hasattr(module, "build_quantile_knot_set"):
        pytest.fail(
            "Expected `numpyro_extras.mixture_knots.build_quantile_knot_set` to exist."
        )
    return module, module.build_quantile_knot_set


def _default_knot_cfg(**overrides):
    cfg = {
        "num_knots": 256,
        "u_min": 1e-6,
        "u_max": 1.0 - 1e-6,
        "grid_type": "logit_u",
        "tail_density": 0.35,
        "min_delta_u": 1e-10,
    }
    cfg.update(overrides)
    return cfg


def _default_slope_cfg(**overrides):
    cfg = {
        "method": "finite_diff_u",
        "delta_u": 5e-5,
        "min_positive_slope": 1e-8,
    }
    cfg.update(overrides)
    return cfg


def _build_knot_set(backend, *, knot_cfg=None, slope_cfg=None):
    module, builder = _require_api()
    output = builder(
        quantile_backend=backend,
        knot_cfg=_default_knot_cfg(**(knot_cfg or {})),
        slope_cfg=_default_slope_cfg(**(slope_cfg or {})),
    )
    assert isinstance(output, module.QuantileKnotSet)
    return output


def _to_float(x):
    return float(np.asarray(x))


def _assert_required_meta(meta):
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    assert not missing, f"Missing metadata keys: {missing}"


class SmoothLogitBackend:
    def icdf(self, u):
        u = jnp.clip(jnp.asarray(u), 1e-12, 1.0 - 1e-12)
        return jnp.log(u) - jnp.log1p(-u)


class NoisyMonotonicityViolationBackend:
    def __init__(self, amplitude=0.08, frequency=90.0):
        self.amplitude = amplitude
        self.frequency = frequency

    def icdf(self, u):
        u = jnp.clip(jnp.asarray(u), 1e-12, 1.0 - 1e-12)
        base = jnp.log(u) - jnp.log1p(-u)
        return base + self.amplitude * jnp.sin(self.frequency * u)


class DuplicateXBackend:
    def icdf(self, u):
        u = jnp.clip(jnp.asarray(u), 1e-12, 1.0 - 1e-12)
        base = jnp.log(u) - jnp.log1p(-u)
        return jnp.round(base * 5.0) / 5.0


class NonFiniteBackend:
    def icdf(self, u):
        u = jnp.clip(jnp.asarray(u), 1e-12, 1.0 - 1e-12)
        base = jnp.log(u) - jnp.log1p(-u)
        is_bad = (u > 0.35) & (u < 0.45)
        return jnp.where(is_bad, jnp.nan, base)


def test_quantile_knotset_schema_contract():
    knot_set = _build_knot_set(SmoothLogitBackend())

    for name in ("u_knots", "x_knots", "du_dx_left", "du_dx_right", "meta"):
        assert hasattr(knot_set, name), f"QuantileKnotSet missing required field `{name}`."

    assert knot_set.u_knots.ndim == 1
    assert knot_set.x_knots.ndim == 1
    assert knot_set.u_knots.shape == knot_set.x_knots.shape
    assert knot_set.u_knots.size >= 2
    assert jnp.all(jnp.isfinite(knot_set.u_knots))
    assert jnp.all(jnp.isfinite(knot_set.x_knots))
    assert jnp.isfinite(knot_set.du_dx_left)
    assert jnp.isfinite(knot_set.du_dx_right)
    assert knot_set.du_dx_left > 0.0
    assert knot_set.du_dx_right > 0.0

    _assert_required_meta(knot_set.meta)
    assert knot_set.meta["grid_type"] == "logit_u"
    assert int(knot_set.meta["point_count"]) == int(knot_set.u_knots.size)


@pytest.mark.parametrize("grid_type", ["uniform_u", "logit_u", "hybrid"])
def test_grid_modes_generate_valid_knots(grid_type):
    knot_cfg = _default_knot_cfg(grid_type=grid_type, num_knots=257, u_min=1e-6, u_max=1.0 - 1e-6)
    knot_set = _build_knot_set(SmoothLogitBackend(), knot_cfg=knot_cfg)

    u_knots = knot_set.u_knots
    x_knots = knot_set.x_knots
    du = np.asarray(jnp.diff(u_knots))
    assert np.all(du > 0.0)
    assert np.all(np.asarray(jnp.diff(x_knots)) >= 0.0)
    assert _to_float(u_knots[0]) <= 5e-6
    assert _to_float(u_knots[-1]) >= 1.0 - 5e-6

    edge_count = max(4, len(du) // 16)
    center_count = max(4, len(du) // 16)
    center_start = len(du) // 2 - center_count // 2
    center_end = center_start + center_count
    center_du = du[center_start:center_end]
    tail_du = np.concatenate([du[:edge_count], du[-edge_count:]])

    if grid_type == "uniform_u":
        assert np.std(du) / np.mean(du) < 0.05
    else:
        assert np.mean(tail_du) < np.mean(center_du)


def test_knots_are_strictly_ordered_and_monotone():
    backend = NoisyMonotonicityViolationBackend()
    knot_set = _build_knot_set(backend, knot_cfg={"grid_type": "uniform_u", "num_knots": 256})

    raw_x = backend.icdf(knot_set.u_knots)
    raw_violations = int(jnp.sum(jnp.diff(raw_x) < 0.0))
    assert raw_violations > 0, "Fixture must inject monotonicity violations in raw samples."

    assert jnp.all(jnp.diff(knot_set.u_knots) > 0.0)
    assert jnp.all(jnp.diff(knot_set.x_knots) >= 0.0)
    assert int(knot_set.meta["monotonicity_violations"]) > 0
    assert int(knot_set.meta["cleanup_count"]) > 0


def test_endpoint_slope_estimation_is_positive_and_finite():
    slope_floor = 1e-4
    knot_set = _build_knot_set(
        DuplicateXBackend(),
        knot_cfg={"num_knots": 128},
        slope_cfg={"min_positive_slope": slope_floor},
    )

    assert jnp.isfinite(knot_set.du_dx_left)
    assert jnp.isfinite(knot_set.du_dx_right)
    assert knot_set.du_dx_left > 0.0
    assert knot_set.du_dx_right > 0.0
    assert knot_set.du_dx_left >= slope_floor
    assert knot_set.du_dx_right >= slope_floor

    raw_x = DuplicateXBackend().icdf(knot_set.u_knots)
    assert jnp.any(jnp.abs(jnp.diff(raw_x)) <= 1e-7), "Fixture must include duplicate or near-duplicate x."


def test_metadata_contains_quality_metrics():
    knot_set = _build_knot_set(
        SmoothLogitBackend(),
        knot_cfg={"grid_type": "hybrid", "num_knots": 192, "tail_density": 0.4},
    )

    meta = knot_set.meta
    _assert_required_meta(meta)

    du = jnp.diff(knot_set.u_knots)
    dx = jnp.diff(knot_set.x_knots)
    assert meta["grid_type"] == "hybrid"
    assert int(meta["point_count"]) == int(knot_set.u_knots.size)
    assert int(meta["non_finite_count"]) >= 0
    assert _to_float(meta["min_du"]) == pytest.approx(_to_float(jnp.min(du)), rel=1e-3, abs=1e-12)
    assert _to_float(meta["max_du"]) == pytest.approx(_to_float(jnp.max(du)), rel=1e-3, abs=1e-12)
    assert _to_float(meta["min_dx"]) == pytest.approx(_to_float(jnp.min(dx)), rel=1e-3, abs=1e-12)
    assert _to_float(meta["max_dx"]) == pytest.approx(_to_float(jnp.max(dx)), rel=1e-3, abs=1e-12)


def test_nonfinite_quantiles_are_reported_and_handled():
    knot_set = _build_knot_set(NonFiniteBackend(), knot_cfg={"num_knots": 128, "grid_type": "uniform_u"})

    assert jnp.all(jnp.isfinite(knot_set.u_knots))
    assert jnp.all(jnp.isfinite(knot_set.x_knots))
    assert jnp.all(jnp.diff(knot_set.u_knots) > 0.0)
    assert jnp.all(jnp.diff(knot_set.x_knots) >= 0.0)
    assert int(knot_set.meta["non_finite_count"]) > 0
    assert int(knot_set.meta["point_count"]) < 128

    small_n = _build_knot_set(SmoothLogitBackend(), knot_cfg={"num_knots": 12})
    assert bool(small_n.meta["small_n_warning"]) is True


@pytest.mark.parametrize(
    "knot_cfg,slope_cfg,error_fragment",
    [
        ({"grid_type": "invalid_mode"}, {}, "grid"),
        ({"u_min": 0.9, "u_max": 0.1}, {}, "u_"),
        ({"num_knots": 1}, {}, "num_knots"),
        ({"min_delta_u": 0.0}, {}, "min_delta_u"),
        ({}, {"method": "bad_method"}, "method"),
        ({}, {"min_positive_slope": 0.0}, "min_positive_slope"),
    ],
)
def test_invalid_configs_raise_useful_errors(knot_cfg, slope_cfg, error_fragment):
    with pytest.raises(Exception) as excinfo:
        _build_knot_set(SmoothLogitBackend(), knot_cfg=knot_cfg, slope_cfg=slope_cfg)
    assert error_fragment in str(excinfo.value).lower()


def test_jax_jit_and_vectorized_backend_interactions_are_deterministic():
    knot_cfg = _default_knot_cfg(grid_type="uniform_u", num_knots=96)
    slope_cfg = _default_slope_cfg()
    backend = SmoothLogitBackend()

    eager_a = _build_knot_set(backend, knot_cfg=knot_cfg, slope_cfg=slope_cfg)
    eager_b = _build_knot_set(backend, knot_cfg=knot_cfg, slope_cfg=slope_cfg)
    np.testing.assert_allclose(np.asarray(eager_a.u_knots), np.asarray(eager_b.u_knots), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(eager_a.x_knots), np.asarray(eager_b.x_knots), rtol=0.0, atol=0.0)

    _, builder = _require_api()
    try:
        jitted_builder = jax.jit(
            lambda: builder(quantile_backend=backend, knot_cfg=knot_cfg, slope_cfg=slope_cfg)
        )
        jitted = jitted_builder()
    except Exception:
        pytest.skip("JIT on the builder is optional; skipping because API is not JIT-compatible.")

    np.testing.assert_allclose(np.asarray(jitted.u_knots), np.asarray(eager_a.u_knots), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(jitted.x_knots), np.asarray(eager_a.x_knots), rtol=1e-6, atol=1e-6)
