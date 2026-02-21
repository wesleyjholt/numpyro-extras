"""Integration and acceptance tests for Task 10 mixture transform builder.

These tests intentionally define the contract for:
1. ``numpyro_extras.mixture_transform_builder.build_mixture_transform(...)``
2. Config dataclasses and diagnostics surfaced by the builder.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions import Categorical, MixtureSameFamily, Normal


ROUNDTRIP_INTERIOR_TOL = 2e-4
QUANTILE_ABS_TOL = 3e-2

CENTRAL_QUANTILE_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)

CASE_DATA = {
    "balanced_bimodal": dict(
        weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
        loc=jnp.array([-2.0, 2.0], dtype=jnp.float32),
        scale=jnp.array([0.8, 1.2], dtype=jnp.float32),
    ),
    "dominant_component": dict(
        weights=jnp.array([0.98, 0.02], dtype=jnp.float32),
        loc=jnp.array([0.0, 5.5], dtype=jnp.float32),
        scale=jnp.array([1.0, 0.45], dtype=jnp.float32),
    ),
    "tight_overlap": dict(
        weights=jnp.array([0.35, 0.65], dtype=jnp.float32),
        loc=jnp.array([-0.3, 0.2], dtype=jnp.float32),
        scale=jnp.array([0.35, 0.45], dtype=jnp.float32),
    ),
}

MIXTURE_CASES = [pytest.param(case, id=name) for name, case in CASE_DATA.items()]


def _require_builder_module():
    try:
        module = importlib.import_module("numpyro_extras.mixture_transform_builder")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.mixture_transform_builder` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )
    if not hasattr(module, "build_mixture_transform"):
        pytest.fail(
            "Expected `numpyro_extras.mixture_transform_builder.build_mixture_transform` to exist."
        )
    return module


def _require_transform_classes():
    try:
        module = importlib.import_module("numpyro_extras.transforms")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.transforms` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )
    missing = [
        name
        for name in ("UniformToMixtureTransform", "NormalToMixtureTransform")
        if not hasattr(module, name)
    ]
    if missing:
        pytest.fail(
            "Expected transform classes to exist in `numpyro_extras.transforms`: "
            f"missing {missing}."
        )
    return module.UniformToMixtureTransform, module.NormalToMixtureTransform


def _make_component_distribution(case):
    return Normal(loc=case["loc"], scale=case["scale"])


def _make_mixture_distribution(case):
    return MixtureSameFamily(
        Categorical(probs=case["weights"]),
        _make_component_distribution(case),
    )


def _explicit_call_patterns(*, base, weights, loc, scale, kwargs):
    attempts = []
    try:
        component_distribution = Normal(loc=loc, scale=scale)
    except Exception:
        component_distribution = None

    if component_distribution is not None:
        attempts.extend(
            [
                dict(base=base, weights=weights, component_distribution=component_distribution, **kwargs),
                dict(base=base, weights=weights, component_dist=component_distribution, **kwargs),
            ]
        )

    attempts.extend(
        [
            dict(
                base=base,
                weights=weights,
                component_family=Normal,
                component_params={"loc": loc, "scale": scale},
                **kwargs,
            ),
            dict(
                base=base,
                weights=weights,
                component_class=Normal,
                component_params={"loc": loc, "scale": scale},
                **kwargs,
            ),
            dict(
                base=base,
                weights=weights,
                component_family=Normal,
                loc=loc,
                scale=scale,
                **kwargs,
            ),
        ]
    )
    return attempts


def _mixture_object_call_patterns(*, base, mixture_distribution, kwargs):
    return [
        dict(base=base, mixture_distribution=mixture_distribution, **kwargs),
        dict(base=base, mixture=mixture_distribution, **kwargs),
        dict(base=base, distribution=mixture_distribution, **kwargs),
    ]


def _build_mixture_transform(
    module,
    *,
    base,
    mixture_distribution=None,
    weights=None,
    loc=None,
    scale=None,
    **kwargs,
):
    attempts = []
    if mixture_distribution is not None:
        attempts.extend(
            _mixture_object_call_patterns(
                base=base,
                mixture_distribution=mixture_distribution,
                kwargs=kwargs,
            )
        )
    if weights is not None and loc is not None and scale is not None:
        attempts.extend(
            _explicit_call_patterns(
                base=base,
                weights=weights,
                loc=loc,
                scale=scale,
                kwargs=kwargs,
            )
        )

    errors = []
    for call_kwargs in attempts:
        try:
            return module.build_mixture_transform(**call_kwargs)
        except TypeError as exc:
            errors.append(f"keys={sorted(call_kwargs.keys())}: {exc}")

    pytest.fail(
        "Could not call `build_mixture_transform` with expected integration signatures.\n"
        + "\n".join(errors)
    )


def _is_transform_like(obj):
    return callable(obj) and hasattr(obj, "_inverse") and hasattr(obj, "log_abs_det_jacobian")


def _normalize_builder_result(result):
    if _is_transform_like(result):
        return result, result

    if isinstance(result, Mapping):
        for key in ("transform", "mixture_transform", "value", "built_transform"):
            value = result.get(key)
            if _is_transform_like(value):
                return value, result

    for key in ("transform", "mixture_transform", "value", "built_transform"):
        if hasattr(result, key):
            value = getattr(result, key)
            if _is_transform_like(value):
                return value, result

    if isinstance(result, tuple):
        for value in result:
            if _is_transform_like(value):
                return value, result

    pytest.fail(
        "Builder output must be either a Transform object or a container carrying one "
        "under `transform`/`mixture_transform`."
    )


def _as_mapping(obj):
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return None


def _extract_diagnostics_mapping(payload, transform):
    candidates = []
    for obj in (payload, transform):
        if isinstance(obj, Mapping):
            for key in ("diagnostics", "metadata", "meta", "info"):
                if key in obj:
                    candidates.append(obj[key])
            candidates.append(obj)
        for key in ("diagnostics", "metadata", "meta", "build_diagnostics", "info"):
            if hasattr(obj, key):
                candidates.append(getattr(obj, key))

    for candidate in candidates:
        mapping = _as_mapping(candidate)
        if mapping:
            return mapping
    return None


def _extract_config_mapping(payload, transform, diagnostics):
    candidates = [diagnostics, payload, transform]
    for obj in candidates:
        mapping = _as_mapping(obj)
        if mapping:
            for key in (
                "config",
                "build_config",
                "configs",
                "mixture_transform_build_config",
            ):
                if key in mapping:
                    nested = _as_mapping(mapping[key])
                    if nested:
                        return nested
            if any(k in mapping for k in ("solver_cfg", "knot_cfg", "interp_cfg", "tail_cfg", "transform_cfg")):
                return mapping

        for key in ("config", "build_config", "configs", "mixture_transform_build_config"):
            if hasattr(obj, key):
                nested = _as_mapping(getattr(obj, key))
                if nested:
                    return nested

    return None


def _find_key_values(obj, target_key):
    values = []

    def _walk(node):
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) == target_key:
                    values.append(value)
                _walk(value)
            return
        if dataclasses.is_dataclass(node):
            _walk(dataclasses.asdict(node))
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for value in node:
                _walk(value)

    _walk(obj)
    return values


def _value_present(values, expected):
    if isinstance(expected, (str, bool)):
        return any(value == expected for value in values)
    expected_float = float(expected)
    for value in values:
        try:
            if np.isclose(float(value), expected_float, rtol=0.0, atol=1e-12):
                return True
        except Exception:
            continue
    return False


def _assert_required_config_dataclasses(module):
    names = (
        "MixtureQuantileConfig",
        "KnotGenerationConfig",
        "InterpolatorConfig",
        "TailConfig",
        "TransformConfig",
        "MixtureTransformBuildConfig",
    )
    for name in names:
        if not hasattr(module, name):
            pytest.fail(f"Expected config dataclass `{name}` to exist in builder module.")
        cls = getattr(module, name)
        assert dataclasses.is_dataclass(cls), f"`{name}` must be a dataclass."


def _assert_invalid_explicit_input_rejected(
    module,
    *,
    base,
    weights,
    loc,
    scale,
    expected_tokens,
):
    saw_supported_signature = False
    noninformative_errors = []
    for call_kwargs in _explicit_call_patterns(
        base=base,
        weights=weights,
        loc=loc,
        scale=scale,
        kwargs={},
    ):
        try:
            module.build_mixture_transform(**call_kwargs)
        except TypeError:
            continue
        except Exception as exc:
            saw_supported_signature = True
            message = str(exc).lower()
            if any(token in message for token in expected_tokens):
                return
            noninformative_errors.append(message)
        else:
            saw_supported_signature = True
            pytest.fail(
                "Builder accepted invalid explicit-parameter input; expected a fail-fast validation error."
            )

    if not saw_supported_signature:
        pytest.fail("Could not locate a supported explicit-parameter signature for builder validation tests.")

    pytest.fail(
        "Builder raised for invalid explicit input but error text lacked useful context. "
        f"Messages: {noninformative_errors}"
    )


@pytest.mark.parametrize(
    "base, expected_class_name",
    [("uniform", "UniformToMixtureTransform"), ("normal", "NormalToMixtureTransform")],
)
def test_builder_returns_expected_transform_type(base, expected_class_name):
    module = _require_builder_module()
    UniformToMixtureTransform, NormalToMixtureTransform = _require_transform_classes()
    expected_cls = (
        UniformToMixtureTransform if expected_class_name == "UniformToMixtureTransform" else NormalToMixtureTransform
    )
    case = CASE_DATA["balanced_bimodal"]

    result_from_mixture_obj = _build_mixture_transform(
        module,
        base=base,
        mixture_distribution=_make_mixture_distribution(case),
    )
    transform, payload = _normalize_builder_result(result_from_mixture_obj)
    assert isinstance(transform, expected_cls)

    diagnostics = _extract_diagnostics_mapping(payload, transform)
    assert diagnostics is not None, (
        "Builder output must expose diagnostics/metadata with config, knot, and approximation context."
    )
    lower_keys = [str(key).lower() for key in diagnostics.keys()]
    assert any("config" in key for key in lower_keys), "Diagnostics must include chosen config references."
    assert any(("knot" in key) or ("meta" in key) for key in lower_keys), (
        "Diagnostics must include knot diagnostics/metadata."
    )
    assert any(("approx" in key) or ("error" in key) or ("summary" in key) for key in lower_keys), (
        "Diagnostics must include approximation summary metrics."
    )

    result_from_explicit = _build_mixture_transform(
        module,
        base=base,
        weights=case["weights"],
        loc=case["loc"],
        scale=case["scale"],
    )
    explicit_transform, _ = _normalize_builder_result(result_from_explicit)
    assert isinstance(explicit_transform, expected_cls)


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_builder_default_roundtrip_uniform_base(case):
    module = _require_builder_module()
    transform, _ = _normalize_builder_result(
        _build_mixture_transform(
            module,
            base="uniform",
            mixture_distribution=_make_mixture_distribution(case),
        )
    )

    u_grid = jnp.linspace(1e-7, 1.0 - 1e-7, 1024, dtype=jnp.float32)
    y = transform(u_grid)
    recovered_u = transform._inverse(y)
    ladj = transform.log_abs_det_jacobian(u_grid, y)

    assert jnp.all(jnp.isfinite(y)), "Uniform-base forward outputs must be finite."
    assert jnp.all(jnp.isfinite(recovered_u)), "Uniform-base inverse outputs must be finite."
    assert jnp.all(jnp.isfinite(ladj)), "Uniform-base Jacobian outputs must be finite."

    interior_mask = (u_grid >= 1e-3) & (u_grid <= 1.0 - 1e-3)
    max_interior_error = float(jnp.max(jnp.abs(recovered_u[interior_mask] - u_grid[interior_mask])))
    assert max_interior_error < ROUNDTRIP_INTERIOR_TOL, (
        "Uniform-base roundtrip error exceeded tolerance: "
        f"{max_interior_error} >= {ROUNDTRIP_INTERIOR_TOL}"
    )


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_builder_default_roundtrip_normal_base(case):
    module = _require_builder_module()
    transform, _ = _normalize_builder_result(
        _build_mixture_transform(
            module,
            base="normal",
            mixture_distribution=_make_mixture_distribution(case),
        )
    )

    z_grid = jnp.linspace(-8.0, 8.0, 1024, dtype=jnp.float32)
    y = transform(z_grid)
    recovered_z = transform._inverse(y)
    ladj = transform.log_abs_det_jacobian(z_grid, y)

    assert jnp.all(jnp.isfinite(y)), "Normal-base forward outputs must be finite."
    assert jnp.all(jnp.isfinite(recovered_z)), "Normal-base inverse outputs must be finite."
    assert jnp.all(jnp.isfinite(ladj)), "Normal-base Jacobian outputs must be finite."

    interior_mask = jnp.abs(z_grid) <= 4.0
    max_interior_error = float(jnp.max(jnp.abs(recovered_z[interior_mask] - z_grid[interior_mask])))
    assert max_interior_error < ROUNDTRIP_INTERIOR_TOL, (
        "Normal-base roundtrip error exceeded tolerance: "
        f"{max_interior_error} >= {ROUNDTRIP_INTERIOR_TOL}"
    )


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_distributional_sanity_against_direct_mixture_sampling(case):
    module = _require_builder_module()
    mixture_distribution = _make_mixture_distribution(case)
    transform, _ = _normalize_builder_result(
        _build_mixture_transform(
            module,
            base="normal",
            mixture_distribution=mixture_distribution,
        )
    )

    key_a, key_b = jax.random.split(jax.random.PRNGKey(2024))
    n_samples = 24_000

    z = jax.random.normal(key_a, shape=(n_samples,), dtype=jnp.float32)
    transformed_samples = transform(z)
    direct_samples = mixture_distribution.sample(key_b, sample_shape=(n_samples,))

    assert jnp.all(jnp.isfinite(transformed_samples))
    assert jnp.all(jnp.isfinite(direct_samples))

    transformed_quantiles = np.quantile(np.asarray(transformed_samples), CENTRAL_QUANTILE_LEVELS)
    direct_quantiles = np.quantile(np.asarray(direct_samples), CENTRAL_QUANTILE_LEVELS)
    max_abs_quantile_error = float(np.max(np.abs(transformed_quantiles - direct_quantiles)))

    assert max_abs_quantile_error < QUANTILE_ABS_TOL, (
        "Central quantiles from transform-based samples deviate too much from direct mixture samples: "
        f"{max_abs_quantile_error} >= {QUANTILE_ABS_TOL}"
    )


def test_custom_config_propagates_through_pipeline():
    module = _require_builder_module()
    _assert_required_config_dataclasses(module)

    cfg_a = module.MixtureTransformBuildConfig()
    cfg_b = module.MixtureTransformBuildConfig()
    assert dataclasses.asdict(cfg_a) == dataclasses.asdict(cfg_b), (
        "Default MixtureTransformBuildConfig must be deterministic and reproducible."
    )

    case = CASE_DATA["balanced_bimodal"]
    solver_cfg = {"rtol": 3e-7, "atol": 2e-7, "max_steps": 77}
    knot_cfg = {
        "num_knots": 111,
        "u_min": 1e-5,
        "u_max": 1.0 - 1e-5,
        "grid_type": "hybrid",
        "tail_density": 0.41,
    }
    interp_cfg = {"interior_method": "linear", "clip_u_eps": 2e-9, "safe_arctanh_eps": 2e-7}
    tail_cfg = {"enforce_c1_stitch": True, "min_tail_scale": 1e-7}
    transform_cfg = {"clip_u_eps": 3e-9, "validate_args": False}

    result = _build_mixture_transform(
        module,
        base="uniform",
        mixture_distribution=_make_mixture_distribution(case),
        solver_cfg=solver_cfg,
        knot_cfg=knot_cfg,
        interp_cfg=interp_cfg,
        tail_cfg=tail_cfg,
        transform_cfg=transform_cfg,
    )
    transform, payload = _normalize_builder_result(result)
    diagnostics = _extract_diagnostics_mapping(payload, transform)
    config_mapping = _extract_config_mapping(payload, transform, diagnostics)

    assert config_mapping is not None, (
        "Builder output must expose chosen nested config values through returned config/diagnostics."
    )
    expected_config_values = {
        "max_steps": solver_cfg["max_steps"],
        "num_knots": knot_cfg["num_knots"],
        "tail_density": knot_cfg["tail_density"],
        "interior_method": interp_cfg["interior_method"],
        "min_tail_scale": tail_cfg["min_tail_scale"],
        "validate_args": transform_cfg["validate_args"],
    }

    for key, expected in expected_config_values.items():
        values = _find_key_values(config_mapping, key)
        assert values, f"Config snapshot is missing key `{key}`."
        assert _value_present(values, expected), f"Config key `{key}` does not include expected value `{expected}`."


def test_invalid_inputs_fail_fast_with_context(monkeypatch):
    module = _require_builder_module()
    case = CASE_DATA["balanced_bimodal"]

    _assert_invalid_explicit_input_rejected(
        module,
        base="uniform",
        weights=jnp.array([0.9, 0.3], dtype=jnp.float32),
        loc=case["loc"],
        scale=case["scale"],
        expected_tokens=("weight", "sum", "normalize", "invalid"),
    )
    _assert_invalid_explicit_input_rejected(
        module,
        base="uniform",
        weights=case["weights"],
        loc=jnp.array([-1.0, 1.0, 3.0], dtype=jnp.float32),
        scale=jnp.array([1.0, 0.7], dtype=jnp.float32),
        expected_tokens=("shape", "broadcast", "component", "mismatch", "invalid"),
    )

    injected = {"patched": False}

    def _raise_bracket_failure(*_args, **_kwargs):
        raise ValueError("quantile stage bracket failure during integration build")

    for name in ("build_mixture_quantile_backend", "_build_mixture_quantile_backend"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _raise_bracket_failure)
            injected["patched"] = True
            break

    build_kwargs = dict(
        base="uniform",
        mixture_distribution=_make_mixture_distribution(case),
    )
    if not injected["patched"]:
        build_kwargs["knot_cfg"] = {"num_knots": 1}

    with pytest.raises(Exception) as exc_info:
        _build_mixture_transform(module, **build_kwargs)

    message = str(exc_info.value).lower()
    assert any(token in message for token in ("quantile", "bracket", "solver", "knot", "interpol", "pipeline")), (
        "Failure-path exceptions must include meaningful stage/context in error text."
    )
    if injected["patched"]:
        assert "bracket" in message, "Patched bracket failure context should survive exception propagation."

    diagnostics = getattr(exc_info.value, "diagnostics", None) or getattr(exc_info.value, "meta", None)
    if diagnostics is not None:
        diagnostics_mapping = _as_mapping(diagnostics)
        assert diagnostics_mapping is not None
        keys = [str(key).lower() for key in diagnostics_mapping.keys()]
        assert any(key in keys for key in ("reason", "stage", "context", "error", "message"))


def test_builder_jit_and_batch_compatibility():
    module = _require_builder_module()
    case = CASE_DATA["dominant_component"]
    mixture_distribution = _make_mixture_distribution(case)

    uniform_transform, _ = _normalize_builder_result(
        _build_mixture_transform(
            module,
            base="uniform",
            mixture_distribution=mixture_distribution,
        )
    )
    normal_transform, _ = _normalize_builder_result(
        _build_mixture_transform(
            module,
            base="normal",
            mixture_distribution=mixture_distribution,
        )
    )

    u_batch = jnp.stack(
        [
            jnp.linspace(1e-4, 1.0 - 1e-4, 128, dtype=jnp.float32),
            jnp.linspace(2e-4, 1.0 - 2e-4, 128, dtype=jnp.float32),
        ],
        axis=0,
    )
    z_batch = jnp.stack(
        [
            jnp.linspace(-6.0, 6.0, 128, dtype=jnp.float32),
            jnp.linspace(-5.0, 5.0, 128, dtype=jnp.float32),
        ],
        axis=0,
    )

    uniform_y = uniform_transform(u_batch)
    normal_y = normal_transform(z_batch)
    assert uniform_y.shape == u_batch.shape
    assert normal_y.shape == z_batch.shape
    assert uniform_transform._inverse(uniform_y).shape == u_batch.shape
    assert normal_transform._inverse(normal_y).shape == z_batch.shape

    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: uniform_transform(x))(u_batch)),
        np.asarray(uniform_y),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda x: normal_transform(x))(z_batch)),
        np.asarray(normal_y),
        rtol=2e-6,
        atol=1e-5,
    )

    assert jnp.all(jnp.isfinite(uniform_transform.log_abs_det_jacobian(u_batch, uniform_y)))
    assert jnp.all(jnp.isfinite(normal_transform.log_abs_det_jacobian(z_batch, normal_y)))
