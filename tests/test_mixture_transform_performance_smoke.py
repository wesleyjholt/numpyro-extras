"""Non-gating performance smoke checks for Task 10 integration builder."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest
from numpyro.distributions import Categorical, MixtureSameFamily, Normal


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


def _is_transform_like(obj):
    return callable(obj) and hasattr(obj, "_inverse") and hasattr(obj, "log_abs_det_jacobian")


def _normalize_builder_result(result):
    if _is_transform_like(result):
        return result
    if isinstance(result, Mapping):
        for key in ("transform", "mixture_transform", "value", "built_transform"):
            candidate = result.get(key)
            if _is_transform_like(candidate):
                return candidate
    for key in ("transform", "mixture_transform", "value", "built_transform"):
        if hasattr(result, key):
            candidate = getattr(result, key)
            if _is_transform_like(candidate):
                return candidate
    if isinstance(result, tuple):
        for candidate in result:
            if _is_transform_like(candidate):
                return candidate
    pytest.fail("Builder output must include a Transform object.")


def _build_with_distribution(module, *, base, mixture_distribution):
    attempts = (
        {"base": base, "mixture_distribution": mixture_distribution},
        {"base": base, "mixture": mixture_distribution},
        {"base": base, "distribution": mixture_distribution},
    )
    errors = []
    for kwargs in attempts:
        try:
            return module.build_mixture_transform(**kwargs)
        except TypeError as exc:
            errors.append(f"keys={sorted(kwargs.keys())}: {exc}")
    pytest.fail(
        "Could not call `build_mixture_transform` with a mixture-distribution input mode.\n"
        + "\n".join(errors)
    )


def _measure_forward_throughput(transform, x):
    y = transform(x)
    jax.block_until_ready(y)

    start = time.perf_counter()
    y = transform(x)
    jax.block_until_ready(y)
    elapsed = time.perf_counter() - start
    return x.size / max(elapsed, 1e-9)


def test_builder_performance_smoke():
    module = _require_builder_module()
    mixture_distribution = MixtureSameFamily(
        Categorical(probs=jnp.array([0.55, 0.45], dtype=jnp.float32)),
        Normal(
            loc=jnp.array([-1.8, 2.4], dtype=jnp.float32),
            scale=jnp.array([0.9, 1.1], dtype=jnp.float32),
        ),
    )

    start_uniform = time.perf_counter()
    uniform_transform = _normalize_builder_result(
        _build_with_distribution(module, base="uniform", mixture_distribution=mixture_distribution)
    )
    uniform_build_seconds = time.perf_counter() - start_uniform

    start_normal = time.perf_counter()
    normal_transform = _normalize_builder_result(
        _build_with_distribution(module, base="normal", mixture_distribution=mixture_distribution)
    )
    normal_build_seconds = time.perf_counter() - start_normal

    uniform_x = jnp.linspace(1e-6, 1.0 - 1e-6, 16_384, dtype=jnp.float32)
    normal_x = jnp.linspace(-6.0, 6.0, 16_384, dtype=jnp.float32)

    uniform_throughput = _measure_forward_throughput(uniform_transform, uniform_x)
    normal_throughput = _measure_forward_throughput(normal_transform, normal_x)

    # Coarse bounds only: intended to catch pathological regressions, not micro-performance.
    assert uniform_build_seconds < 45.0
    assert normal_build_seconds < 45.0
    assert uniform_throughput > 100.0
    assert normal_throughput > 100.0
