"""Optional non-gating performance smoke checks for mixture transform builder."""

from __future__ import annotations

import importlib
import os
import time
from types import ModuleType

import jax
import numpyro.distributions as dist
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


MODULE_NAME = "numpyro_extras.mixture_transform_builder"
RUN_PERF = os.environ.get("NUMPYRO_EXTRAS_RUN_PERF", "0") == "1"


def _import_builder_module() -> ModuleType:
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.mixture_transform_builder` "
                "(expected at `src/numpyro_extras/mixture_transform_builder.py`)."
            )
        raise


def _build_transform_normal_base():
    module = _import_builder_module()
    builder = getattr(module, "build_mixture_transform", None)
    if builder is None or not callable(builder):
        pytest.fail(
            "Missing callable `build_mixture_transform` in "
            "`numpyro_extras.mixture_transform_builder`."
        )

    mixture = dist.MixtureSameFamily(
        dist.Categorical(probs=jnp.asarray([0.55, 0.45], dtype=jnp.float64)),
        dist.Normal(
            loc=jnp.asarray([-1.5, 2.0], dtype=jnp.float64),
            scale=jnp.asarray([0.9, 1.2], dtype=jnp.float64),
        ),
    )
    attempts = (
        {"base": "normal", "mixture_distribution": mixture},
        {"base": "normal", "mixture": mixture},
    )
    errors = []
    for kwargs in attempts:
        try:
            result = builder(**kwargs)
            if isinstance(result, dict):
                transform = result.get("transform")
            else:
                transform = getattr(result, "transform", None) or result
            if transform is None:
                pytest.fail(
                    "Builder result must expose a transform via `transform` key/attr, "
                    "or return the transform directly."
                )
            return transform
        except TypeError as exc:
            message = str(exc).lower()
            if not any(
                token in message
                for token in (
                    "unexpected keyword",
                    "positional argument",
                    "required positional",
                    "got multiple values",
                    "takes ",
                    "missing ",
                )
            ):
                raise
            errors.append(str(exc))
            continue

    pytest.fail(
        "Unable to call `build_mixture_transform` for performance smoke path. "
        f"Recent TypeErrors={errors[-2:]}"
    )


@pytest.mark.skipif(
    not RUN_PERF,
    reason="Set NUMPYRO_EXTRAS_RUN_PERF=1 to run non-gating performance smoke tests.",
)
def test_builder_performance_smoke():
    build_start = time.perf_counter()
    transform = _build_transform_normal_base()
    build_seconds = time.perf_counter() - build_start

    batch = jax.random.normal(
        jax.random.PRNGKey(2026), shape=(200_000,), dtype=jnp.float64
    )
    forward_jit = jax.jit(transform.__call__)

    # Warmup compile outside throughput timing.
    _ = forward_jit(batch[:2048]).block_until_ready()

    forward_start = time.perf_counter()
    out = forward_jit(batch).block_until_ready()
    forward_seconds = time.perf_counter() - forward_start

    assert jnp.all(jnp.isfinite(out)), "Forward performance smoke produced NaN/Inf."
    assert build_seconds < 30.0, (
        "Builder performance smoke exceeded coarse upper bound: "
        f"build_seconds={build_seconds:.3f}s"
    )
    assert forward_seconds < 15.0, (
        "Forward throughput smoke exceeded coarse upper bound: "
        f"forward_seconds={forward_seconds:.3f}s for batch_size={batch.size}."
    )
