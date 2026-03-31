from __future__ import annotations

import jax
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from numpyro_extras.distribution_transform_builder import (
    DistributionTransformBuildConfig,
    build_distribution_transform,
)
from numpyro_extras.transforms import NormalToDistributionTransform


def test_build_distribution_transform_with_direct_distribution_input():
    target = dist.Normal(loc=jnp.asarray(0.5), scale=jnp.asarray(1.4))
    result = build_distribution_transform(
        base="normal",
        distribution=target,
        build_cfg=DistributionTransformBuildConfig(),
    )
    transform = result.transform

    assert isinstance(transform, NormalToDistributionTransform)

    z = jnp.linspace(-5.0, 5.0, 513, dtype=jnp.float64)
    y = transform(z)
    z_back = transform._inverse(y)

    assert jnp.all(jnp.isfinite(y))
    assert jnp.max(jnp.abs(z_back - z)) < 2e-4


def test_build_distribution_transform_with_uniform_base():
    target = dist.Logistic(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0))
    result = build_distribution_transform(base="uniform", distribution=target)
    u = jnp.linspace(1e-8, 1.0 - 1e-8, 257, dtype=jnp.float64)
    x = result.transform(u)
    assert jnp.all(jnp.isfinite(x))
