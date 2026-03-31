from __future__ import annotations

import jax
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from numpyro_extras.distribution_quantile import build_distribution_quantile_backend


def test_build_distribution_quantile_backend_matches_base_distribution():
    target = dist.Normal(loc=jnp.asarray(0.75), scale=jnp.asarray(1.3))
    backend = build_distribution_quantile_backend(distribution=target)

    u = jnp.linspace(1e-6, 1.0 - 1e-6, 513, dtype=jnp.float64)
    x = backend.icdf(u)
    u_roundtrip = backend.cdf(x)

    assert jnp.max(jnp.abs(x - target.icdf(u))) < 1e-4
    assert jnp.max(jnp.abs(u_roundtrip - u)) < 1e-5


def test_build_distribution_quantile_backend_with_non_gaussian_distribution():
    target = dist.Logistic(loc=jnp.asarray(0.0), scale=jnp.asarray(1.0))
    backend = build_distribution_quantile_backend(distribution=target)
    assert float(backend.cdf(jnp.asarray(0.0))) == 0.5
