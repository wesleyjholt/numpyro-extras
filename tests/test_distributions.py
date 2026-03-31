from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from numpyro_extras import ShiftedScaledBeta


def test_shifted_scaled_beta_matches_base_distribution_for_log_prob_mean_and_variance():
    concentration1 = jnp.asarray(2.5)
    concentration0 = jnp.asarray(3.5)
    loc = jnp.asarray(-1.2)
    scale = jnp.asarray(4.0)

    shifted = ShiftedScaledBeta(concentration1, concentration0, loc=loc, scale=scale)
    base = dist.Beta(concentration1, concentration0)
    y = jnp.asarray(0.7)
    x = (y - loc) / scale

    assert float(shifted.log_prob(y)) == pytest.approx(
        float(base.log_prob(x) - jnp.log(scale)),
        rel=1e-6,
        abs=1e-6,
    )
    assert float(shifted.mean) == pytest.approx(float(loc + scale * base.mean))
    assert float(shifted.variance) == pytest.approx(float((scale**2) * base.variance))


def test_shifted_scaled_beta_cdf_and_icdf_roundtrip():
    distribution = ShiftedScaledBeta(2.0, 5.0, loc=-3.0, scale=7.5)
    q = jnp.linspace(1e-6, 1.0 - 1e-6, 128)
    x = distribution.icdf(q)
    q_roundtrip = distribution.cdf(x)
    assert jnp.max(jnp.abs(q_roundtrip - q)) < 1e-5


def test_shifted_scaled_beta_icdf_respects_endpoints():
    distribution = ShiftedScaledBeta(1.5, 4.0, loc=2.0, scale=3.0)
    q = jnp.asarray([0.0, 1.0])
    x = distribution.icdf(q)
    assert jnp.allclose(x, jnp.asarray([2.0, 5.0]))


def test_shifted_scaled_beta_cdf_clips_outside_support():
    distribution = ShiftedScaledBeta(3.0, 2.0, loc=-2.0, scale=5.0)
    x = jnp.asarray([-10.0, -2.0, 3.0, 10.0])
    cdf = distribution.cdf(x)
    assert jnp.allclose(cdf, jnp.asarray([0.0, 0.0, 1.0, 1.0]))
