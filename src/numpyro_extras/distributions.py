"""Additional NumPyro distributions."""

from __future__ import annotations

from typing import Any
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betainc
from numpyro.distributions import Beta
from numpyro.distributions import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.distribution import ArrayLike
from numpyro.distributions.util import promote_shapes
from numpyro.distributions.util import validate_sample


class ShiftedScaledBeta(Distribution):
    """Beta distribution shifted by ``loc`` and scaled by ``scale``.

    If ``X ~ Beta(concentration1, concentration0)``, then
    ``Y = loc + scale * X``.

    Support is exactly ``[loc, loc + scale]`` for positive ``scale``.
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    reparametrized_params = ["concentration1", "concentration0", "loc", "scale"]
    pytree_data_fields = ("concentration1", "concentration0", "loc", "scale", "_support")

    def __init__(
        self,
        concentration1: ArrayLike,
        concentration0: ArrayLike,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.concentration1, self.concentration0, self.loc, self.scale = promote_shapes(
            concentration1, concentration0, loc, scale
        )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(concentration1),
            jnp.shape(concentration0),
            jnp.shape(loc),
            jnp.shape(scale),
        )
        self.concentration1 = jnp.broadcast_to(self.concentration1, batch_shape)
        self.concentration0 = jnp.broadcast_to(self.concentration0, batch_shape)
        self.loc = jnp.broadcast_to(self.loc, batch_shape)
        self.scale = jnp.broadcast_to(self.scale, batch_shape)
        self._support = constraints.interval(self.loc, self.loc + self.scale)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return self._support

    @property
    def _base_dist(self) -> Beta:
        return Beta(self.concentration1, self.concentration0)

    def sample(
        self,
        key: jax.dtypes.prng_key,
        sample_shape: tuple[int, ...] = (),
    ) -> ArrayLike:
        return self.loc + self.scale * self._base_dist.sample(key, sample_shape)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        x = (value - self.loc) / self.scale
        return self._base_dist.log_prob(x) - jnp.log(self.scale)

    def cdf(self, value: ArrayLike) -> ArrayLike:
        x = (value - self.loc) / self.scale
        return betainc(self.concentration1, self.concentration0, jnp.clip(x, 0.0, 1.0))

    def icdf(self, q: ArrayLike) -> ArrayLike:
        q = jnp.asarray(q)
        q_clipped = jnp.clip(q, 0.0, 1.0)
        eps = jnp.finfo(q_clipped.dtype).eps
        q_inner = jnp.clip(q_clipped, eps, 1.0 - eps)
        lo = jnp.zeros_like(q_inner)
        hi = jnp.ones_like(q_inner)

        def body_fn(_: Any, bounds: Any) -> Any:
            lo_, hi_ = bounds
            mid = 0.5 * (lo_ + hi_)
            cdf_mid = betainc(self.concentration1, self.concentration0, mid)
            move_left = cdf_mid >= q_inner
            hi_new = jnp.where(move_left, mid, hi_)
            lo_new = jnp.where(move_left, lo_, mid)
            return lo_new, hi_new

        lo, hi = lax.fori_loop(0, 60, body_fn, (lo, hi))
        x = 0.5 * (lo + hi)
        x = jnp.where(q_clipped <= 0.0, 0.0, x)
        x = jnp.where(q_clipped >= 1.0, 1.0, x)
        return self.loc + self.scale * x

    @property
    def mean(self) -> ArrayLike:
        return self.loc + self.scale * self._base_dist.mean

    @property
    def variance(self) -> ArrayLike:
        return self.scale**2 * self._base_dist.variance

    def entropy(self) -> ArrayLike:
        return self._base_dist.entropy() + jnp.log(self.scale)

    @staticmethod
    def infer_shapes(
        concentration1: tuple[int, ...] = (),
        concentration0: tuple[int, ...] = (),
        loc: tuple[int, ...] = (),
        scale: tuple[int, ...] = (),
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        batch_shape = lax.broadcast_shapes(concentration1, concentration0, loc, scale)
        event_shape: tuple[int, ...] = ()
        return batch_shape, event_shape


__all__ = ["ShiftedScaledBeta"]
