"""NumPyro transforms that map base variables into mixture quantiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from numpyro.distributions.transforms import Transform


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _coerce_transform_cfg(transform_cfg: Any) -> tuple[float, bool]:
    clip_u_eps = float(_cfg_get(transform_cfg, "clip_u_eps", 1e-10))
    validate_args = bool(_cfg_get(transform_cfg, "validate_args", False))
    if not jnp.isfinite(clip_u_eps) or not (0.0 < clip_u_eps < 0.5):
        raise ValueError("`transform_cfg.clip_u_eps` must be finite and satisfy 0 < eps < 0.5.")
    return clip_u_eps, validate_args


class UniformToMixtureTransform(Transform):
    """Map `u in (0,1)` into mixture space via an interpolated inverse CDF."""

    domain = constraints.unit_interval
    codomain = constraints.real
    sign = 1

    def __init__(self, interpolator: Any, transform_cfg: Any = None) -> None:
        self.interpolator = interpolator
        self.clip_u_eps, self.validate_args = _coerce_transform_cfg(transform_cfg)

    def _clip_u(self, u: Any) -> jnp.ndarray:
        u_arr = jnp.asarray(u)
        dtype = jnp.result_type(u_arr, jnp.asarray(self.clip_u_eps))
        u_arr = jnp.asarray(u_arr, dtype=dtype)
        eps = jnp.asarray(self.clip_u_eps, dtype=dtype)
        return jnp.clip(u_arr, eps, 1.0 - eps)

    def _log_abs_dxdu(self, u: Any) -> jnp.ndarray:
        if hasattr(self.interpolator, "log_abs_dxdu"):
            return jnp.asarray(self.interpolator.log_abs_dxdu(u))
        dxdu = jnp.asarray(self.interpolator.dxdu(u))
        tiny = jnp.asarray(jnp.finfo(dxdu.dtype).tiny, dtype=dxdu.dtype)
        safe = jnp.maximum(jnp.abs(jnp.where(jnp.isfinite(dxdu), dxdu, tiny)), tiny)
        return jnp.log(safe)

    def __call__(self, x: Any) -> jnp.ndarray:
        u = self._clip_u(x)
        return jnp.asarray(self.interpolator.icdf(u))

    def _inverse(self, y: Any) -> jnp.ndarray:
        return jnp.asarray(self.interpolator.cdf(y))

    def log_abs_det_jacobian(
        self, x: Any, y: Any, intermediates: Any = None
    ) -> jnp.ndarray:
        del y, intermediates
        u = self._clip_u(x)
        return self._log_abs_dxdu(u)

    def tree_flatten(self):
        return (self.interpolator, self.clip_u_eps), (
            ("interpolator", "clip_u_eps"),
            {"validate_args": self.validate_args},
        )


class NormalToMixtureTransform(Transform):
    """Map `z in R` into mixture space via `u=Phi(z)` then interpolated `icdf(u)`."""

    domain = constraints.real
    codomain = constraints.real
    sign = 1

    def __init__(
        self,
        interpolator: Any,
        transform_cfg: Any = None,
        standard_normal: dist.Distribution | None = None,
    ) -> None:
        self.interpolator = interpolator
        self.clip_u_eps, self.validate_args = _coerce_transform_cfg(transform_cfg)
        if standard_normal is None:
            standard_normal = dist.Normal(jnp.asarray(0.0), jnp.asarray(1.0))
        self.standard_normal = standard_normal

    def _clip_u(self, u: Any) -> jnp.ndarray:
        u_arr = jnp.asarray(u)
        dtype = jnp.result_type(u_arr, jnp.asarray(self.clip_u_eps))
        u_arr = jnp.asarray(u_arr, dtype=dtype)
        eps = jnp.asarray(self.clip_u_eps, dtype=dtype)
        return jnp.clip(u_arr, eps, 1.0 - eps)

    def _log_abs_dxdu(self, u: Any) -> jnp.ndarray:
        if hasattr(self.interpolator, "log_abs_dxdu"):
            return jnp.asarray(self.interpolator.log_abs_dxdu(u))
        dxdu = jnp.asarray(self.interpolator.dxdu(u))
        tiny = jnp.asarray(jnp.finfo(dxdu.dtype).tiny, dtype=dxdu.dtype)
        safe = jnp.maximum(jnp.abs(jnp.where(jnp.isfinite(dxdu), dxdu, tiny)), tiny)
        return jnp.log(safe)

    def __call__(self, x: Any) -> jnp.ndarray:
        z = jnp.asarray(x)
        u = self._clip_u(self.standard_normal.cdf(z))
        return jnp.asarray(self.interpolator.icdf(u))

    def _inverse(self, y: Any) -> jnp.ndarray:
        u = jnp.asarray(self.interpolator.cdf(y))
        return jnp.asarray(self.standard_normal.icdf(u))

    def log_abs_det_jacobian(
        self, x: Any, y: Any, intermediates: Any = None
    ) -> jnp.ndarray:
        del y, intermediates
        z = jnp.asarray(x)
        u = self._clip_u(self.standard_normal.cdf(z))
        return self._log_abs_dxdu(u) + self.standard_normal.log_prob(z)

    def tree_flatten(self):
        return (self.interpolator, self.standard_normal, self.clip_u_eps), (
            ("interpolator", "standard_normal", "clip_u_eps"),
            {"validate_args": self.validate_args},
        )


__all__ = ["UniformToMixtureTransform", "NormalToMixtureTransform"]
