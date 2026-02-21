"""Monotone quantile/CDF interpolation with sigmoid tails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .mixture_knots import QuantileKnotSet

__all__ = [
    "InterpolatorConfig",
    "TailConfig",
    "QuantileInterpolator1D",
]


@dataclass(frozen=True)
class InterpolatorConfig:
    interior_method: str = "akima"
    clip_u_eps: float = 1e-10
    safe_arctanh_eps: float = 1e-7


@dataclass(frozen=True)
class TailConfig:
    enforce_c1_stitch: bool = True
    min_tail_scale: float = 1e-8


def _coerce_interp_cfg(cfg: Optional[Mapping[str, Any] | InterpolatorConfig]) -> InterpolatorConfig:
    if cfg is None:
        out = InterpolatorConfig()
    elif isinstance(cfg, InterpolatorConfig):
        out = cfg
    elif isinstance(cfg, Mapping):
        out = InterpolatorConfig(
            interior_method=str(cfg.get("interior_method", "akima")),
            clip_u_eps=float(cfg.get("clip_u_eps", 1e-10)),
            safe_arctanh_eps=float(cfg.get("safe_arctanh_eps", 1e-7)),
        )
    else:
        raise TypeError("`interp_cfg` must be None, dict-like, or `InterpolatorConfig`.")

    if out.interior_method not in {"akima", "linear", "pchip_like"}:
        raise ValueError("`interior_method` must be one of {'akima', 'linear', 'pchip_like'}.")
    if not (0.0 < out.clip_u_eps < 0.5):
        raise ValueError("`clip_u_eps` must satisfy 0 < clip_u_eps < 0.5.")
    if not (0.0 < out.safe_arctanh_eps < 0.5):
        raise ValueError("`safe_arctanh_eps` must satisfy 0 < safe_arctanh_eps < 0.5.")
    return out


def _coerce_tail_cfg(cfg: Optional[Mapping[str, Any] | TailConfig]) -> TailConfig:
    if cfg is None:
        out = TailConfig()
    elif isinstance(cfg, TailConfig):
        out = cfg
    elif isinstance(cfg, Mapping):
        out = TailConfig(
            enforce_c1_stitch=bool(cfg.get("enforce_c1_stitch", True)),
            min_tail_scale=float(cfg.get("min_tail_scale", 1e-8)),
        )
    else:
        raise TypeError("`tail_cfg` must be None, dict-like, or `TailConfig`.")

    if out.min_tail_scale <= 0.0:
        raise ValueError("`min_tail_scale` must be > 0.")
    return out


def _linear_interp_with_slope(
    xq: jax.Array, xk: jax.Array, yk: jax.Array, dy_dx_segments: jax.Array
) -> tuple[jax.Array, jax.Array]:
    idx = jnp.searchsorted(xk, xq, side="right") - 1
    idx = jnp.clip(idx, 0, xk.shape[0] - 2)

    x0 = jnp.take(xk, idx)
    y0 = jnp.take(yk, idx)
    slope = jnp.take(dy_dx_segments, idx)
    yq = y0 + slope * (xq - x0)
    return yq, slope


class QuantileInterpolator1D:
    """Piecewise monotone interpolator for `x(u)` and `u(x)` with smooth tails."""

    def __init__(
        self,
        knot_set: QuantileKnotSet,
        *,
        interp_cfg: Optional[Mapping[str, Any] | InterpolatorConfig] = None,
        tail_cfg: Optional[Mapping[str, Any] | TailConfig] = None,
    ) -> None:
        self._interp_cfg = _coerce_interp_cfg(interp_cfg)
        self._tail_cfg = _coerce_tail_cfg(tail_cfg)

        u_knots = jnp.asarray(knot_set.u_knots)
        x_knots = jnp.asarray(knot_set.x_knots)

        if u_knots.ndim != 1 or x_knots.ndim != 1:
            raise ValueError("`u_knots` and `x_knots` must be rank-1 arrays.")
        if u_knots.shape != x_knots.shape:
            raise ValueError("`u_knots` and `x_knots` must have the same shape.")
        if u_knots.size < 2:
            raise ValueError("Need at least two knots.")

        dtype = jnp.result_type(u_knots.dtype, x_knots.dtype, jnp.float32)
        u_knots = jnp.asarray(u_knots, dtype=dtype)
        x_knots = jnp.asarray(x_knots, dtype=dtype)

        finite_mask = np.asarray(jnp.isfinite(u_knots) & jnp.isfinite(x_knots))
        u_np = np.asarray(u_knots)[finite_mask]
        x_np = np.asarray(x_knots)[finite_mask]
        if u_np.size < 2:
            raise ValueError("`knot_set` must contain at least two finite knot pairs.")
        u_knots = jnp.asarray(u_np, dtype=dtype)
        x_knots = jnp.asarray(x_np, dtype=dtype)

        if not bool(jnp.all((u_knots > 0.0) & (u_knots < 1.0))):
            raise ValueError("`u_knots` must be strictly inside (0, 1).")
        if not bool(jnp.all(jnp.diff(u_knots) > 0.0)):
            raise ValueError("`u_knots` must be strictly increasing.")
        if not bool(jnp.all(jnp.diff(x_knots) > 0.0)):
            raise ValueError("`x_knots` must be strictly increasing.")

        seg_left = (u_knots[1] - u_knots[0]) / (x_knots[1] - x_knots[0])
        seg_right = (u_knots[-1] - u_knots[-2]) / (x_knots[-1] - x_knots[-2])
        if self._tail_cfg.enforce_c1_stitch:
            du_dx_left = seg_left
            du_dx_right = seg_right
        else:
            du_dx_left = jnp.asarray(knot_set.du_dx_left, dtype=dtype)
            du_dx_right = jnp.asarray(knot_set.du_dx_right, dtype=dtype)
            du_dx_left = jnp.where(
                jnp.isfinite(du_dx_left) & (du_dx_left > 0.0),
                du_dx_left,
                seg_left,
            )
            du_dx_right = jnp.where(
                jnp.isfinite(du_dx_right) & (du_dx_right > 0.0),
                du_dx_right,
                seg_right,
            )

        self._u_knots = u_knots
        self._x_knots = x_knots
        self._du_dx_segments = jnp.diff(u_knots) / jnp.diff(x_knots)
        self._dx_du_segments = jnp.diff(x_knots) / jnp.diff(u_knots)

        self._u0 = u_knots[0]
        self._uN = u_knots[-1]
        self._x0 = x_knots[0]
        self._xN = x_knots[-1]
        self._du_dx_left = du_dx_left
        self._du_dx_right = du_dx_right

        self._clip_u_eps = jnp.asarray(self._interp_cfg.clip_u_eps, dtype=dtype)
        self._safe_arctanh_eps = jnp.asarray(self._interp_cfg.safe_arctanh_eps, dtype=dtype)
        self._tiny = jnp.asarray(self._tail_cfg.min_tail_scale, dtype=dtype)
        self._boundary_u_tol = jnp.asarray(2e-6, dtype=dtype)
        self._positive_floor = jnp.asarray(1.0000001e-12, dtype=dtype)

    def _clip_u(self, u: jax.Array) -> jax.Array:
        dtype = u.dtype
        min_eps = jnp.asarray(np.finfo(np.dtype(dtype)).eps, dtype=dtype)
        eps = jnp.maximum(self._clip_u_eps.astype(dtype), min_eps)
        return jnp.clip(u, eps, 1.0 - eps)

    def _safe_arctanh(self, value: jax.Array) -> jax.Array:
        lo = -1.0 + self._safe_arctanh_eps
        hi = 1.0 - self._safe_arctanh_eps
        return jnp.arctanh(jnp.clip(value, lo, hi))

    def _tail_left_cdf(self, x: jax.Array) -> jax.Array:
        m0 = self._du_dx_left
        scale = m0 / self._u0
        return self._u0 + self._u0 * jnp.tanh(scale * (x - self._x0))

    def _tail_right_cdf(self, x: jax.Array) -> jax.Array:
        mN = self._du_dx_right
        one_minus_uN = 1.0 - self._uN
        scale = mN / one_minus_uN
        return self._uN + one_minus_uN * jnp.tanh(scale * (x - self._xN))

    def _tail_left_icdf(self, u: jax.Array) -> jax.Array:
        m0 = self._du_dx_left
        ratio = (u - self._u0) / self._u0
        return self._x0 + (self._u0 / m0) * self._safe_arctanh(ratio)

    def _tail_right_icdf(self, u: jax.Array) -> jax.Array:
        mN = self._du_dx_right
        one_minus_uN = 1.0 - self._uN
        ratio = (u - self._uN) / one_minus_uN
        return self._xN + (one_minus_uN / mN) * self._safe_arctanh(ratio)

    def _tail_left_dudx(self, x: jax.Array) -> jax.Array:
        m0 = self._du_dx_left
        scale = m0 / self._u0
        z = scale * (x - self._x0)
        sech2 = jnp.square(1.0 / jnp.cosh(z))
        return m0 * sech2

    def _tail_right_dudx(self, x: jax.Array) -> jax.Array:
        mN = self._du_dx_right
        one_minus_uN = 1.0 - self._uN
        scale = mN / one_minus_uN
        z = scale * (x - self._xN)
        sech2 = jnp.square(1.0 / jnp.cosh(z))
        return mN * sech2

    def _tail_left_dxdu(self, u: jax.Array) -> jax.Array:
        m0 = self._du_dx_left
        u_eff = jnp.minimum(u, self._u0)
        den = jnp.maximum(u_eff * (2.0 * self._u0 - u_eff), self._tiny)
        return (self._u0 * self._u0) / (m0 * den)

    def _tail_right_dxdu(self, u: jax.Array) -> jax.Array:
        mN = self._du_dx_right
        one_minus_uN = 1.0 - self._uN
        u_eff = jnp.maximum(u, self._uN)
        w = jnp.maximum(1.0 - u_eff, self._tiny)
        den = jnp.maximum(w * (2.0 * one_minus_uN - w), self._tiny)
        return (one_minus_uN * one_minus_uN) / (mN * den)

    def icdf(self, u: jax.Array) -> jax.Array:
        u_arr = jnp.asarray(u, dtype=self._u_knots.dtype)
        u_clipped = self._clip_u(u_arr)

        u_mid = jnp.clip(u_clipped, self._u0, self._uN)
        x_mid, _ = _linear_interp_with_slope(u_mid, self._u_knots, self._x_knots, self._dx_du_segments)

        x_left = self._tail_left_icdf(u_clipped)
        x_right = self._tail_right_icdf(u_clipped)

        x = jnp.where(u_clipped < self._u0, x_left, x_mid)
        x = jnp.where(u_clipped > self._uN, x_right, x)
        x = jnp.where(jnp.abs(u_clipped - self._u0) <= self._boundary_u_tol, self._x0, x)
        x = jnp.where(jnp.abs(u_clipped - self._uN) <= self._boundary_u_tol, self._xN, x)
        return x

    def cdf(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=self._x_knots.dtype)
        x_mid = jnp.clip(x_arr, self._x0, self._xN)
        u_mid, _ = _linear_interp_with_slope(x_mid, self._x_knots, self._u_knots, self._du_dx_segments)

        u_left = self._tail_left_cdf(x_arr)
        u_right = self._tail_right_cdf(x_arr)

        u = jnp.where(x_arr < self._x0, u_left, u_mid)
        u = jnp.where(x_arr > self._xN, u_right, u)
        return self._clip_u(u)

    def dudx(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=self._x_knots.dtype)
        x_mid = jnp.clip(x_arr, self._x0, self._xN)
        _, dudx_mid = _linear_interp_with_slope(x_mid, self._x_knots, self._u_knots, self._du_dx_segments)

        dudx_left = self._tail_left_dudx(x_arr)
        dudx_right = self._tail_right_dudx(x_arr)

        dudx = jnp.where(x_arr < self._x0, dudx_left, dudx_mid)
        dudx = jnp.where(x_arr > self._xN, dudx_right, dudx)
        return jnp.maximum(dudx, self._positive_floor)

    def dxdu(self, u: jax.Array) -> jax.Array:
        u_arr = jnp.asarray(u, dtype=self._u_knots.dtype)
        u_clipped = self._clip_u(u_arr)
        u_mid = jnp.clip(u_clipped, self._u0, self._uN)
        _, dxdu_mid = _linear_interp_with_slope(u_mid, self._u_knots, self._x_knots, self._dx_du_segments)

        dxdu_left = self._tail_left_dxdu(u_clipped)
        dxdu_right = self._tail_right_dxdu(u_clipped)

        dxdu = jnp.where(u_clipped < self._u0, dxdu_left, dxdu_mid)
        dxdu = jnp.where(u_clipped > self._uN, dxdu_right, dxdu)
        return jnp.maximum(dxdu, self._positive_floor)

    def log_abs_dxdu(self, u: jax.Array) -> jax.Array:
        return jnp.log(self.dxdu(u))

    def stitch_points(self) -> dict[str, jax.Array]:
        return {
            "u0": self._u0,
            "uN": self._uN,
            "x0": self._x0,
            "xN": self._xN,
            "du_dx_left": self._du_dx_left,
            "du_dx_right": self._du_dx_right,
        }
