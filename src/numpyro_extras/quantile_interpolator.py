"""Quantile interpolation with sigmoid tails for stable CDF/ICDF approximations."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from interpax import Interpolator1D


@dataclass(frozen=True)
class InterpConfig:
    interior_method: str = "akima"
    clip_u_eps: float = 1e-10
    safe_arctanh_eps: float = 1e-7


@dataclass(frozen=True)
class TailConfig:
    enforce_c1_stitch: bool = True
    min_tail_scale: float = 1e-8


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_interp_cfg(cfg: InterpConfig | Mapping[str, Any] | Any | None) -> InterpConfig:
    if cfg is None:
        return InterpConfig()
    if isinstance(cfg, InterpConfig):
        return cfg
    kwargs = {f.name: _cfg_get(cfg, f.name, f.default) for f in fields(InterpConfig)}
    return InterpConfig(**kwargs)


def _as_tail_cfg(cfg: TailConfig | Mapping[str, Any] | Any | None) -> TailConfig:
    if cfg is None:
        return TailConfig()
    if isinstance(cfg, TailConfig):
        return cfg
    kwargs = {f.name: _cfg_get(cfg, f.name, f.default) for f in fields(TailConfig)}
    return TailConfig(**kwargs)


def _method_to_interpax(method: str) -> str:
    mapping = {
        "akima": "akima",
        "linear": "linear",
        "pchip_like": "monotonic",
    }
    if method not in mapping:
        raise ValueError("`interior_method` must be one of {'akima', 'linear', 'pchip_like'}.")
    return mapping[method]


def _validate_interp_cfg(cfg: InterpConfig) -> None:
    _method_to_interpax(cfg.interior_method)
    if not jnp.isfinite(cfg.clip_u_eps) or not (0.0 < float(cfg.clip_u_eps) < 0.5):
        raise ValueError("`clip_u_eps` must be finite and satisfy 0 < clip_u_eps < 0.5.")
    if not jnp.isfinite(cfg.safe_arctanh_eps) or not (
        0.0 < float(cfg.safe_arctanh_eps) < 1.0
    ):
        raise ValueError(
            "`safe_arctanh_eps` must be finite and satisfy 0 < safe_arctanh_eps < 1."
        )


def _validate_tail_cfg(cfg: TailConfig) -> None:
    if not jnp.isfinite(cfg.min_tail_scale) or not (float(cfg.min_tail_scale) > 0.0):
        raise ValueError("`min_tail_scale` must be finite and > 0.")


def _sech2(z: jax.Array) -> jax.Array:
    c = jnp.cosh(z)
    return 1.0 / (c * c)


def _safe_positive(value: jax.Array, floor: jax.Array) -> jax.Array:
    return jnp.where(jnp.isfinite(value) & (value > 0.0), value, floor)


@dataclass(frozen=True)
class QuantileInterpolator1D:
    """Bidirectional monotone interpolator with ECDF-style tanh tails."""

    u_knots: jax.Array
    x_knots: jax.Array
    u0: jax.Array
    uN: jax.Array
    x0: jax.Array
    xN: jax.Array
    m0: jax.Array
    mN: jax.Array
    clip_u_eps: float
    safe_arctanh_eps: float
    min_tail_scale: float
    enforce_c1_stitch: bool
    interior_method: str
    stitch_h: float
    _x_of_u_value_interp: Interpolator1D
    _u_of_x_value_interp: Interpolator1D
    _u_of_x_slope_interp: Interpolator1D

    def __init__(
        self,
        *,
        knot_set: Any,
        interp_cfg: InterpConfig | Mapping[str, Any] | Any | None = None,
        tail_cfg: TailConfig | Mapping[str, Any] | Any | None = None,
    ) -> None:
        interp_cfg_obj = _as_interp_cfg(interp_cfg)
        tail_cfg_obj = _as_tail_cfg(tail_cfg)
        _validate_interp_cfg(interp_cfg_obj)
        _validate_tail_cfg(tail_cfg_obj)

        u_knots = jnp.asarray(getattr(knot_set, "u_knots"))
        x_knots = jnp.asarray(getattr(knot_set, "x_knots"))
        if u_knots.ndim != 1 or x_knots.ndim != 1:
            raise ValueError("`u_knots` and `x_knots` must be rank-1 arrays.")
        if u_knots.shape != x_knots.shape:
            raise ValueError("`u_knots` and `x_knots` must have the same shape.")
        if int(u_knots.shape[0]) < 3:
            raise ValueError("At least 3 knots are required.")
        if not bool(jnp.all(jnp.isfinite(u_knots)) and jnp.all(jnp.isfinite(x_knots))):
            raise ValueError("Knot arrays must contain only finite values.")
        if not bool(jnp.all((u_knots > 0.0) & (u_knots < 1.0))):
            raise ValueError("`u_knots` must lie strictly within (0, 1).")
        if not bool(jnp.all(jnp.diff(u_knots) > 0.0)):
            raise ValueError("`u_knots` must be strictly increasing.")
        if not bool(jnp.all(jnp.diff(x_knots) > 0.0)):
            raise ValueError("`x_knots` must be strictly increasing.")

        dtype = jnp.result_type(u_knots.dtype, x_knots.dtype)
        u_knots = jnp.asarray(u_knots, dtype=dtype)
        x_knots = jnp.asarray(x_knots, dtype=dtype)

        value_method = "linear"
        slope_method = _method_to_interpax(interp_cfg_obj.interior_method)
        x_of_u_value_interp = Interpolator1D(
            u_knots, x_knots, method=value_method, extrap=False
        )
        u_of_x_value_interp = Interpolator1D(
            x_knots, u_knots, method=value_method, extrap=False
        )
        u_of_x_slope_interp = Interpolator1D(
            x_knots, u_knots, method=slope_method, extrap=False
        )

        u0 = u_knots[0]
        uN = u_knots[-1]
        x0 = x_knots[0]
        xN = x_knots[-1]

        floor = jnp.asarray(tail_cfg_obj.min_tail_scale, dtype=dtype)
        stitch_h = jnp.asarray(1e-6, dtype=dtype) * jnp.maximum(
            jnp.asarray(1.0, dtype=dtype), jnp.maximum(jnp.abs(x0), jnp.abs(xN))
        )
        cdf_blend_h = jnp.asarray(4.0, dtype=dtype) * stitch_h
        if tail_cfg_obj.enforce_c1_stitch:
            m0_raw = jnp.asarray(u_of_x_slope_interp(x0 + stitch_h, dx=1), dtype=dtype)
            mN_raw = jnp.asarray(u_of_x_slope_interp(xN - stitch_h, dx=1), dtype=dtype)
        else:
            m0_raw = jnp.asarray(getattr(knot_set, "du_dx_left"), dtype=dtype)
            mN_raw = jnp.asarray(getattr(knot_set, "du_dx_right"), dtype=dtype)

        m0 = _safe_positive(m0_raw, floor)
        mN = _safe_positive(mN_raw, floor)

        object.__setattr__(self, "u_knots", u_knots)
        object.__setattr__(self, "x_knots", x_knots)
        object.__setattr__(self, "u0", u0)
        object.__setattr__(self, "uN", uN)
        object.__setattr__(self, "x0", x0)
        object.__setattr__(self, "xN", xN)
        object.__setattr__(self, "m0", m0)
        object.__setattr__(self, "mN", mN)
        object.__setattr__(self, "clip_u_eps", float(interp_cfg_obj.clip_u_eps))
        object.__setattr__(
            self, "safe_arctanh_eps", float(interp_cfg_obj.safe_arctanh_eps)
        )
        object.__setattr__(self, "min_tail_scale", float(tail_cfg_obj.min_tail_scale))
        object.__setattr__(self, "enforce_c1_stitch", bool(tail_cfg_obj.enforce_c1_stitch))
        object.__setattr__(self, "interior_method", str(interp_cfg_obj.interior_method))
        object.__setattr__(self, "stitch_h", float(cdf_blend_h))
        object.__setattr__(self, "_x_of_u_value_interp", x_of_u_value_interp)
        object.__setattr__(self, "_u_of_x_value_interp", u_of_x_value_interp)
        object.__setattr__(self, "_u_of_x_slope_interp", u_of_x_slope_interp)

    def _clip_u(self, u: Any) -> jax.Array:
        u_arr = jnp.asarray(u, dtype=self.u_knots.dtype)
        return jnp.clip(u_arr, self.clip_u_eps, 1.0 - self.clip_u_eps)

    def _clip_arctanh_arg(self, z: jax.Array) -> jax.Array:
        eps = jnp.asarray(self.safe_arctanh_eps, dtype=self.u_knots.dtype)
        return jnp.clip(z, -1.0 + eps, 1.0 - eps)

    def cdf(self, x: Any) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=self.x_knots.dtype)
        left = x_arr < self.x0
        right = x_arr > self.xN
        interior_x = jnp.clip(x_arr, self.x0, self.xN)
        u_interior = jnp.asarray(
            self._u_of_x_value_interp(interior_x), dtype=self.x_knots.dtype
        )
        u_interior = jnp.clip(u_interior, self.u0, self.uN)

        blend_h = jnp.asarray(self.stitch_h, dtype=self.x_knots.dtype)
        left_tangent = self.u0 + self.m0 * (interior_x - self.x0)
        right_tangent = self.uN + self.mN * (interior_x - self.xN)
        u_interior = jnp.where(interior_x <= self.x0 + blend_h, left_tangent, u_interior)
        u_interior = jnp.where(interior_x >= self.xN - blend_h, right_tangent, u_interior)
        u_interior = jnp.clip(u_interior, self.u0, self.uN)

        tiny = jnp.asarray(jnp.finfo(self.u0.dtype).tiny, dtype=self.u0.dtype)
        u0_scale = jnp.maximum(self.u0, tiny)
        one_minus_uN = 1.0 - self.uN
        uN_scale = jnp.maximum(one_minus_uN, tiny)

        left_arg = (self.m0 / u0_scale) * (x_arr - self.x0)
        right_arg = (self.mN / uN_scale) * (x_arr - self.xN)
        u_left = self.u0 + self.u0 * jnp.tanh(left_arg)
        u_right = self.uN + one_minus_uN * jnp.tanh(right_arg)

        u = jnp.where(left, u_left, jnp.where(right, u_right, u_interior))
        return jnp.clip(u, 0.0, 1.0)

    def icdf(self, u: Any) -> jax.Array:
        u_arr = self._clip_u(u)
        left = u_arr < self.u0
        right = u_arr > self.uN
        interior_u = jnp.clip(u_arr, self.u0, self.uN)
        x_interior = jnp.asarray(
            self._x_of_u_value_interp(interior_u), dtype=self.x_knots.dtype
        )
        x_interior = jnp.clip(x_interior, self.x0, self.xN)

        tiny = jnp.asarray(jnp.finfo(self.u0.dtype).tiny, dtype=self.u0.dtype)
        u0_scale = jnp.maximum(self.u0, tiny)
        one_minus_uN = 1.0 - self.uN
        uN_scale = jnp.maximum(one_minus_uN, tiny)

        z_left = self._clip_arctanh_arg((u_arr - self.u0) / u0_scale)
        z_right = self._clip_arctanh_arg((u_arr - self.uN) / uN_scale)
        x_left = self.x0 + (self.u0 / self.m0) * jnp.arctanh(z_left)
        x_right = self.xN + (one_minus_uN / self.mN) * jnp.arctanh(z_right)

        return jnp.where(left, x_left, jnp.where(right, x_right, x_interior))

    def dudx(self, x: Any) -> jax.Array:
        x_arr = jnp.asarray(x, dtype=self.x_knots.dtype)
        left = x_arr < self.x0
        right = x_arr > self.xN
        interior_x = jnp.clip(x_arr, self.x0, self.xN)
        dudx_interior = jnp.asarray(
            self._u_of_x_slope_interp(interior_x, dx=1), dtype=self.x_knots.dtype
        )

        tiny = jnp.asarray(jnp.finfo(self.u0.dtype).tiny, dtype=self.u0.dtype)
        u0_scale = jnp.maximum(self.u0, tiny)
        one_minus_uN = 1.0 - self.uN
        uN_scale = jnp.maximum(one_minus_uN, tiny)

        left_arg = (self.m0 / u0_scale) * (x_arr - self.x0)
        right_arg = (self.mN / uN_scale) * (x_arr - self.xN)
        dudx_left = self.m0 * _sech2(left_arg)
        dudx_right = self.mN * _sech2(right_arg)

        dudx = jnp.where(left, dudx_left, jnp.where(right, dudx_right, dudx_interior))
        finite_dudx = jnp.where(jnp.isfinite(dudx), dudx, tiny)
        return jnp.maximum(finite_dudx, tiny)

    def dxdu(self, u: Any) -> jax.Array:
        u_arr = self._clip_u(u)
        x_arr = self.icdf(u_arr)
        dudx_vals = self.dudx(x_arr)
        tiny = jnp.asarray(jnp.finfo(dudx_vals.dtype).tiny, dtype=dudx_vals.dtype)
        safe_dudx = jnp.maximum(jnp.where(jnp.isfinite(dudx_vals), dudx_vals, tiny), tiny)
        return 1.0 / safe_dudx

    def log_abs_dxdu(self, u: Any) -> jax.Array:
        dxdu = self.dxdu(u)
        tiny = jnp.asarray(jnp.finfo(dxdu.dtype).tiny, dtype=dxdu.dtype)
        return jnp.log(jnp.maximum(jnp.abs(dxdu), tiny))

    def stitch_points(self) -> dict[str, float]:
        return {
            "u0": float(self.u0),
            "uN": float(self.uN),
            "x0": float(self.x0),
            "xN": float(self.xN),
            "m0": float(self.m0),
            "mN": float(self.mN),
            "enforce_c1_stitch": bool(self.enforce_c1_stitch),
            "interior_method": self.interior_method,
        }


__all__ = [
    "InterpConfig",
    "TailConfig",
    "QuantileInterpolator1D",
]
