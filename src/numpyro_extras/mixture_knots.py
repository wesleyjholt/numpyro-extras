"""Interpolation-knot generation utilities for quantile backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "KnotConfig",
    "QuantileKnotSet",
    "SlopeConfig",
    "build_quantile_knot_set",
]


@dataclass(frozen=True)
class KnotConfig:
    num_knots: int = 256
    u_min: float = 1e-6
    u_max: float = 1.0 - 1e-6
    grid_type: str = "logit_u"
    tail_density: float = 0.35
    min_delta_u: float = 1e-10


@dataclass(frozen=True)
class SlopeConfig:
    method: str = "finite_diff_u"
    delta_u: float = 5e-5
    min_positive_slope: float = 1e-8


@dataclass(frozen=True)
class QuantileKnotSet:
    u_knots: jax.Array
    x_knots: jax.Array
    du_dx_left: jax.Array
    du_dx_right: jax.Array
    meta: dict[str, Any]


def _coerce_knot_cfg(knot_cfg: Optional[Mapping[str, Any] | KnotConfig]) -> KnotConfig:
    if knot_cfg is None:
        cfg = KnotConfig()
    elif isinstance(knot_cfg, KnotConfig):
        cfg = knot_cfg
    elif isinstance(knot_cfg, Mapping):
        cfg = KnotConfig(
            num_knots=int(knot_cfg.get("num_knots", 256)),
            u_min=float(knot_cfg.get("u_min", 1e-6)),
            u_max=float(knot_cfg.get("u_max", 1.0 - 1e-6)),
            grid_type=str(knot_cfg.get("grid_type", "logit_u")),
            tail_density=float(knot_cfg.get("tail_density", 0.35)),
            min_delta_u=float(knot_cfg.get("min_delta_u", 1e-10)),
        )
    else:
        raise TypeError("`knot_cfg` must be None, dict-like, or `KnotConfig`.")

    if cfg.num_knots < 2:
        raise ValueError("`num_knots` must be >= 2.")
    if not (0.0 < cfg.u_min < cfg.u_max < 1.0):
        raise ValueError("`u_min` and `u_max` must satisfy 0 < u_min < u_max < 1.")
    if cfg.grid_type not in {"uniform_u", "logit_u", "hybrid"}:
        raise ValueError("`grid_type` must be one of {'uniform_u', 'logit_u', 'hybrid'}.")
    if not (0.0 < cfg.tail_density < 1.0):
        raise ValueError("`tail_density` must satisfy 0 < tail_density < 1.")
    if cfg.min_delta_u <= 0.0:
        raise ValueError("`min_delta_u` must be > 0.")
    if cfg.min_delta_u * (cfg.num_knots - 1) >= (cfg.u_max - cfg.u_min):
        raise ValueError("`min_delta_u` is too large for the configured knot range.")
    return cfg


def _coerce_slope_cfg(slope_cfg: Optional[Mapping[str, Any] | SlopeConfig]) -> SlopeConfig:
    if slope_cfg is None:
        cfg = SlopeConfig()
    elif isinstance(slope_cfg, SlopeConfig):
        cfg = slope_cfg
    elif isinstance(slope_cfg, Mapping):
        cfg = SlopeConfig(
            method=str(slope_cfg.get("method", "finite_diff_u")),
            delta_u=float(slope_cfg.get("delta_u", 5e-5)),
            min_positive_slope=float(slope_cfg.get("min_positive_slope", 1e-8)),
        )
    else:
        raise TypeError("`slope_cfg` must be None, dict-like, or `SlopeConfig`.")

    if cfg.method not in {"finite_diff_u", "autodiff_on_interp_seed"}:
        raise ValueError("`method` must be one of {'finite_diff_u', 'autodiff_on_interp_seed'}.")
    if cfg.delta_u <= 0.0:
        raise ValueError("`delta_u` must be > 0.")
    if cfg.min_positive_slope <= 0.0:
        raise ValueError("`min_positive_slope` must be > 0.")
    return cfg


def _logit(u: jax.Array) -> jax.Array:
    return jnp.log(u) - jnp.log1p(-u)


def _sigmoid(z: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(z)


def _build_u_grid(cfg: KnotConfig) -> jax.Array:
    if cfg.grid_type == "uniform_u":
        u = jnp.linspace(cfg.u_min, cfg.u_max, cfg.num_knots)
    else:
        z0 = _logit(jnp.asarray(cfg.u_min))
        z1 = _logit(jnp.asarray(cfg.u_max))
        # A slight center-warp keeps a logit-tail-focused grid while providing
        # enough interior resolution for quantized-backend edge cases.
        t = jnp.linspace(-1.0, 1.0, cfg.num_knots)
        t_warped = jnp.sign(t) * jnp.power(jnp.abs(t), 1.05)
        z = z0 + (z1 - z0) * (t_warped + 1.0) * 0.5
        u_logit = _sigmoid(z)
        if cfg.grid_type == "logit_u":
            u = u_logit
        else:
            u_uniform = jnp.linspace(cfg.u_min, cfg.u_max, cfg.num_knots)
            u = (1.0 - cfg.tail_density) * u_uniform + cfg.tail_density * u_logit

    u = jnp.asarray(u, dtype=jnp.result_type(cfg.u_min, cfg.u_max, jnp.float32))
    u = u.at[0].set(jnp.asarray(cfg.u_min, dtype=u.dtype))
    u = u.at[-1].set(jnp.asarray(cfg.u_max, dtype=u.dtype))
    return u


def _enforce_strict_u(u_knots: jax.Array, min_delta_u: float) -> jax.Array:
    u_np = np.asarray(u_knots, dtype=float)
    out = np.empty_like(u_np)
    out[0] = u_np[0]
    for i in range(1, out.size):
        out[i] = max(u_np[i], out[i - 1] + min_delta_u)
    return jnp.asarray(out, dtype=u_knots.dtype)


def _estimate_endpoint_slopes(
    u_knots: jax.Array, x_knots: jax.Array, slope_cfg: SlopeConfig
) -> tuple[jax.Array, jax.Array]:
    u_np = np.asarray(u_knots)
    x_np = np.asarray(x_knots)
    tiny_dx = max(np.finfo(x_np.dtype).eps, 1e-12)

    left_du = float(u_np[1] - u_np[0])
    right_du = float(u_np[-1] - u_np[-2])
    left_dx = float(x_np[1] - x_np[0])
    right_dx = float(x_np[-1] - x_np[-2])

    left = left_du / max(left_dx, tiny_dx)
    right = right_du / max(right_dx, tiny_dx)
    if not np.isfinite(left):
        left = slope_cfg.min_positive_slope
    if not np.isfinite(right):
        right = slope_cfg.min_positive_slope

    left = max(left, slope_cfg.min_positive_slope)
    right = max(right, slope_cfg.min_positive_slope)
    return jnp.asarray(left), jnp.asarray(right)


def build_quantile_knot_set(
    *,
    quantile_backend: Any,
    knot_cfg: Optional[Mapping[str, Any] | KnotConfig] = None,
    slope_cfg: Optional[Mapping[str, Any] | SlopeConfig] = None,
) -> QuantileKnotSet:
    if quantile_backend is None or not callable(getattr(quantile_backend, "icdf", None)):
        raise TypeError("`quantile_backend` must define callable `icdf(u)`.")

    knot_cfg_resolved = _coerce_knot_cfg(knot_cfg)
    slope_cfg_resolved = _coerce_slope_cfg(slope_cfg)

    u_grid = _build_u_grid(knot_cfg_resolved)
    x_raw = jnp.asarray(quantile_backend.icdf(u_grid))
    if x_raw.shape != u_grid.shape:
        x_raw = jnp.reshape(x_raw, u_grid.shape)

    finite_mask = jnp.isfinite(u_grid) & jnp.isfinite(x_raw)
    non_finite_count = int(np.asarray(jnp.sum(~finite_mask)))
    u_finite = u_grid[finite_mask]
    x_finite = x_raw[finite_mask]
    if u_finite.size < 2:
        raise ValueError("Not enough finite knot points to estimate endpoint slopes.")

    u_strict = _enforce_strict_u(u_finite, knot_cfg_resolved.min_delta_u)

    x_np = np.asarray(x_finite)
    monotonicity_violations = int(np.sum(np.diff(x_np) < 0.0))
    x_clean_np = np.maximum.accumulate(x_np)
    cleanup_count = int(np.sum(np.abs(x_clean_np - x_np) > 0.0))
    x_clean = jnp.asarray(x_clean_np, dtype=x_finite.dtype)

    du = jnp.diff(u_strict)
    dx = jnp.diff(x_clean)
    if du.size < 1:
        raise ValueError("Need at least two knot points after filtering/cleanup.")

    du_dx_left, du_dx_right = _estimate_endpoint_slopes(u_strict, x_clean, slope_cfg_resolved)

    meta: dict[str, Any] = {
        "grid_type": knot_cfg_resolved.grid_type,
        "point_count": int(u_strict.size),
        "cleanup_count": cleanup_count,
        "monotonicity_violations": monotonicity_violations,
        "non_finite_count": non_finite_count,
        "min_du": float(np.asarray(jnp.min(du))),
        "max_du": float(np.asarray(jnp.max(du))),
        "min_dx": float(np.asarray(jnp.min(dx))),
        "max_dx": float(np.asarray(jnp.max(dx))),
        "small_n_warning": bool(u_strict.size < 16),
        "slope_method": slope_cfg_resolved.method,
        "clip_epsilon": min(knot_cfg_resolved.u_min, 1.0 - knot_cfg_resolved.u_max),
    }

    return QuantileKnotSet(
        u_knots=u_strict,
        x_knots=x_clean,
        du_dx_left=du_dx_left,
        du_dx_right=du_dx_right,
        meta=meta,
    )
