"""Knot generation utilities for quantile interpolation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import jax
import jax.numpy as jnp


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
    du_dx_left: float
    du_dx_right: float
    meta: dict[str, Any]


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_knot_cfg(cfg: KnotConfig | Mapping[str, Any] | Any | None) -> KnotConfig:
    if cfg is None:
        return KnotConfig()
    if isinstance(cfg, KnotConfig):
        return cfg
    kwargs = {f.name: _cfg_get(cfg, f.name, f.default) for f in fields(KnotConfig)}
    return KnotConfig(**kwargs)


def _as_slope_cfg(cfg: SlopeConfig | Mapping[str, Any] | Any | None) -> SlopeConfig:
    if cfg is None:
        return SlopeConfig()
    if isinstance(cfg, SlopeConfig):
        return cfg
    kwargs = {f.name: _cfg_get(cfg, f.name, f.default) for f in fields(SlopeConfig)}
    return SlopeConfig(**kwargs)


def _validate_knot_cfg(cfg: KnotConfig) -> None:
    if int(cfg.num_knots) < 3:
        raise ValueError("`num_knots` must be >= 3.")
    if not jnp.isfinite(cfg.u_min) or not jnp.isfinite(cfg.u_max):
        raise ValueError("`u_min` and `u_max` must be finite.")
    if not (0.0 < float(cfg.u_min) < float(cfg.u_max) < 1.0):
        raise ValueError("`u_min` and `u_max` must satisfy 0 < u_min < u_max < 1.")
    if cfg.grid_type not in {"uniform_u", "logit_u", "hybrid"}:
        raise ValueError(
            "`grid_type` must be one of {'uniform_u', 'logit_u', 'hybrid'}."
        )
    if not jnp.isfinite(cfg.tail_density) or not (0.0 <= float(cfg.tail_density) <= 1.0):
        raise ValueError("`tail_density` must be finite and in [0, 1].")
    if not jnp.isfinite(cfg.min_delta_u) or not (float(cfg.min_delta_u) > 0.0):
        raise ValueError("`min_delta_u` must be finite and > 0.")
    width = float(cfg.u_max - cfg.u_min)
    if float(cfg.min_delta_u) * (int(cfg.num_knots) - 1) >= width:
        raise ValueError(
            "`min_delta_u` is too large for requested knot count and u bounds."
        )


def _validate_slope_cfg(cfg: SlopeConfig) -> None:
    if cfg.method not in {"finite_diff_u", "autodiff_on_interp_seed"}:
        raise ValueError(
            "`slope_cfg.method` must be 'finite_diff_u' or 'autodiff_on_interp_seed'."
        )
    if not jnp.isfinite(cfg.delta_u) or not (float(cfg.delta_u) > 0.0):
        raise ValueError("`slope_cfg.delta_u` must be finite and > 0.")
    if not jnp.isfinite(cfg.min_positive_slope) or not (
        float(cfg.min_positive_slope) > 0.0
    ):
        raise ValueError("`slope_cfg.min_positive_slope` must be finite and > 0.")


def _logit(u: jax.Array) -> jax.Array:
    return jnp.log(u) - jnp.log1p(-u)


def _expit(z: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(z)


def _enforce_min_delta_u(u: jax.Array, min_delta_u: float) -> jax.Array:
    du = jnp.diff(u)
    du = jnp.maximum(du, jnp.asarray(min_delta_u, dtype=u.dtype))
    return u[0] + jnp.concatenate([jnp.zeros((1,), dtype=u.dtype), jnp.cumsum(du)])


def _build_u_grid(cfg: KnotConfig, dtype: jnp.dtype) -> jax.Array:
    u_min = jnp.asarray(cfg.u_min, dtype=dtype)
    u_max = jnp.asarray(cfg.u_max, dtype=dtype)
    n = int(cfg.num_knots)

    uniform = jnp.linspace(u_min, u_max, n, dtype=dtype)
    if cfg.grid_type == "uniform_u":
        return uniform

    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    u_min_clip = jnp.clip(u_min, eps, 1.0 - eps)
    u_max_clip = jnp.clip(u_max, eps, 1.0 - eps)
    z = jnp.linspace(_logit(u_min_clip), _logit(u_max_clip), n, dtype=dtype)
    logit_grid = _expit(z)

    if cfg.grid_type == "logit_u":
        return logit_grid

    alpha = jnp.asarray(cfg.tail_density, dtype=dtype)
    return (1.0 - alpha) * uniform + alpha * logit_grid


def _estimate_endpoint_slope_seeds(
    u_knots: jax.Array, x_knots: jax.Array, slope_cfg: SlopeConfig
) -> tuple[float, float]:
    du_left = u_knots[1] - u_knots[0]
    du_right = u_knots[-1] - u_knots[-2]
    dx_left = x_knots[1] - x_knots[0]
    dx_right = x_knots[-1] - x_knots[-2]

    dtype = x_knots.dtype
    tiny_dx = jnp.maximum(
        jnp.asarray(slope_cfg.delta_u, dtype=dtype) * jnp.asarray(1e-6, dtype=dtype),
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    floor = jnp.asarray(slope_cfg.min_positive_slope, dtype=dtype)

    left = du_left / jnp.maximum(dx_left, tiny_dx)
    right = du_right / jnp.maximum(dx_right, tiny_dx)
    left = jnp.where(jnp.isfinite(left), jnp.maximum(left, floor), floor)
    right = jnp.where(jnp.isfinite(right), jnp.maximum(right, floor), floor)
    return float(left), float(right)


def _meta_roundtrip_error_if_available(
    quantile_backend: Any, u_knots: jax.Array, x_knots: jax.Array
) -> dict[str, float]:
    cdf_fn = getattr(quantile_backend, "cdf", None)
    if not callable(cdf_fn):
        return {}
    try:
        u_roundtrip = jnp.asarray(cdf_fn(x_knots), dtype=x_knots.dtype).reshape(u_knots.shape)
        err = jnp.abs(u_roundtrip - u_knots)
        return {
            "roundtrip_max_abs_error": float(jnp.max(err)),
            "roundtrip_mean_abs_error": float(jnp.mean(err)),
        }
    except Exception:
        return {}


def build_quantile_knot_set(
    *,
    quantile_backend: Any,
    knot_cfg: KnotConfig | Mapping[str, Any] | Any | None = None,
    slope_cfg: SlopeConfig | Mapping[str, Any] | Any | None = None,
) -> QuantileKnotSet:
    """Build a deterministic knot set `(u_i, x_i)` from a quantile backend."""

    icdf_fn = getattr(quantile_backend, "icdf", None)
    if not callable(icdf_fn):
        raise ValueError("`quantile_backend` must expose callable `icdf(u)`.")

    knot_cfg_obj = _as_knot_cfg(knot_cfg)
    slope_cfg_obj = _as_slope_cfg(slope_cfg)
    _validate_knot_cfg(knot_cfg_obj)
    _validate_slope_cfg(slope_cfg_obj)

    dtype = jnp.asarray(float(knot_cfg_obj.u_min)).dtype
    u_raw = _build_u_grid(knot_cfg_obj, dtype=dtype)
    u_raw = _enforce_min_delta_u(u_raw, knot_cfg_obj.min_delta_u)

    x_raw = jnp.asarray(icdf_fn(u_raw))
    if x_raw.shape != u_raw.shape:
        x_raw = jnp.asarray(x_raw, dtype=u_raw.dtype).reshape(u_raw.shape)

    finite_mask = jnp.isfinite(u_raw) & jnp.isfinite(x_raw)
    non_finite_count = int(u_raw.shape[0] - int(jnp.sum(finite_mask)))

    u_knots = u_raw[finite_mask]
    x_knots = x_raw[finite_mask]
    if int(u_knots.shape[0]) < 3:
        raise ValueError(
            "Knot generation produced fewer than 3 finite knots after filtering."
        )

    u_knots = _enforce_min_delta_u(u_knots, knot_cfg_obj.min_delta_u)

    monotonicity_violations = int(jnp.sum(jnp.diff(x_knots) < 0.0))
    x_clean = jnp.maximum.accumulate(x_knots)
    cleanup_count = int(jnp.sum(x_clean != x_knots))

    midpoint_violations = 0
    try:
        u_mid = 0.5 * (u_knots[:-1] + u_knots[1:])
        x_mid = jnp.asarray(icdf_fn(u_mid))
        if x_mid.shape != u_mid.shape:
            x_mid = x_mid.reshape(u_mid.shape)
        midpoint_violations = int(
            jnp.sum(
                (~jnp.isfinite(x_mid))
                | (x_mid < x_knots[:-1])
                | (x_mid > x_knots[1:])
            )
        )
    except Exception:
        midpoint_violations = 0

    cleanup_count = max(cleanup_count, midpoint_violations)
    x_knots = x_clean

    du_dx_left, du_dx_right = _estimate_endpoint_slope_seeds(
        u_knots, x_knots, slope_cfg_obj
    )

    du = jnp.diff(u_knots)
    dx = jnp.diff(x_knots)
    meta: dict[str, Any] = {
        "grid_type": knot_cfg_obj.grid_type,
        "num_knots": int(u_knots.shape[0]),
        "cleanup_count": cleanup_count,
        "pre_cleanup_monotonicity_violations": monotonicity_violations,
        "midpoint_monotonicity_violations": midpoint_violations,
        "min_du": float(jnp.min(du)),
        "max_du": float(jnp.max(du)),
        "min_dx": float(jnp.min(dx)),
        "max_dx": float(jnp.max(dx)),
        "non_finite_count": non_finite_count,
        "small_num_knots_warning": bool(int(knot_cfg_obj.num_knots) < 16),
        "endpoint_slope_role": (
            "seed_only_for_non_c1_fallback; not canonical C1 slope"
        ),
        "clip_epsilon": float(min(knot_cfg_obj.u_min, 1.0 - knot_cfg_obj.u_max)),
        "requested_num_knots": int(knot_cfg_obj.num_knots),
        "slope_method": slope_cfg_obj.method,
    }
    meta.update(_meta_roundtrip_error_if_available(quantile_backend, u_knots, x_knots))

    return QuantileKnotSet(
        u_knots=u_knots,
        x_knots=x_knots,
        du_dx_left=du_dx_left,
        du_dx_right=du_dx_right,
        meta=meta,
    )


__all__ = [
    "KnotConfig",
    "SlopeConfig",
    "QuantileKnotSet",
    "build_quantile_knot_set",
]
