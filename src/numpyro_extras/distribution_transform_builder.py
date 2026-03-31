"""Integration builder for 1D distribution quantile transforms.

This module assembles the end-to-end pipeline:
distribution quantile backend -> knots -> interpolator -> NumPyro transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .distribution_quantile import (
    BracketConfig,
    SolverConfig,
    build_distribution_quantile_backend,
)
from .quantile_interpolator import (
    InterpConfig,
    QuantileInterpolator1D,
    TailConfig as InterpolatorTailConfig,
)
from .quantile_knots import (
    KnotConfig,
    QuantileKnotSet,
    SlopeConfig,
    build_quantile_knot_set,
)
from .transforms import NormalToDistributionTransform, UniformToDistributionTransform


@dataclass(frozen=True)
class DistributionQuantileConfig:
    """Numerical settings for backend inverse-CDF solves."""

    rtol: float = 1e-8
    atol: float = 1e-8
    max_steps: int = 256
    bracket_cfg: BracketConfig = field(default_factory=BracketConfig)
    eps: float = 1e-10


@dataclass(frozen=True)
class KnotGenerationConfig:
    """Knot placement and endpoint slope-seed settings."""

    num_knots: int = 257
    u_min: float = 1e-6
    u_max: float = 1.0 - 1e-6
    grid_type: str = "logit_u"
    tail_density: float = 0.35
    min_delta_u: float = 1e-10
    slope_method: str = "finite_diff_u"
    slope_delta_u: float = 5e-5
    min_positive_slope: float = 1e-8


@dataclass(frozen=True)
class InterpolatorConfig:
    interior_method: str = "akima"
    clip_u_eps: float = 1e-12
    safe_arctanh_eps: float = 1e-7
    cdf_eval_method: str = "interpolate"
    cdf_root_max_steps: int = 64
    cdf_root_u_tol: float = 1e-10


@dataclass(frozen=True)
class TailConfig:
    enforce_c1_stitch: bool = True
    min_tail_scale: float = 1e-8


@dataclass(frozen=True)
class TransformConfig:
    clip_u_eps: float = 1e-12
    validate_args: bool = False


@dataclass(frozen=True)
class DistributionTransformBuildConfig:
    solver_cfg: DistributionQuantileConfig = field(default_factory=DistributionQuantileConfig)
    knot_cfg: KnotGenerationConfig = field(default_factory=KnotGenerationConfig)
    interp_cfg: InterpolatorConfig = field(default_factory=InterpolatorConfig)
    tail_cfg: TailConfig = field(default_factory=TailConfig)
    transform_cfg: TransformConfig = field(default_factory=TransformConfig)


@dataclass
class DistributionTransformBuildError(ValueError):
    """Builder error with stage context and diagnostics handles."""

    message: str
    stage: str
    context: Mapping[str, Any] | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        ValueError.__init__(self, self.message)
        if self.failure_reason is None:
            self.failure_reason = self.message
        self.diagnostics = {
            "stage": self.stage,
            "reason": self.failure_reason,
            "context": dict(self.context or {}),
        }


@dataclass(frozen=True)
class DistributionTransformBuildResult:
    transform: Any
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _ShapeSafeInterpolator:
    base: Any

    def _map_preserve_shape(self, fn_name: str, values: Any) -> jax.Array:
        x = jnp.asarray(values)
        flat = x.reshape(-1)
        fn = getattr(self.base, fn_name)
        out_flat = jnp.asarray(fn(flat))
        return out_flat.reshape(x.shape)

    def cdf(self, x: Any) -> jax.Array:
        return self._map_preserve_shape("cdf", x)

    def icdf(self, u: Any) -> jax.Array:
        return self._map_preserve_shape("icdf", u)

    def dudx(self, x: Any) -> jax.Array:
        return self._map_preserve_shape("dudx", x)

    def dxdu(self, u: Any) -> jax.Array:
        return self._map_preserve_shape("dxdu", u)

    def log_abs_dxdu(self, u: Any) -> jax.Array:
        return self._map_preserve_shape("log_abs_dxdu", u)

    def stitch_points(self) -> dict[str, float]:
        return self.base.stitch_points()


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_mapping(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return {
            key: value
            for key, value in vars(cfg).items()
            if not key.startswith("_")
        }
    return {}


def _as_bracket_cfg(cfg: Any) -> BracketConfig:
    if cfg is None:
        return BracketConfig()
    if isinstance(cfg, BracketConfig):
        return cfg
    payload = _cfg_to_mapping(cfg)
    return BracketConfig(
        x_init_low=float(_cfg_get(payload, "x_init_low", BracketConfig.x_init_low)),
        x_init_high=float(_cfg_get(payload, "x_init_high", BracketConfig.x_init_high)),
        expansion_factor=float(
            _cfg_get(payload, "expansion_factor", BracketConfig.expansion_factor)
        ),
        max_expansions=int(
            _cfg_get(payload, "max_expansions", BracketConfig.max_expansions)
        ),
    )


def _as_distribution_quantile_cfg(cfg: Any) -> DistributionQuantileConfig:
    if cfg is None:
        return DistributionQuantileConfig()
    if isinstance(cfg, DistributionQuantileConfig):
        return cfg
    payload = _cfg_to_mapping(cfg)
    raw_max_steps = _cfg_get(payload, "max_steps", None)
    if raw_max_steps is None:
        raw_max_steps = _cfg_get(payload, "maxiter", DistributionQuantileConfig.max_steps)
    return DistributionQuantileConfig(
        rtol=float(_cfg_get(payload, "rtol", DistributionQuantileConfig.rtol)),
        atol=float(_cfg_get(payload, "atol", DistributionQuantileConfig.atol)),
        max_steps=int(raw_max_steps),
        bracket_cfg=_as_bracket_cfg(_cfg_get(payload, "bracket_cfg", None)),
        eps=float(_cfg_get(payload, "eps", DistributionQuantileConfig.eps)),
    )


def _as_knot_cfg(cfg: Any) -> KnotGenerationConfig:
    if cfg is None:
        return KnotGenerationConfig()
    if isinstance(cfg, KnotGenerationConfig):
        return cfg

    payload = _cfg_to_mapping(cfg)
    slope_cfg = _cfg_to_mapping(_cfg_get(payload, "slope_cfg", None))

    return KnotGenerationConfig(
        num_knots=int(_cfg_get(payload, "num_knots", KnotGenerationConfig.num_knots)),
        u_min=float(_cfg_get(payload, "u_min", KnotGenerationConfig.u_min)),
        u_max=float(_cfg_get(payload, "u_max", KnotGenerationConfig.u_max)),
        grid_type=str(_cfg_get(payload, "grid_type", KnotGenerationConfig.grid_type)),
        tail_density=float(
            _cfg_get(payload, "tail_density", KnotGenerationConfig.tail_density)
        ),
        min_delta_u=float(
            _cfg_get(payload, "min_delta_u", KnotGenerationConfig.min_delta_u)
        ),
        slope_method=str(
            _cfg_get(
                slope_cfg,
                "method",
                _cfg_get(payload, "slope_method", KnotGenerationConfig.slope_method),
            )
        ),
        slope_delta_u=float(
            _cfg_get(
                slope_cfg,
                "delta_u",
                _cfg_get(payload, "slope_delta_u", KnotGenerationConfig.slope_delta_u),
            )
        ),
        min_positive_slope=float(
            _cfg_get(
                slope_cfg,
                "min_positive_slope",
                _cfg_get(
                    payload,
                    "min_positive_slope",
                    KnotGenerationConfig.min_positive_slope,
                ),
            )
        ),
    )


def _as_interp_cfg(cfg: Any) -> InterpolatorConfig:
    if cfg is None:
        return InterpolatorConfig()
    if isinstance(cfg, InterpolatorConfig):
        return cfg
    payload = _cfg_to_mapping(cfg)
    return InterpolatorConfig(
        interior_method=str(
            _cfg_get(payload, "interior_method", InterpolatorConfig.interior_method)
        ),
        clip_u_eps=float(_cfg_get(payload, "clip_u_eps", InterpolatorConfig.clip_u_eps)),
        safe_arctanh_eps=float(
            _cfg_get(payload, "safe_arctanh_eps", InterpolatorConfig.safe_arctanh_eps)
        ),
        cdf_eval_method=str(
            _cfg_get(payload, "cdf_eval_method", InterpolatorConfig.cdf_eval_method)
        ),
        cdf_root_max_steps=int(
            _cfg_get(payload, "cdf_root_max_steps", InterpolatorConfig.cdf_root_max_steps)
        ),
        cdf_root_u_tol=float(
            _cfg_get(payload, "cdf_root_u_tol", InterpolatorConfig.cdf_root_u_tol)
        ),
    )


def _as_tail_cfg(cfg: Any) -> TailConfig:
    if cfg is None:
        return TailConfig()
    if isinstance(cfg, TailConfig):
        return cfg
    payload = _cfg_to_mapping(cfg)
    return TailConfig(
        enforce_c1_stitch=bool(
            _cfg_get(payload, "enforce_c1_stitch", TailConfig.enforce_c1_stitch)
        ),
        min_tail_scale=float(
            _cfg_get(payload, "min_tail_scale", TailConfig.min_tail_scale)
        ),
    )


def _as_transform_cfg(cfg: Any) -> TransformConfig:
    if cfg is None:
        return TransformConfig()
    if isinstance(cfg, TransformConfig):
        return cfg
    payload = _cfg_to_mapping(cfg)
    return TransformConfig(
        clip_u_eps=float(_cfg_get(payload, "clip_u_eps", TransformConfig.clip_u_eps)),
        validate_args=bool(
            _cfg_get(payload, "validate_args", TransformConfig.validate_args)
        ),
    )


def _as_build_cfg(
    *,
    build_cfg: Any,
    config: Any,
    cfg: Any,
    distribution_transform_cfg: Any,
    solver_cfg: Any,
    knot_cfg: Any,
    interp_cfg: Any,
    tail_cfg: Any,
    transform_cfg: Any,
) -> DistributionTransformBuildConfig:
    candidates = [build_cfg, config, cfg, distribution_transform_cfg]
    base_payload: dict[str, Any] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, DistributionTransformBuildConfig):
            base_payload = {
                "solver_cfg": candidate.solver_cfg,
                "knot_cfg": candidate.knot_cfg,
                "interp_cfg": candidate.interp_cfg,
                "tail_cfg": candidate.tail_cfg,
                "transform_cfg": candidate.transform_cfg,
            }
            break
        base_payload = _cfg_to_mapping(candidate)
        if base_payload:
            break

    if solver_cfg is None:
        solver_cfg = _cfg_get(base_payload, "solver_cfg", None)
    if knot_cfg is None:
        knot_cfg = _cfg_get(base_payload, "knot_cfg", None)
    if interp_cfg is None:
        interp_cfg = _cfg_get(base_payload, "interp_cfg", None)
    if tail_cfg is None:
        tail_cfg = _cfg_get(base_payload, "tail_cfg", None)
    if transform_cfg is None:
        transform_cfg = _cfg_get(base_payload, "transform_cfg", None)

    return DistributionTransformBuildConfig(
        solver_cfg=_as_distribution_quantile_cfg(solver_cfg),
        knot_cfg=_as_knot_cfg(knot_cfg),
        interp_cfg=_as_interp_cfg(interp_cfg),
        tail_cfg=_as_tail_cfg(tail_cfg),
        transform_cfg=_as_transform_cfg(transform_cfg),
    )


def _validate_distribution(distribution: Any) -> None:
    cdf_fn = getattr(distribution, "cdf", None)
    log_prob_fn = getattr(distribution, "log_prob", None)
    if not callable(cdf_fn) or not callable(log_prob_fn):
        raise ValueError(
            "`distribution` must expose callable `cdf` and `log_prob`."
        )
    try:
        probe = jnp.asarray(0.0)
        _ = jnp.asarray(distribution.cdf(probe))
        _ = jnp.asarray(distribution.log_prob(probe))
    except Exception as exc:
        raise ValueError(
            "Target distribution must support scalar `cdf` and `log_prob` evaluation."
        ) from exc


def _validate_knot_set_or_raise(knot_set: QuantileKnotSet) -> None:
    u_knots = jnp.asarray(knot_set.u_knots)
    x_knots = jnp.asarray(knot_set.x_knots)
    if u_knots.ndim != 1 or x_knots.ndim != 1:
        raise ValueError("Invalid knot set: expected rank-1 `u_knots` and `x_knots`.")
    if u_knots.shape != x_knots.shape:
        raise ValueError("Invalid knot set: knot arrays must have matching shape.")
    if int(u_knots.shape[0]) < 3:
        raise ValueError("Invalid knot set: expected at least 3 knots.")
    if not bool(jnp.all(jnp.diff(u_knots) > 0.0)):
        raise ValueError("Invalid knot set: `u_knots` must be strictly increasing.")
    if not bool(jnp.all(jnp.diff(x_knots) > 0.0)):
        raise ValueError(
            "Invalid knot set: `x_knots` must be strictly increasing for monotone interpolation."
        )


def _diagnose_approximation(*, transform: Any, base: str, distribution: Any) -> dict[str, Any]:
    dtype = jnp.float64
    if base == "uniform":
        base_grid = jnp.linspace(1e-9, 1.0 - 1e-9, 1025, dtype=dtype)
    else:
        base_grid = jnp.linspace(-6.0, 6.0, 1025, dtype=dtype)

    y = jnp.asarray(transform(base_grid), dtype=dtype)
    x_back = jnp.asarray(transform._inverse(y), dtype=dtype)
    err = jnp.abs(x_back - base_grid)

    if base == "uniform":
        interior = (base_grid > 1e-4) & (base_grid < 1.0 - 1e-4)
    else:
        interior = jnp.abs(base_grid) < 4.5

    cdf_values = jnp.asarray(distribution.cdf(y), dtype=dtype)
    return {
        "finite_forward": bool(np.all(np.asarray(jax.device_get(jnp.isfinite(y))))),
        "finite_inverse": bool(np.all(np.asarray(jax.device_get(jnp.isfinite(x_back))))),
        "roundtrip_max_abs_error": float(np.asarray(jax.device_get(jnp.max(err)))),
        "roundtrip_mean_abs_error": float(np.asarray(jax.device_get(jnp.mean(err)))),
        "roundtrip_interior_max_abs_error": float(
            np.asarray(jax.device_get(jnp.max(err[interior])))
        ),
        "monotone_forward": bool(np.all(np.asarray(jax.device_get(jnp.diff(y) > 0.0)))),
        "target_cdf_grid_min": float(np.asarray(jax.device_get(jnp.min(cdf_values)))),
        "target_cdf_grid_max": float(np.asarray(jax.device_get(jnp.max(cdf_values)))),
    }


def _raise_stage_error(*, stage: str, exc: Exception, context: Mapping[str, Any]) -> DistributionTransformBuildError:
    reason = str(exc)
    message = f"{stage} stage failed: {reason}"
    return DistributionTransformBuildError(
        message=message,
        stage=stage,
        context=dict(context),
        failure_reason=reason,
    )


def _normalize_base(base: str) -> Literal["uniform", "normal"]:
    base_norm = str(base).strip().lower()
    if base_norm not in {"uniform", "normal"}:
        raise ValueError("`base` must be one of {'uniform', 'normal'}.")
    return base_norm  # type: ignore[return-value]


def _build_pipeline(
    *,
    base: Literal["uniform", "normal"],
    distribution: Any,
    cfg: DistributionTransformBuildConfig,
) -> DistributionTransformBuildResult:
    stage_context: dict[str, Any] = {"base": base}

    solver_cfg = SolverConfig(
        rtol=cfg.solver_cfg.rtol,
        atol=cfg.solver_cfg.atol,
        max_steps=cfg.solver_cfg.max_steps,
    )

    try:
        backend = build_distribution_quantile_backend(
            distribution=distribution,
            solver_cfg=solver_cfg,
            bracket_cfg=cfg.solver_cfg.bracket_cfg,
            eps=cfg.solver_cfg.eps,
        )
        backend_diagnostics = backend.validate()
    except Exception as exc:
        raise _raise_stage_error(stage="backend", exc=exc, context=stage_context) from exc

    knot_cfg = KnotConfig(
        num_knots=cfg.knot_cfg.num_knots,
        u_min=cfg.knot_cfg.u_min,
        u_max=cfg.knot_cfg.u_max,
        grid_type=cfg.knot_cfg.grid_type,
        tail_density=cfg.knot_cfg.tail_density,
        min_delta_u=cfg.knot_cfg.min_delta_u,
    )
    slope_cfg = SlopeConfig(
        method=cfg.knot_cfg.slope_method,
        delta_u=cfg.knot_cfg.slope_delta_u,
        min_positive_slope=cfg.knot_cfg.min_positive_slope,
    )

    try:
        knot_set = build_quantile_knot_set(
            quantile_backend=backend,
            knot_cfg=knot_cfg,
            slope_cfg=slope_cfg,
        )
        _validate_knot_set_or_raise(knot_set)
    except Exception as exc:
        raise _raise_stage_error(
            stage="knot_generation",
            exc=exc,
            context={**stage_context, "requested_num_knots": cfg.knot_cfg.num_knots},
        ) from exc

    interp_cfg = InterpConfig(
        interior_method=cfg.interp_cfg.interior_method,
        clip_u_eps=cfg.interp_cfg.clip_u_eps,
        safe_arctanh_eps=cfg.interp_cfg.safe_arctanh_eps,
        cdf_eval_method=cfg.interp_cfg.cdf_eval_method,
        cdf_root_max_steps=cfg.interp_cfg.cdf_root_max_steps,
        cdf_root_u_tol=cfg.interp_cfg.cdf_root_u_tol,
    )
    tail_cfg = InterpolatorTailConfig(
        enforce_c1_stitch=cfg.tail_cfg.enforce_c1_stitch,
        min_tail_scale=cfg.tail_cfg.min_tail_scale,
    )

    try:
        raw_interpolator = QuantileInterpolator1D(
            knot_set=knot_set,
            interp_cfg=interp_cfg,
            tail_cfg=tail_cfg,
        )
        interpolator = _ShapeSafeInterpolator(raw_interpolator)
        x_probe = jnp.linspace(knot_set.x_knots[0], knot_set.x_knots[-1], 257)
        u_probe = interpolator.cdf(x_probe)
        if not bool(jnp.all(jnp.isfinite(u_probe)) and jnp.all(jnp.diff(u_probe) >= 0.0)):
            raise ValueError(
                "Interpolator monotonicity/finiteness check failed on diagnostic grid."
            )
    except Exception as exc:
        raise _raise_stage_error(
            stage="interpolator",
            exc=exc,
            context={
                **stage_context,
                "interior_method": cfg.interp_cfg.interior_method,
                "enforce_c1_stitch": cfg.tail_cfg.enforce_c1_stitch,
            },
        ) from exc

    transform_cfg_payload = {
        "clip_u_eps": cfg.transform_cfg.clip_u_eps,
        "validate_args": cfg.transform_cfg.validate_args,
    }
    try:
        if base == "uniform":
            transform = UniformToDistributionTransform(
                interpolator=interpolator,
                transform_cfg=transform_cfg_payload,
            )
        else:
            transform = NormalToDistributionTransform(
                interpolator=interpolator,
                transform_cfg=transform_cfg_payload,
            )
    except Exception as exc:
        raise _raise_stage_error(stage="transform", exc=exc, context=stage_context) from exc

    stitch_provenance = {
        "enforce_c1_stitch": bool(cfg.tail_cfg.enforce_c1_stitch),
        "boundary_slope_source": (
            "interior_boundary_gradient"
            if cfg.tail_cfg.enforce_c1_stitch
            else "knot_endpoint_slope_seed"
        ),
        "endpoint_slope_role": (
            "canonical_c1" if cfg.tail_cfg.enforce_c1_stitch else "fallback_seed"
        ),
        "stitch_points": interpolator.stitch_points(),
    }

    approximation = _diagnose_approximation(
        transform=transform,
        base=base,
        distribution=distribution,
    )
    diagnostics = {
        "config": cfg,
        "backend": backend_diagnostics,
        "knot": dict(knot_set.meta),
        "interpolation": stitch_provenance,
        "approximation": approximation,
    }
    metadata = {
        "base": base,
        "distribution_type": type(distribution).__name__,
    }
    return DistributionTransformBuildResult(
        transform=transform,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def build_distribution_transform(
    *,
    base: str,
    distribution: Any,
    build_cfg: Any = None,
    config: Any = None,
    cfg: Any = None,
    distribution_transform_cfg: Any = None,
    solver_cfg: Any = None,
    knot_cfg: Any = None,
    interp_cfg: Any = None,
    tail_cfg: Any = None,
    transform_cfg: Any = None,
) -> DistributionTransformBuildResult:
    """Build a transform from a scalar base distribution to a 1D target distribution."""

    base_norm = _normalize_base(base)
    try:
        _validate_distribution(distribution)
    except Exception as exc:
        raise _raise_stage_error(
            stage="input_validation",
            exc=exc,
            context={"base": base_norm},
        ) from exc

    effective_cfg = _as_build_cfg(
        build_cfg=build_cfg,
        config=config,
        cfg=cfg,
        distribution_transform_cfg=distribution_transform_cfg,
        solver_cfg=solver_cfg,
        knot_cfg=knot_cfg,
        interp_cfg=interp_cfg,
        tail_cfg=tail_cfg,
        transform_cfg=transform_cfg,
    )
    return _build_pipeline(
        base=base_norm,
        distribution=distribution,
        cfg=effective_cfg,
    )


def build_uniform_to_distribution_transform(**kwargs: Any) -> DistributionTransformBuildResult:
    return build_distribution_transform(base="uniform", **kwargs)


def build_normal_to_distribution_transform(**kwargs: Any) -> DistributionTransformBuildResult:
    return build_distribution_transform(base="normal", **kwargs)


__all__ = [
    "DistributionQuantileConfig",
    "DistributionTransformBuildConfig",
    "DistributionTransformBuildError",
    "DistributionTransformBuildResult",
    "InterpolatorConfig",
    "KnotGenerationConfig",
    "TailConfig",
    "TransformConfig",
    "build_distribution_transform",
    "build_normal_to_distribution_transform",
    "build_uniform_to_distribution_transform",
]
