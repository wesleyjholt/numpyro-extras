"""Factory API for building mixture transforms end-to-end.

The builder assembles the full pipeline:
1. mixture quantile backend,
2. interpolation knot generation,
3. monotone interpolator with tail handling,
4. NumPyro transform wrapper for a selected base distribution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .mixture_knots import QuantileKnotSet, build_quantile_knot_set
from .mixture_quantile import build_mixture_quantile_backend
from .quantile_interpolator import QuantileInterpolator1D
from .transforms import NormalToMixtureTransform, UniformToMixtureTransform

__all__ = [
    "InterpolatorConfig",
    "KnotGenerationConfig",
    "MixtureQuantileConfig",
    "MixtureTransformBuildConfig",
    "MixtureTransformBuildError",
    "TailConfig",
    "TransformConfig",
    "build_mixture_transform",
    "build_normal_to_mixture_transform",
    "build_uniform_to_mixture_transform",
]


_DEFAULT_SOLVER_CFG = {"rtol": 1e-6, "atol": 1e-6, "max_steps": 128}
_DEFAULT_BRACKET_CFG = {
    "x_init_low": -8.0,
    "x_init_high": 8.0,
    "expansion_factor": 2.0,
    "max_expansions": 32,
}
_DEFAULT_KNOT_CFG = {
    "num_knots": 256,
    "u_min": 1e-6,
    "u_max": 1.0 - 1e-6,
    "grid_type": "logit_u",
    "tail_density": 0.35,
    "min_delta_u": 1e-10,
}
_DEFAULT_SLOPE_CFG = {"method": "finite_diff_u", "delta_u": 5e-5, "min_positive_slope": 1e-8}


@dataclass(frozen=True)
class MixtureQuantileConfig:
    solver_cfg: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_SOLVER_CFG))
    bracket_cfg: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_BRACKET_CFG))
    eps: float = 1e-10


@dataclass(frozen=True)
class KnotGenerationConfig:
    knot_cfg: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_KNOT_CFG))
    slope_cfg: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_SLOPE_CFG))


@dataclass(frozen=True)
class InterpolatorConfig:
    interior_method: str = "akima"
    clip_u_eps: float = 1e-10
    safe_arctanh_eps: float = 1e-7


@dataclass(frozen=True)
class TailConfig:
    enforce_c1_stitch: bool = True
    min_tail_scale: float = 1e-8


@dataclass(frozen=True)
class TransformConfig:
    clip_u_eps: float = 1e-10
    validate_args: bool = False


@dataclass(frozen=True)
class MixtureTransformBuildConfig:
    base: str = "normal"
    quantile_cfg: MixtureQuantileConfig = field(default_factory=MixtureQuantileConfig)
    knot_generation_cfg: KnotGenerationConfig = field(default_factory=KnotGenerationConfig)
    interp_cfg: InterpolatorConfig = field(default_factory=InterpolatorConfig)
    tail_cfg: TailConfig = field(default_factory=TailConfig)
    transform_cfg: TransformConfig = field(default_factory=TransformConfig)


class MixtureTransformBuildError(RuntimeError):
    """Error raised when a pipeline stage fails during builder assembly."""

    def __init__(self, message: str, *, stage: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = dict(diagnostics)


def _normalize_base(base: str) -> str:
    base_norm = str(base).strip().lower()
    if base_norm not in {"uniform", "normal"}:
        raise ValueError("`base` must be one of {'uniform', 'normal'}.")
    return base_norm


def _merge_mapping(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = dict(base)
    if override is None:
        return merged
    if not isinstance(override, Mapping):
        raise TypeError("Config overrides must be dict-like mappings.")
    merged.update(dict(override))
    return merged


def _coerce_quantile_cfg(value: Optional[MixtureQuantileConfig | Mapping[str, Any]]) -> MixtureQuantileConfig:
    if value is None:
        return MixtureQuantileConfig()
    if isinstance(value, MixtureQuantileConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("`quantile_cfg` must be a MixtureQuantileConfig or dict-like mapping.")
    solver_cfg = _merge_mapping(_DEFAULT_SOLVER_CFG, value.get("solver_cfg"))
    bracket_cfg = _merge_mapping(_DEFAULT_BRACKET_CFG, value.get("bracket_cfg"))
    eps = float(value.get("eps", 1e-10))
    return MixtureQuantileConfig(solver_cfg=solver_cfg, bracket_cfg=bracket_cfg, eps=eps)


def _coerce_knot_generation_cfg(
    value: Optional[KnotGenerationConfig | Mapping[str, Any]]
) -> KnotGenerationConfig:
    if value is None:
        return KnotGenerationConfig()
    if isinstance(value, KnotGenerationConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("`knot_generation_cfg` must be a KnotGenerationConfig or dict-like mapping.")
    knot_cfg = _merge_mapping(_DEFAULT_KNOT_CFG, value.get("knot_cfg"))
    slope_cfg = _merge_mapping(_DEFAULT_SLOPE_CFG, value.get("slope_cfg"))
    return KnotGenerationConfig(knot_cfg=knot_cfg, slope_cfg=slope_cfg)


def _coerce_interp_cfg(value: Optional[InterpolatorConfig | Mapping[str, Any]]) -> InterpolatorConfig:
    if value is None:
        return InterpolatorConfig()
    if isinstance(value, InterpolatorConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("`interp_cfg` must be an InterpolatorConfig or dict-like mapping.")
    return InterpolatorConfig(
        interior_method=str(value.get("interior_method", "akima")),
        clip_u_eps=float(value.get("clip_u_eps", 1e-10)),
        safe_arctanh_eps=float(value.get("safe_arctanh_eps", 1e-7)),
    )


def _coerce_tail_cfg(value: Optional[TailConfig | Mapping[str, Any]]) -> TailConfig:
    if value is None:
        return TailConfig()
    if isinstance(value, TailConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("`tail_cfg` must be a TailConfig or dict-like mapping.")
    return TailConfig(
        enforce_c1_stitch=bool(value.get("enforce_c1_stitch", True)),
        min_tail_scale=float(value.get("min_tail_scale", 1e-8)),
    )


def _coerce_transform_cfg(value: Optional[TransformConfig | Mapping[str, Any]]) -> TransformConfig:
    if value is None:
        return TransformConfig()
    if isinstance(value, TransformConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("`transform_cfg` must be a TransformConfig or dict-like mapping.")
    return TransformConfig(
        clip_u_eps=float(value.get("clip_u_eps", 1e-10)),
        validate_args=bool(value.get("validate_args", False)),
    )


def _coerce_build_config(
    *,
    base: str,
    build_config: Optional[MixtureTransformBuildConfig | Mapping[str, Any]],
    solver_cfg: Optional[Mapping[str, Any]],
    bracket_cfg: Optional[Mapping[str, Any]],
    knot_cfg: Optional[Mapping[str, Any]],
    slope_cfg: Optional[Mapping[str, Any]],
    interp_cfg: Optional[Mapping[str, Any]],
    tail_cfg: Optional[Mapping[str, Any]],
    transform_cfg: Optional[Mapping[str, Any]],
) -> MixtureTransformBuildConfig:
    if build_config is None:
        cfg = MixtureTransformBuildConfig(base=_normalize_base(base))
    elif isinstance(build_config, MixtureTransformBuildConfig):
        cfg = build_config
    elif isinstance(build_config, Mapping):
        quantile_payload = build_config.get("quantile_cfg")
        if quantile_payload is None:
            quantile_payload = {}
            if "solver_cfg" in build_config:
                quantile_payload["solver_cfg"] = build_config["solver_cfg"]
            if "bracket_cfg" in build_config:
                quantile_payload["bracket_cfg"] = build_config["bracket_cfg"]
            if "eps" in build_config:
                quantile_payload["eps"] = build_config["eps"]

        knot_payload = build_config.get("knot_generation_cfg")
        if knot_payload is None:
            knot_payload = {}
            if "knot_cfg" in build_config:
                knot_payload["knot_cfg"] = build_config["knot_cfg"]
            if "slope_cfg" in build_config:
                knot_payload["slope_cfg"] = build_config["slope_cfg"]

        cfg = MixtureTransformBuildConfig(
            base=_normalize_base(str(build_config.get("base", base))),
            quantile_cfg=_coerce_quantile_cfg(quantile_payload),
            knot_generation_cfg=_coerce_knot_generation_cfg(knot_payload),
            interp_cfg=_coerce_interp_cfg(build_config.get("interp_cfg")),
            tail_cfg=_coerce_tail_cfg(build_config.get("tail_cfg")),
            transform_cfg=_coerce_transform_cfg(build_config.get("transform_cfg")),
        )
    else:
        raise TypeError("`build_config` must be None, MixtureTransformBuildConfig, or dict-like.")

    quantile_cfg = MixtureQuantileConfig(
        solver_cfg=_merge_mapping(cfg.quantile_cfg.solver_cfg, solver_cfg),
        bracket_cfg=_merge_mapping(cfg.quantile_cfg.bracket_cfg, bracket_cfg),
        eps=float(cfg.quantile_cfg.eps),
    )
    knot_generation_cfg = KnotGenerationConfig(
        knot_cfg=_merge_mapping(cfg.knot_generation_cfg.knot_cfg, knot_cfg),
        slope_cfg=_merge_mapping(cfg.knot_generation_cfg.slope_cfg, slope_cfg),
    )
    interp_cfg_resolved = _coerce_interp_cfg(
        _merge_mapping(dataclasses.asdict(cfg.interp_cfg), interp_cfg)
    )
    tail_cfg_resolved = _coerce_tail_cfg(
        _merge_mapping(dataclasses.asdict(cfg.tail_cfg), tail_cfg)
    )
    transform_cfg_resolved = _coerce_transform_cfg(
        _merge_mapping(dataclasses.asdict(cfg.transform_cfg), transform_cfg)
    )

    return MixtureTransformBuildConfig(
        base=_normalize_base(base),
        quantile_cfg=quantile_cfg,
        knot_generation_cfg=knot_generation_cfg,
        interp_cfg=interp_cfg_resolved,
        tail_cfg=tail_cfg_resolved,
        transform_cfg=transform_cfg_resolved,
    )


def _build_config_snapshot(cfg: MixtureTransformBuildConfig) -> dict[str, Any]:
    return {
        "base": cfg.base,
        "solver_cfg": dict(cfg.quantile_cfg.solver_cfg),
        "bracket_cfg": dict(cfg.quantile_cfg.bracket_cfg),
        "knot_cfg": dict(cfg.knot_generation_cfg.knot_cfg),
        "slope_cfg": dict(cfg.knot_generation_cfg.slope_cfg),
        "interp_cfg": dataclasses.asdict(cfg.interp_cfg),
        "tail_cfg": dataclasses.asdict(cfg.tail_cfg),
        "transform_cfg": dataclasses.asdict(cfg.transform_cfg),
        "eps": float(cfg.quantile_cfg.eps),
    }


def _validate_weights(weights: Any) -> jax.Array:
    weights_arr = jnp.asarray(weights)
    if weights_arr.ndim != 1:
        raise ValueError("Invalid weights: expected rank-1 `weights` with shape [K].")

    weights_np = np.asarray(weights_arr, dtype=float)
    if not np.all(np.isfinite(weights_np)):
        raise ValueError("Invalid weights: all weights must be finite.")
    if np.any(weights_np < 0.0):
        raise ValueError("Invalid weights: all weights must be non-negative.")

    weight_sum = float(np.sum(weights_np))
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Invalid weights: weights must sum to 1 (normalize first); got sum={weight_sum:.8f}."
        )
    return weights_arr


def _extract_mixture_weights(mixture_obj: Any) -> jax.Array:
    probs = getattr(mixture_obj, "probs", None)
    if probs is not None:
        return jnp.asarray(probs)
    logits = getattr(mixture_obj, "logits", None)
    if logits is not None:
        return jax.nn.softmax(jnp.asarray(logits), axis=-1)
    raise TypeError(
        "Mixture distribution must expose either `probs` or `logits` on its mixing distribution."
    )


def _resolve_component_distribution(
    *,
    component_distribution: Any,
    component_dist: Any,
    component_family: Any,
    component_class: Any,
    component_params: Optional[Mapping[str, Any]],
    component_param_overrides: Mapping[str, Any],
) -> Any:
    if component_distribution is not None and component_dist is not None:
        if component_distribution is not component_dist:
            raise ValueError(
                "Provide only one of `component_distribution` or `component_dist`."
            )
    resolved = component_distribution if component_distribution is not None else component_dist
    if resolved is not None:
        return resolved

    if component_family is not None and component_class is not None:
        if component_family is not component_class:
            raise ValueError("Provide only one of `component_family` or `component_class`.")
    family = component_family if component_family is not None else component_class
    if family is None:
        raise ValueError(
            "Explicit input mode requires `component_distribution`/`component_dist` or "
            "`component_family`/`component_class`."
        )

    params: dict[str, Any] = {}
    if component_params is not None:
        if not isinstance(component_params, Mapping):
            raise TypeError("`component_params` must be dict-like when provided.")
        params.update(dict(component_params))
    params.update(dict(component_param_overrides))
    if not params:
        raise ValueError(
            "No component parameters provided. Supply `component_params` or keyword parameters "
            "such as `loc=` and `scale=`."
        )

    try:
        return family(**params)
    except Exception as exc:
        raise ValueError(
            "Invalid component parameter shape/value mismatch while constructing "
            "`component_distribution`."
        ) from exc


def _validate_component_shape(weights: jax.Array, component_distribution: Any) -> None:
    batch_shape = tuple(getattr(component_distribution, "batch_shape", ()))
    n_components = int(weights.shape[0])
    if not batch_shape:
        if n_components != 1:
            raise ValueError(
                "Invalid component shape mismatch: scalar component distribution requires "
                "`weights` with exactly one component."
            )
        return
    last_dim = int(batch_shape[-1])
    if last_dim != n_components:
        raise ValueError(
            "Invalid component shape mismatch: last component batch dimension must match "
            f"number of weights (expected {n_components}, got {batch_shape})."
        )


def _resolve_inputs(
    *,
    mixture_distribution: Any,
    mixture: Any,
    distribution: Any,
    weights: Any,
    component_distribution: Any,
    component_dist: Any,
    component_family: Any,
    component_class: Any,
    component_params: Optional[Mapping[str, Any]],
    component_param_overrides: Mapping[str, Any],
) -> tuple[jax.Array, Any, str]:
    mixture_candidates = [
        candidate for candidate in (mixture_distribution, mixture, distribution) if candidate is not None
    ]
    if len(mixture_candidates) > 1:
        raise ValueError(
            "Provide exactly one of `mixture_distribution`, `mixture`, or `distribution`."
        )

    if mixture_candidates:
        mixture_obj = mixture_candidates[0]
        mixing = getattr(mixture_obj, "mixture_distribution", None)
        if mixing is None:
            mixing = getattr(mixture_obj, "mixing_distribution", None)
        component = getattr(mixture_obj, "component_distribution", None)
        if mixing is None or component is None:
            raise TypeError(
                "Mixture input must expose mixing and component distribution attributes "
                "(`mixture_distribution`/`mixing_distribution` plus `component_distribution`)."
            )
        if weights is not None:
            raise ValueError("Do not pass `weights` when a full mixture distribution is provided.")
        weights_arr = _validate_weights(_extract_mixture_weights(mixing))
        _validate_component_shape(weights_arr, component)
        return weights_arr, component, "mixture_distribution"

    if weights is None:
        raise ValueError(
            "No mixture input provided. Supply either a mixture distribution object or explicit "
            "`weights` with component specification."
        )

    weights_arr = _validate_weights(weights)
    component = _resolve_component_distribution(
        component_distribution=component_distribution,
        component_dist=component_dist,
        component_family=component_family,
        component_class=component_class,
        component_params=component_params,
        component_param_overrides=component_param_overrides,
    )
    _validate_component_shape(weights_arr, component)
    return weights_arr, component, "explicit_parameters"


def _summarize_approximation(
    *,
    knot_set: QuantileKnotSet,
    interpolator: QuantileInterpolator1D,
) -> dict[str, Any]:
    u_knots = jnp.asarray(knot_set.u_knots)
    x_knots = jnp.asarray(knot_set.x_knots)
    recovered_u = interpolator.cdf(x_knots)
    recovered_x = interpolator.icdf(u_knots)

    u_abs_err = np.abs(np.asarray(recovered_u - u_knots))
    x_abs_err = np.abs(np.asarray(recovered_x - x_knots))
    return {
        "u_roundtrip_max_abs_error": float(np.max(u_abs_err)),
        "u_roundtrip_mean_abs_error": float(np.mean(u_abs_err)),
        "x_roundtrip_max_abs_error": float(np.max(x_abs_err)),
        "x_roundtrip_mean_abs_error": float(np.mean(x_abs_err)),
        "num_knots": int(u_knots.shape[0]),
    }


def _raise_stage_error(stage: str, exc: Exception, diagnostics: Mapping[str, Any]) -> None:
    payload = dict(diagnostics)
    payload.update(
        {
            "stage": stage,
            "reason": type(exc).__name__,
            "message": str(exc),
            "context": "pipeline",
        }
    )
    raise MixtureTransformBuildError(
        f"Mixture transform build pipeline failed at `{stage}`: {exc}",
        stage=stage,
        diagnostics=payload,
    ) from exc


def build_mixture_transform(
    *,
    base: str = "normal",
    mixture_distribution: Any = None,
    mixture: Any = None,
    distribution: Any = None,
    weights: Any = None,
    component_distribution: Any = None,
    component_dist: Any = None,
    component_family: Any = None,
    component_class: Any = None,
    component_params: Optional[Mapping[str, Any]] = None,
    build_config: Optional[MixtureTransformBuildConfig | Mapping[str, Any]] = None,
    solver_cfg: Optional[Mapping[str, Any]] = None,
    bracket_cfg: Optional[Mapping[str, Any]] = None,
    knot_cfg: Optional[Mapping[str, Any]] = None,
    slope_cfg: Optional[Mapping[str, Any]] = None,
    interp_cfg: Optional[Mapping[str, Any]] = None,
    tail_cfg: Optional[Mapping[str, Any]] = None,
    transform_cfg: Optional[Mapping[str, Any]] = None,
    **component_param_overrides: Any,
) -> dict[str, Any]:
    """Build a mixture transform from either a mixture object or explicit parameters."""

    resolved_cfg = _coerce_build_config(
        base=base,
        build_config=build_config,
        solver_cfg=solver_cfg,
        bracket_cfg=bracket_cfg,
        knot_cfg=knot_cfg,
        slope_cfg=slope_cfg,
        interp_cfg=interp_cfg,
        tail_cfg=tail_cfg,
        transform_cfg=transform_cfg,
    )
    config_snapshot = _build_config_snapshot(resolved_cfg)
    diagnostics: dict[str, Any] = {"config": config_snapshot}

    try:
        weights_arr, component_dist_obj, input_mode = _resolve_inputs(
            mixture_distribution=mixture_distribution,
            mixture=mixture,
            distribution=distribution,
            weights=weights,
            component_distribution=component_distribution,
            component_dist=component_dist,
            component_family=component_family,
            component_class=component_class,
            component_params=component_params,
            component_param_overrides=component_param_overrides,
        )
    except Exception as exc:
        _raise_stage_error("input_validation", exc, diagnostics)

    diagnostics["input_mode"] = input_mode
    diagnostics["component_count"] = int(weights_arr.shape[0])
    diagnostics["component_distribution"] = component_dist_obj.__class__.__name__

    try:
        quantile_backend = build_mixture_quantile_backend(
            weights=weights_arr,
            component_distribution=component_dist_obj,
            solver_cfg=config_snapshot["solver_cfg"],
            bracket_cfg=config_snapshot["bracket_cfg"],
            eps=config_snapshot["eps"],
        )
    except Exception as exc:
        _raise_stage_error("quantile_backend", exc, diagnostics)

    try:
        knot_set = build_quantile_knot_set(
            quantile_backend=quantile_backend,
            knot_cfg=config_snapshot["knot_cfg"],
            slope_cfg=config_snapshot["slope_cfg"],
        )
    except Exception as exc:
        _raise_stage_error("knot_generation", exc, diagnostics)

    diagnostics["knot_meta"] = dict(knot_set.meta)

    try:
        interpolator = QuantileInterpolator1D(
            knot_set,
            interp_cfg=config_snapshot["interp_cfg"],
            tail_cfg=config_snapshot["tail_cfg"],
        )
    except Exception as exc:
        _raise_stage_error("interpolator", exc, diagnostics)

    diagnostics["stitch_points"] = {
        key: float(np.asarray(value))
        for key, value in interpolator.stitch_points().items()
    }

    try:
        if resolved_cfg.base == "uniform":
            transform = UniformToMixtureTransform(
                interpolator=interpolator,
                transform_cfg=config_snapshot["transform_cfg"],
            )
        else:
            transform = NormalToMixtureTransform(
                interpolator=interpolator,
                transform_cfg=config_snapshot["transform_cfg"],
            )
    except Exception as exc:
        _raise_stage_error("transform_wrapper", exc, diagnostics)

    diagnostics["approx_summary"] = _summarize_approximation(
        knot_set=knot_set,
        interpolator=interpolator,
    )

    for attr_name, value in (
        ("diagnostics", diagnostics),
        ("config", config_snapshot),
        ("knot_meta", diagnostics["knot_meta"]),
        ("approx_summary", diagnostics["approx_summary"]),
    ):
        try:
            setattr(transform, attr_name, value)
        except Exception:
            pass

    return {
        "transform": transform,
        "diagnostics": diagnostics,
        "config": config_snapshot,
        "quantile_backend": quantile_backend,
        "knot_set": knot_set,
        "interpolator": interpolator,
    }


def build_uniform_to_mixture_transform(**kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for `build_mixture_transform(base="uniform", ...)`."""

    return build_mixture_transform(base="uniform", **kwargs)


def build_normal_to_mixture_transform(**kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper for `build_mixture_transform(base="normal", ...)`."""

    return build_mixture_transform(base="normal", **kwargs)
