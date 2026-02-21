"""Mixture inverse-CDF backend for 1D continuous mixtures.

This module provides a reusable numerical backend for:
1. weighted mixture CDF evaluation,
2. weighted mixture log-density evaluation, and
3. inverse-CDF evaluation via robust bracketed bisection.

Assumptions and limitations:
- Mixture weights are validated on construction and must sum to 1.
- Component distribution must expose ``cdf`` and ``log_prob``.
- ``icdf`` uses deterministic bracketing + bisection and clamps ``u`` to
  ``[eps, 1 - eps]`` (default ``eps=1e-10``) for tail stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from jax import core as jax_core
import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = [
    "BracketConfig",
    "MixtureQuantileBackend",
    "SolverConfig",
    "build_mixture_quantile_backend",
]


@dataclass(frozen=True)
class SolverConfig:
    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 128


@dataclass(frozen=True)
class BracketConfig:
    x_init_low: float = -8.0
    x_init_high: float = 8.0
    expansion_factor: float = 2.0
    max_expansions: int = 32


def _coerce_solver_cfg(solver_cfg: Optional[Mapping[str, Any] | SolverConfig]) -> SolverConfig:
    if solver_cfg is None:
        return SolverConfig()
    if isinstance(solver_cfg, SolverConfig):
        return solver_cfg
    if isinstance(solver_cfg, Mapping):
        return SolverConfig(
            rtol=float(solver_cfg.get("rtol", 1e-6)),
            atol=float(solver_cfg.get("atol", 1e-6)),
            max_steps=int(solver_cfg.get("max_steps", 128)),
        )
    raise TypeError("`solver_cfg` must be None, dict-like, or `SolverConfig`.")


def _coerce_bracket_cfg(bracket_cfg: Optional[Mapping[str, Any] | BracketConfig]) -> BracketConfig:
    if bracket_cfg is None:
        return BracketConfig()
    if isinstance(bracket_cfg, BracketConfig):
        return bracket_cfg
    if isinstance(bracket_cfg, Mapping):
        return BracketConfig(
            x_init_low=float(bracket_cfg.get("x_init_low", -8.0)),
            x_init_high=float(bracket_cfg.get("x_init_high", 8.0)),
            expansion_factor=float(bracket_cfg.get("expansion_factor", 2.0)),
            max_expansions=int(bracket_cfg.get("max_expansions", 32)),
        )
    raise TypeError("`bracket_cfg` must be None, dict-like, or `BracketConfig`.")


def _validate_weights(weights: jax.Array) -> None:
    if weights.ndim != 1:
        raise ValueError("`weights` must be a rank-1 array of shape [K].")

    weights_np = np.asarray(weights)
    if not np.all(np.isfinite(weights_np)):
        raise ValueError("`weights` must be finite.")
    if np.any(weights_np < 0):
        raise ValueError("`weights` must be non-negative.")

    weight_sum = float(weights_np.sum())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"`weights` must sum to 1 within atol=1e-6, got sum={weight_sum:.8f}."
        )


def _ensure_distribution_contract(component_distribution: Any) -> None:
    for name in ("cdf", "log_prob"):
        method = getattr(component_distribution, name, None)
        if method is None or not callable(method):
            raise TypeError(
                "`component_distribution` must define callable `cdf(x)` and `log_prob(x)`."
            )


@dataclass(frozen=True)
class MixtureQuantileBackend:
    weights: jax.Array
    component_distribution: Any
    solver_cfg: SolverConfig
    bracket_cfg: BracketConfig
    eps: float = 1e-10
    passthrough_single_component: bool = False

    def _expand_x(self, x: jax.Array) -> jax.Array:
        return jnp.expand_dims(x, axis=-1)

    def cdf(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x)
        component_cdf = self.component_distribution.cdf(self._expand_x(x_arr))
        mixed = jnp.sum(self.weights * component_cdf, axis=-1)
        return mixed.astype(x_arr.dtype)

    def log_prob(self, x: jax.Array) -> jax.Array:
        x_arr = jnp.asarray(x)
        log_w = jnp.where(self.weights > 0.0, jnp.log(self.weights), -jnp.inf)
        component_lp = self.component_distribution.log_prob(self._expand_x(x_arr))
        mixed = logsumexp(log_w + component_lp, axis=-1)
        return mixed.astype(x_arr.dtype)

    def _cdf_scalar(self, x_scalar: jax.Array) -> jax.Array:
        return self.cdf(x_scalar)

    def _discover_bracket(self, u_scalar: jax.Array) -> Tuple[jax.Array, ...]:
        dtype = u_scalar.dtype
        factor = jnp.asarray(self.bracket_cfg.expansion_factor, dtype=dtype)
        max_expansions = int(self.bracket_cfg.max_expansions)

        low0 = jnp.asarray(self.bracket_cfg.x_init_low, dtype=dtype)
        high0 = jnp.asarray(self.bracket_cfg.x_init_high, dtype=dtype)
        g_low0 = self._cdf_scalar(low0) - u_scalar
        g_high0 = self._cdf_scalar(high0) - u_scalar
        bracketed0 = jnp.logical_and(g_low0 <= 0.0, g_high0 >= 0.0)

        def cond_fn(state: Tuple[jax.Array, ...]) -> jax.Array:
            _, _, _, _, i, bracketed = state
            return jnp.logical_and(~bracketed, i < max_expansions)

        def body_fn(state: Tuple[jax.Array, ...]) -> Tuple[jax.Array, ...]:
            low, high, _, _, i, _ = state
            low = jnp.where(low <= 0.0, low * factor, -low * factor)
            high = jnp.where(high >= 0.0, high * factor, -high * factor)
            g_low = self._cdf_scalar(low) - u_scalar
            g_high = self._cdf_scalar(high) - u_scalar
            bracketed = jnp.logical_and(g_low <= 0.0, g_high >= 0.0)
            return low, high, g_low, g_high, i + 1, bracketed

        init = (
            low0,
            high0,
            g_low0,
            g_high0,
            jnp.asarray(0, dtype=jnp.int32),
            bracketed0,
        )
        return lax.while_loop(cond_fn, body_fn, init)

    def _bisection_converged(
        self, low: jax.Array, high: jax.Array, g_mid: jax.Array, mid: jax.Array, u: jax.Array
    ) -> jax.Array:
        atol = jnp.asarray(self.solver_cfg.atol, dtype=mid.dtype)
        rtol = jnp.asarray(self.solver_cfg.rtol, dtype=mid.dtype)
        x_tol = atol + rtol * jnp.maximum(jnp.abs(mid), 1.0)
        del g_mid, u
        # Width-based stopping is robust in tails where CDF slope is very small.
        return (high - low) <= x_tol

    def _solve_scalar(self, u_scalar: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        low, high, _, _, _, bracketed = self._discover_bracket(u_scalar)
        max_steps = int(self.solver_cfg.max_steps)

        def solve_fn(_: None) -> Tuple[jax.Array, jax.Array, jax.Array]:
            mid0 = 0.5 * (low + high)
            g_mid0 = self._cdf_scalar(mid0) - u_scalar
            conv0 = self._bisection_converged(low, high, g_mid0, mid0, u_scalar)
            init = (
                low,
                high,
                mid0,
                g_mid0,
                jnp.asarray(0, dtype=jnp.int32),
                conv0,
            )

            def cond_fn(state: Tuple[jax.Array, ...]) -> jax.Array:
                _, _, _, _, step, converged = state
                return jnp.logical_and(~converged, step < max_steps)

            def body_fn(state: Tuple[jax.Array, ...]) -> Tuple[jax.Array, ...]:
                low_i, high_i, mid_i, g_mid_i, step_i, _ = state
                root_is_left = g_mid_i > 0.0
                low_next = jnp.where(root_is_left, low_i, mid_i)
                high_next = jnp.where(root_is_left, mid_i, high_i)
                mid_next = 0.5 * (low_next + high_next)
                g_mid_next = self._cdf_scalar(mid_next) - u_scalar
                conv_next = self._bisection_converged(
                    low_next, high_next, g_mid_next, mid_next, u_scalar
                )
                return low_next, high_next, mid_next, g_mid_next, step_i + 1, conv_next

            _, _, mid, _, steps, converged = lax.while_loop(cond_fn, body_fn, init)
            return mid, converged, jnp.maximum(steps, 1)

        def fail_fn(_: None) -> Tuple[jax.Array, jax.Array, jax.Array]:
            x_fail = jnp.asarray(jnp.nan, dtype=u_scalar.dtype)
            return x_fail, jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32)

        x, converged, n_steps = lax.cond(bracketed, solve_fn, fail_fn, operand=None)
        return x, converged, n_steps, bracketed

    def _icdf_impl(self, u: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        u_arr = jnp.asarray(u)
        dtype_eps = jnp.finfo(u_arr.dtype).eps
        clip_eps = jnp.maximum(jnp.asarray(self.eps, dtype=u_arr.dtype), dtype_eps)
        u_clipped = jnp.clip(u_arr, clip_eps, 1.0 - clip_eps)
        flat_u = jnp.reshape(u_clipped, (-1,))
        x_flat, converged_flat, steps_flat, bracketed_flat = jax.vmap(self._solve_scalar)(flat_u)
        out_shape = u_clipped.shape
        return (
            jnp.reshape(x_flat, out_shape).astype(u_arr.dtype),
            jnp.reshape(converged_flat, out_shape),
            jnp.reshape(steps_flat, out_shape),
            jnp.reshape(bracketed_flat, out_shape),
        )

    def _raise_if_needed(self, failed_any: jax.Array) -> None:
        if isinstance(failed_any, jax_core.Tracer):
            return
        if bool(np.asarray(failed_any)):
            raise ValueError(
                "Inverse-CDF root-finding failed due to bracketing or convergence limits."
            )

    def icdf(self, u: jax.Array) -> jax.Array:
        if self.passthrough_single_component:
            u_arr = jnp.asarray(u)
            return self.component_distribution.icdf(u_arr).astype(u_arr.dtype)

        x, converged, _, bracketed = self._icdf_impl(u)
        failed_any = jnp.any(~jnp.logical_and(converged, bracketed))
        self._raise_if_needed(failed_any)
        return x

    def icdf_with_status(self, u: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        if self.passthrough_single_component:
            x = self.icdf(u)
            converged = jnp.isfinite(x)
            n_steps = jnp.ones_like(x, dtype=jnp.int32)
            return x, converged, n_steps

        x, converged, n_steps, bracketed = self._icdf_impl(u)
        return x, jnp.logical_and(converged, bracketed), n_steps

    def validate(self) -> dict[str, Any]:
        u = jnp.linspace(1e-4, 1.0 - 1e-4, 64, dtype=self.weights.dtype)
        x, converged, _, bracketed = self._icdf_impl(u)
        recovered = self.cdf(x)
        max_err = float(jnp.max(jnp.abs(recovered - u)))
        monotone = bool(np.asarray(jnp.all(jnp.diff(x) >= -1e-7)))
        all_converged = bool(np.asarray(jnp.all(jnp.logical_and(converged, bracketed))))
        return {
            "max_roundtrip_error": max_err,
            "monotone": monotone,
            "all_converged": all_converged,
            "eps": self.eps,
        }


def build_mixture_quantile_backend(
    *,
    weights: jax.Array,
    component_distribution: Any,
    solver_cfg: Optional[Mapping[str, Any] | SolverConfig] = None,
    bracket_cfg: Optional[Mapping[str, Any] | BracketConfig] = None,
    eps: float = 1e-10,
) -> MixtureQuantileBackend:
    weights_arr = jnp.asarray(weights)
    _validate_weights(weights_arr)
    _ensure_distribution_contract(component_distribution)

    solver_cfg_resolved = _coerce_solver_cfg(solver_cfg)
    bracket_cfg_resolved = _coerce_bracket_cfg(bracket_cfg)
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("`eps` must satisfy 0 < eps < 0.5.")

    return MixtureQuantileBackend(
        weights=weights_arr,
        component_distribution=component_distribution,
        solver_cfg=solver_cfg_resolved,
        bracket_cfg=bracket_cfg_resolved,
        eps=float(eps),
        passthrough_single_component=bool(
            weights_arr.shape[0] == 1 and np.isclose(float(np.asarray(weights_arr[0])), 1.0, atol=1e-6)
        ),
    )
