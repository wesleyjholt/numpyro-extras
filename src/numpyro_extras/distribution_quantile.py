"""Inverse-CDF backend for 1D continuous target distributions.

Assumptions:
- The target is scalar-valued and exposes callable `cdf(x)` and `log_prob(x)`.
- Inputs are finite; `u` values are clamped to `[eps, 1 - eps]` internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.core as jcore
import jax.numpy as jnp
import numpy as np


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


@dataclass(frozen=True)
class DistributionQuantileBackend:
    distribution: Any
    solver_cfg: SolverConfig
    bracket_cfg: BracketConfig
    eps: float

    def cdf(self, x: Any) -> jax.Array:
        x_arr = jnp.asarray(x)
        return jnp.asarray(self.distribution.cdf(x_arr))

    def log_prob(self, x: Any) -> jax.Array:
        x_arr = jnp.asarray(x)
        return jnp.asarray(self.distribution.log_prob(x_arr))

    def _solve_one(self, u_scalar: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        cdf_probe = self.cdf(jnp.asarray(0.0))
        dtype = jnp.asarray(cdf_probe).dtype
        u_clipped = jnp.clip(jnp.asarray(u_scalar, dtype=dtype), self.eps, 1.0 - self.eps)

        init_low = jnp.asarray(self.bracket_cfg.x_init_low, dtype=dtype)
        init_high = jnp.asarray(self.bracket_cfg.x_init_high, dtype=dtype)
        expansion = jnp.asarray(self.bracket_cfg.expansion_factor, dtype=dtype)
        max_expansions = jnp.asarray(self.bracket_cfg.max_expansions, dtype=jnp.int32)
        max_steps = jnp.asarray(self.solver_cfg.max_steps, dtype=jnp.int32)
        atol = jnp.asarray(self.solver_cfg.atol, dtype=dtype)
        rtol = jnp.asarray(self.solver_cfg.rtol, dtype=dtype)

        g_low0 = self.cdf(init_low) - u_clipped
        g_high0 = self.cdf(init_high) - u_clipped
        bracketed0 = (g_low0 <= 0.0) & (g_high0 >= 0.0)

        def _expand_cond(state: tuple[jax.Array, ...]) -> jax.Array:
            _, _, _, _, i, bracketed = state
            return (~bracketed) & (i < max_expansions)

        def _expand_body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            low, high, _, _, i, _ = state
            low_next = jnp.where(low < 0.0, low * expansion, low - (expansion - 1.0))
            high_next = jnp.where(
                high > 0.0, high * expansion, high + (expansion - 1.0)
            )
            g_low_next = self.cdf(low_next) - u_clipped
            g_high_next = self.cdf(high_next) - u_clipped
            bracketed_next = (g_low_next <= 0.0) & (g_high_next >= 0.0)
            return (
                low_next,
                high_next,
                g_low_next,
                g_high_next,
                i + jnp.asarray(1, dtype=jnp.int32),
                bracketed_next,
            )

        low, high, g_low, g_high, expansion_steps, bracketed = jax.lax.while_loop(
            _expand_cond,
            _expand_body,
            (
                init_low,
                init_high,
                g_low0,
                g_high0,
                jnp.asarray(0, dtype=jnp.int32),
                bracketed0,
            ),
        )

        def _bisection(payload: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array, jax.Array]:
            low0, high0, g_low0, g_high0, expansion_steps0 = payload

            def _bisect_cond(state: tuple[jax.Array, ...]) -> jax.Array:
                low_i, high_i, _, _, i = state
                width = high_i - low_i
                tol = atol + rtol * jnp.maximum(jnp.abs(low_i), jnp.abs(high_i))
                return (i < max_steps) & (width > tol)

            def _bisect_body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
                low_i, high_i, g_low_i, g_high_i, i = state
                mid = 0.5 * (low_i + high_i)
                g_mid = self.cdf(mid) - u_clipped
                choose_right = g_mid <= 0.0
                low_next = jnp.where(choose_right, mid, low_i)
                high_next = jnp.where(choose_right, high_i, mid)
                g_low_next = jnp.where(choose_right, g_mid, g_low_i)
                g_high_next = jnp.where(choose_right, g_high_i, g_mid)
                return (
                    low_next,
                    high_next,
                    g_low_next,
                    g_high_next,
                    i + jnp.asarray(1, dtype=jnp.int32),
                )

            low_f, high_f, _, _, bisection_steps = jax.lax.while_loop(
                _bisect_cond,
                _bisect_body,
                (
                    low0,
                    high0,
                    g_low0,
                    g_high0,
                    jnp.asarray(0, dtype=jnp.int32),
                ),
            )
            x = 0.5 * (low_f + high_f)
            width = high_f - low_f
            tol = atol + rtol * jnp.maximum(jnp.abs(low_f), jnp.abs(high_f))
            residual = jnp.abs(self.cdf(x) - u_clipped)
            converged = (width <= tol) | (residual <= atol)
            return x, converged, expansion_steps0 + bisection_steps

        def _no_bisection(payload: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array, jax.Array]:
            low0, high0, _, _, expansion_steps0 = payload
            x_mid = 0.5 * (low0 + high0)
            return jnp.asarray(x_mid, dtype=dtype), jnp.asarray(False), expansion_steps0

        return jax.lax.cond(
            bracketed,
            _bisection,
            _no_bisection,
            (low, high, g_low, g_high, expansion_steps),
        )

    def icdf_with_status(self, u: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
        self._validate_u_if_concrete(u)
        cdf_probe = self.cdf(jnp.asarray(0.0))
        dtype = jnp.asarray(cdf_probe).dtype
        u_arr = jnp.asarray(u, dtype=dtype)
        flat_u = u_arr.reshape(-1)
        x_flat, converged_flat, n_steps_flat = jax.vmap(self._solve_one)(flat_u)
        return (
            x_flat.reshape(u_arr.shape),
            converged_flat.reshape(u_arr.shape),
            n_steps_flat.reshape(u_arr.shape),
        )

    def icdf(self, u: Any) -> jax.Array:
        x, converged, _ = self.icdf_with_status(u)
        if not _contains_tracer(converged):
            converged_np = np.asarray(jax.device_get(converged), dtype=bool)
            if not converged_np.all():
                raise ValueError(
                    "Failed to bracket root or converge bisection in distribution icdf."
                )
        return x

    def validate(self) -> dict[str, Any]:
        dtype = jnp.asarray(self.cdf(jnp.asarray(0.0))).dtype
        u = jnp.linspace(self.eps, 1.0 - self.eps, 257, dtype=dtype)
        x, converged, n_steps = self.icdf_with_status(u)
        roundtrip_err = jnp.max(jnp.abs(self.cdf(x) - u))
        min_dx = jnp.min(jnp.diff(x))
        return {
            "all_converged": bool(np.all(np.asarray(jax.device_get(converged)))),
            "max_roundtrip_error": float(np.asarray(jax.device_get(roundtrip_err))),
            "min_delta_x": float(np.asarray(jax.device_get(min_dx))),
            "max_steps": int(np.asarray(jax.device_get(jnp.max(n_steps)))),
        }

    def _validate_u_if_concrete(self, u: Any) -> None:
        if _contains_tracer(u):
            return
        u_np = np.asarray(jax.device_get(jnp.asarray(u)))
        if not np.all(np.isfinite(u_np)):
            raise ValueError("`u` must contain only finite values.")


def _contains_tracer(x: Any) -> bool:
    leaves = jax.tree_util.tree_leaves(x)
    return any(isinstance(leaf, jcore.Tracer) for leaf in leaves)


def _as_solver_cfg(solver_cfg: SolverConfig | Mapping[str, Any] | None) -> SolverConfig:
    if solver_cfg is None:
        return SolverConfig()
    if isinstance(solver_cfg, SolverConfig):
        return solver_cfg
    if isinstance(solver_cfg, Mapping):
        return SolverConfig(**dict(solver_cfg))
    raise TypeError("`solver_cfg` must be a dict-like mapping or SolverConfig.")


def _as_bracket_cfg(
    bracket_cfg: BracketConfig | Mapping[str, Any] | None,
) -> BracketConfig:
    if bracket_cfg is None:
        return BracketConfig()
    if isinstance(bracket_cfg, BracketConfig):
        return bracket_cfg
    if isinstance(bracket_cfg, Mapping):
        return BracketConfig(**dict(bracket_cfg))
    raise TypeError("`bracket_cfg` must be a dict-like mapping or BracketConfig.")


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
            "`distribution` must support scalar `cdf` and `log_prob` evaluation."
        ) from exc


def _validate_solver_cfg(solver_cfg: SolverConfig) -> None:
    if solver_cfg.rtol < 0.0:
        raise ValueError("`solver_cfg.rtol` must be >= 0.")
    if solver_cfg.atol <= 0.0:
        raise ValueError("`solver_cfg.atol` must be > 0.")
    if solver_cfg.max_steps <= 0:
        raise ValueError("`solver_cfg.max_steps` must be > 0.")


def _validate_bracket_cfg(bracket_cfg: BracketConfig) -> None:
    if bracket_cfg.x_init_low > bracket_cfg.x_init_high:
        raise ValueError("`bracket_cfg.x_init_low` must be <= `x_init_high`.")
    if bracket_cfg.expansion_factor <= 1.0:
        raise ValueError("`bracket_cfg.expansion_factor` must be > 1.")
    if bracket_cfg.max_expansions < 0:
        raise ValueError("`bracket_cfg.max_expansions` must be >= 0.")


def build_distribution_quantile_backend(
    *,
    distribution: Any,
    solver_cfg: SolverConfig | Mapping[str, Any] | None = None,
    bracket_cfg: BracketConfig | Mapping[str, Any] | None = None,
    eps: float = 1e-10,
) -> DistributionQuantileBackend:
    """Build the inverse-CDF backend for a 1D continuous target distribution."""

    solver = _as_solver_cfg(solver_cfg)
    bracket = _as_bracket_cfg(bracket_cfg)

    if not np.isfinite(eps) or not (0.0 < eps < 0.5):
        raise ValueError("`eps` must be finite and satisfy 0 < eps < 0.5.")

    _validate_solver_cfg(solver)
    _validate_bracket_cfg(bracket)
    _validate_distribution(distribution)

    return DistributionQuantileBackend(
        distribution=distribution,
        solver_cfg=solver,
        bracket_cfg=bracket,
        eps=float(eps),
    )


__all__ = [
    "BracketConfig",
    "DistributionQuantileBackend",
    "SolverConfig",
    "build_distribution_quantile_backend",
]
