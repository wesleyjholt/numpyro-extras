"""Step 01 manifest for Task 06 mixture inverse-CDF backend.

Required module:
- `numpyro_extras.mixture_quantile` (expected file: `src/numpyro_extras/mixture_quantile.py`)

Required public API:
- `build_mixture_quantile_backend(...)`
- backend object methods: `cdf(x)`, `icdf(u)`, `log_prob(x)`
- optional diagnostics: `icdf_with_status(u)`, `validate()`

Behavioral pass gates:
- constructor validates mixture inputs (`weights`, component distribution, configs)
- scalar and batched JAX array support for `cdf/icdf/log_prob`
- single-component reduction matches base distribution (`cdf/icdf/log_prob`)
- roundtrip and monotonicity contracts hold
- tails remain finite and stable under hard settings
- JAX `jit` + `vmap` compatibility
- explicit bracketing failure errors in strict failure scenarios
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import jax
import numpyro.distributions as dist
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


MODULE_NAME = "numpyro_extras.mixture_quantile"
BACKEND_CTOR_NAME = "build_mixture_quantile_backend"
DEFAULT_EPS = 1e-10
TAIL_U = jnp.array([1e-8, 1e-6, 1.0 - 1e-6, 1.0 - 1e-8], dtype=jnp.float64)


MIXTURE_CASES = [
    pytest.param(
        dict(weights=[0.50, 0.50], loc=[-2.0, 2.0], scale=[0.8, 1.1]),
        id="balanced_bimodal",
    ),
    pytest.param(
        dict(weights=[0.99, 0.01], loc=[-1.5, 4.0], scale=[1.0, 0.6]),
        id="imbalanced_bimodal",
    ),
    pytest.param(
        dict(weights=[0.45, 0.55], loc=[0.0, 0.12], scale=[1.0, 1.05]),
        id="nearly_overlapping",
    ),
    pytest.param(
        dict(weights=[0.60, 0.40], loc=[-9.0, 10.0], scale=[0.7, 1.4]),
        id="wide_separation",
    ),
]


def _import_mixture_quantile_module() -> ModuleType:
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.mixture_quantile` "
                "(expected at `src/numpyro_extras/mixture_quantile.py`)."
            )
        raise


def _get_backend_constructor():
    module = _import_mixture_quantile_module()
    ctor = getattr(module, BACKEND_CTOR_NAME, None)
    if ctor is None:
        pytest.fail(
            "Missing `build_mixture_quantile_backend` in "
            "`numpyro_extras.mixture_quantile`."
        )
    if not callable(ctor):
        pytest.fail("`build_mixture_quantile_backend` must be callable.")
    return ctor


def _build_backend(
    *,
    weights,
    loc,
    scale,
    solver_cfg=None,
    bracket_cfg=None,
    eps=DEFAULT_EPS,
):
    ctor = _get_backend_constructor()
    kwargs = {
        "weights": jnp.asarray(weights, dtype=jnp.float64),
        "component_distribution": dist.Normal(
            loc=jnp.asarray(loc, dtype=jnp.float64),
            scale=jnp.asarray(scale, dtype=jnp.float64),
        ),
        "eps": eps,
    }
    if solver_cfg is not None:
        kwargs["solver_cfg"] = solver_cfg
    if bracket_cfg is not None:
        kwargs["bracket_cfg"] = bracket_cfg
    return ctor(**kwargs)


def test_api_contract_mixture_quantile_backend():
    ctor = _get_backend_constructor()
    signature = inspect.signature(ctor)
    for param_name in ("weights", "component_distribution"):
        assert param_name in signature.parameters, (
            f"`{BACKEND_CTOR_NAME}` must accept `{param_name}`."
        )

    backend = _build_backend(
        weights=[0.5, 0.5],
        loc=[-1.0, 1.0],
        scale=[0.9, 1.2],
    )

    for method_name in ("cdf", "icdf", "log_prob"):
        method = getattr(backend, method_name, None)
        assert callable(method), f"Backend missing callable `{method_name}`."

    x_scalar = jnp.asarray(0.15, dtype=jnp.float64)
    u_scalar = jnp.asarray(0.73, dtype=jnp.float64)
    assert jnp.shape(backend.cdf(x_scalar)) == (), "`cdf` must accept scalar x."
    assert jnp.shape(backend.icdf(u_scalar)) == (), "`icdf` must accept scalar u."
    assert jnp.shape(backend.log_prob(x_scalar)) == (), (
        "`log_prob` must accept scalar x."
    )

    x_batch = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float64)
    u_batch = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float64)
    assert backend.cdf(x_batch).shape == x_batch.shape, (
        "`cdf` must preserve batched shape."
    )
    assert backend.icdf(u_batch).shape == u_batch.shape, (
        "`icdf` must preserve batched shape."
    )
    assert backend.log_prob(x_batch).shape == x_batch.shape, (
        "`log_prob` must preserve batched shape."
    )

    if hasattr(backend, "icdf_with_status"):
        status_out = backend.icdf_with_status(u_batch)
        assert len(status_out) == 3, "`icdf_with_status` must return (x, converged, n_steps)."
        x_status, converged, n_steps = status_out
        assert jnp.shape(x_status) == u_batch.shape
        assert jnp.shape(converged) == u_batch.shape
        assert jnp.all(jnp.asarray(n_steps) >= 0), "`n_steps` must be non-negative."

    if hasattr(backend, "validate"):
        validate_out = backend.validate()
        assert isinstance(validate_out, dict), "`validate()` must return a dict."


def test_single_component_reduces_to_base_distribution():
    backend = _build_backend(weights=[1.0], loc=[0.75], scale=[1.3])
    base = dist.Normal(loc=0.75, scale=1.3)

    interior_u = jnp.linspace(1e-4, 1.0 - 1e-4, 513, dtype=jnp.float64)
    x_probe_interior = base.icdf(interior_u)
    x_probe_tails = base.icdf(TAIL_U)

    icdf_interior_err = float(
        jnp.max(jnp.abs(backend.icdf(interior_u) - base.icdf(interior_u)))
    )
    icdf_tail_err = float(jnp.max(jnp.abs(backend.icdf(TAIL_U) - base.icdf(TAIL_U))))
    assert icdf_interior_err < 1e-5, (
        f"Single-component icdf mismatch (interior): max_err={icdf_interior_err:.3e}"
    )
    assert icdf_tail_err < 1e-4, (
        f"Single-component icdf mismatch (tails): max_err={icdf_tail_err:.3e}"
    )

    cdf_interior_err = float(
        jnp.max(jnp.abs(backend.cdf(x_probe_interior) - base.cdf(x_probe_interior)))
    )
    cdf_tail_err = float(
        jnp.max(jnp.abs(backend.cdf(x_probe_tails) - base.cdf(x_probe_tails)))
    )
    assert cdf_interior_err < 1e-5, (
        f"Single-component cdf mismatch (interior): max_err={cdf_interior_err:.3e}"
    )
    assert cdf_tail_err < 1e-4, (
        f"Single-component cdf mismatch (tails): max_err={cdf_tail_err:.3e}"
    )

    log_prob_interior_err = float(
        jnp.max(
            jnp.abs(
                backend.log_prob(x_probe_interior) - base.log_prob(x_probe_interior)
            )
        )
    )
    log_prob_tail_err = float(
        jnp.max(jnp.abs(backend.log_prob(x_probe_tails) - base.log_prob(x_probe_tails)))
    )
    assert log_prob_interior_err < 1e-5, (
        "Single-component log_prob mismatch (interior): "
        f"max_err={log_prob_interior_err:.3e}"
    )
    assert log_prob_tail_err < 1e-4, (
        f"Single-component log_prob mismatch (tails): max_err={log_prob_tail_err:.3e}"
    )


def test_roundtrip_cdf_icdf_dense_grid():
    backend = _build_backend(
        weights=[0.5, 0.5],
        loc=[-2.5, 2.0],
        scale=[0.8, 1.0],
    )
    u_grid = jnp.linspace(1e-6, 1.0 - 1e-6, 2001, dtype=jnp.float64)
    x_grid = backend.icdf(u_grid)
    u_roundtrip = backend.cdf(x_grid)
    err = jnp.abs(u_roundtrip - u_grid)

    interior_mask = (u_grid > 1e-4) & (u_grid < 1.0 - 1e-4)
    interior_max_err = float(jnp.max(err[interior_mask]))
    tail_max_err = float(jnp.max(err[~interior_mask]))

    assert interior_max_err < 5e-5, (
        f"Roundtrip interior tolerance violated: max_err={interior_max_err:.3e}"
    )
    assert tail_max_err < 2e-4, (
        f"Roundtrip tail tolerance violated: max_err={tail_max_err:.3e}"
    )


def test_tail_inputs_remain_finite():
    backend = _build_backend(
        weights=[0.999, 0.001],
        loc=[-8.0, 14.0],
        scale=[0.05, 4.5],
    )
    x_tail = backend.icdf(TAIL_U)
    cdf_tail = backend.cdf(x_tail)
    log_prob_tail = backend.log_prob(x_tail)

    assert jnp.all(jnp.isfinite(x_tail)), "Tail icdf returned non-finite values."
    assert jnp.all(jnp.isfinite(cdf_tail)), "Tail cdf returned non-finite values."
    assert jnp.all(jnp.isfinite(log_prob_tail)), (
        "Tail log_prob returned non-finite values."
    )

    tail_roundtrip_err = float(jnp.max(jnp.abs(cdf_tail - TAIL_U)))
    assert tail_roundtrip_err < 2e-4, (
        f"Tail roundtrip tolerance violated: max_err={tail_roundtrip_err:.3e}"
    )


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_icdf_is_monotone(case):
    backend = _build_backend(
        weights=case["weights"],
        loc=case["loc"],
        scale=case["scale"],
    )
    u_grid = jnp.linspace(1e-6, 1.0 - 1e-6, 2049, dtype=jnp.float64)
    x_grid = backend.icdf(u_grid)
    min_dx = float(jnp.min(jnp.diff(x_grid)))
    assert min_dx >= -1e-8, f"`icdf` is not monotone: min_delta={min_dx:.3e}"


def test_jit_and_vmap_compatibility():
    backend = _build_backend(
        weights=[0.55, 0.45],
        loc=[-1.5, 2.0],
        scale=[0.9, 1.2],
    )
    u = jnp.linspace(1e-5, 1.0 - 1e-5, 256, dtype=jnp.float64)

    icdf_jit = jax.jit(backend.icdf)
    cdf_jit = jax.jit(backend.cdf)
    x_jit = icdf_jit(u)
    u_jit = cdf_jit(x_jit)
    x_vmap = jax.vmap(lambda ui: backend.icdf(ui))(u)

    assert x_jit.shape == u.shape, "JIT icdf output shape mismatch."
    assert u_jit.shape == u.shape, "JIT cdf output shape mismatch."
    assert x_vmap.shape == u.shape, "VMAP icdf output shape mismatch."
    assert x_jit.dtype == u.dtype, "JIT icdf dtype must match input dtype."
    assert u_jit.dtype == u.dtype, "JIT cdf dtype must match input dtype."

    roundtrip_err = float(jnp.max(jnp.abs(u_jit - u)))
    assert roundtrip_err < 5e-5, (
        f"JIT roundtrip tolerance violated: max_err={roundtrip_err:.3e}"
    )
    assert jnp.allclose(x_vmap, x_jit, atol=1e-7, rtol=1e-7), (
        "VMAP icdf does not match JIT icdf."
    )

    if hasattr(backend, "icdf_with_status"):
        x_status, converged, n_steps = jax.jit(backend.icdf_with_status)(u)
        assert x_status.shape == u.shape
        assert converged.shape == u.shape
        assert jnp.all(converged), "`icdf_with_status` reported non-convergence."
        assert jnp.all(jnp.asarray(n_steps) >= 0), "`n_steps` must be non-negative."


def test_bracket_failure_reports_useful_error():
    backend = _build_backend(
        weights=[0.5, 0.5],
        loc=[-12.0, 12.0],
        scale=[0.8, 0.8],
        bracket_cfg={
            "x_init_low": 0.0,
            "x_init_high": 0.0,
            "expansion_factor": 2.0,
            "max_expansions": 0,
        },
    )
    with pytest.raises(ValueError, match=r"(?i)bracket|root|converg"):
        backend.icdf(jnp.asarray(0.99, dtype=jnp.float64))


def test_invalid_weights_are_rejected():
    ctor = _get_backend_constructor()
    component_distribution = dist.Normal(
        loc=jnp.asarray([-1.0, 1.0], dtype=jnp.float64),
        scale=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
    )

    def _construct_and_eval(weights):
        backend = ctor(
            weights=jnp.asarray(weights, dtype=jnp.float64),
            component_distribution=component_distribution,
        )
        _ = backend.icdf(jnp.asarray(0.5, dtype=jnp.float64))

    with pytest.raises((ValueError, TypeError), match=r"(?i)weight|sum|normal|non[- ]?negative"):
        _construct_and_eval([-0.1, 1.1])

    with pytest.raises((ValueError, TypeError), match=r"(?i)weight|sum|normal"):
        _construct_and_eval([0.2, 0.2])


def test_invalid_u_values_are_clamped_deterministically():
    backend = _build_backend(
        weights=[0.7, 0.3],
        loc=[-2.0, 1.5],
        scale=[1.1, 0.7],
        eps=DEFAULT_EPS,
    )
    bad_u = jnp.asarray([-0.3, 0.0, 1.0, 1.3], dtype=jnp.float64)
    x = backend.icdf(bad_u)
    assert jnp.all(jnp.isfinite(x)), "icdf must handle out-of-range u deterministically."

    u_roundtrip = backend.cdf(x)
    expected = jnp.clip(bad_u, DEFAULT_EPS, 1.0 - DEFAULT_EPS)
    max_err = float(jnp.max(jnp.abs(u_roundtrip - expected)))
    assert max_err < 2e-4, (
        "Out-of-range u handling must match deterministic clipping: "
        f"max_err={max_err:.3e}"
    )
