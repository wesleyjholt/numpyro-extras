"""Tests for the mixture inverse-CDF backend (Task 06 pre-implementation gates).

Required API names (strict):
1. ``numpyro_extras.mixture_quantile.build_mixture_quantile_backend(...)``
2. Returned backend exposes ``cdf(x)``, ``icdf(u)``, and ``log_prob(x)``.

Optional API:
1. ``icdf_with_status(u) -> (x, converged, n_steps)``.

Pass gates for Task 06 implementation:
1. Single-component reduction matches analytic Normal component.
2. Roundtrip ``cdf(icdf(u))`` is accurate on dense interior and extreme tails.
3. ``icdf`` is monotone for sorted ``u``.
4. Tail behavior remains finite in difficult mixtures.
5. Bracketing failure is explicit and informative.
6. ``jit`` and ``vmap`` are supported with stable shapes/dtypes.
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from numpyro.distributions import Normal


INTERIOR_TOL = 5e-5
TAIL_TOL = 2e-4
SINGLE_COMPONENT_INTERIOR_TOL = 1e-5
SINGLE_COMPONENT_TAIL_TOL = 1e-4

U_DENSE = jnp.linspace(1e-6, 1.0 - 1e-6, 512, dtype=jnp.float32)
U_TAIL = jnp.array([1e-8, 1e-6, 1.0 - 1e-6, 1.0 - 1e-8], dtype=jnp.float32)

MIXTURE_CASES = [
    pytest.param(
        dict(
            weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
            loc=jnp.array([-2.0, 2.0], dtype=jnp.float32),
            scale=jnp.array([0.9, 1.1], dtype=jnp.float32),
        ),
        id="balanced_bimodal",
    ),
    pytest.param(
        dict(
            weights=jnp.array([0.99, 0.01], dtype=jnp.float32),
            loc=jnp.array([0.0, 5.0], dtype=jnp.float32),
            scale=jnp.array([1.0, 0.4], dtype=jnp.float32),
        ),
        id="imbalanced_bimodal",
    ),
    pytest.param(
        dict(
            weights=jnp.array([0.45, 0.55], dtype=jnp.float32),
            loc=jnp.array([0.0, 0.1], dtype=jnp.float32),
            scale=jnp.array([1.0, 1.05], dtype=jnp.float32),
        ),
        id="overlapping_components",
    ),
    pytest.param(
        dict(
            weights=jnp.array([0.3, 0.7], dtype=jnp.float32),
            loc=jnp.array([-8.0, 8.0], dtype=jnp.float32),
            scale=jnp.array([0.5, 0.8], dtype=jnp.float32),
        ),
        id="wide_separation",
    ),
]


def _assert_allclose(actual, expected, *, rtol=1e-6, atol=1e-6, msg=""):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol, err_msg=msg)


def _max_abs_error(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def _require_backend_module():
    try:
        module = importlib.import_module("numpyro_extras.mixture_quantile")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.mixture_quantile` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )
    return module


def _build_backend(
    *,
    weights,
    loc,
    scale,
    solver_cfg=None,
    bracket_cfg=None,
):
    module = _require_backend_module()
    if not hasattr(module, "build_mixture_quantile_backend"):
        pytest.fail(
            "Expected `numpyro_extras.mixture_quantile.build_mixture_quantile_backend` to exist."
        )

    component_distribution = Normal(loc=loc, scale=scale)
    return module.build_mixture_quantile_backend(
        weights=weights,
        component_distribution=component_distribution,
        solver_cfg=solver_cfg,
        bracket_cfg=bracket_cfg,
    )


def _oracle_mixture_cdf(x, weights, component_distribution):
    return jnp.sum(weights * component_distribution.cdf(jnp.expand_dims(x, axis=-1)), axis=-1)


def _oracle_mixture_log_prob(x, weights, component_distribution):
    return logsumexp(
        jnp.log(weights) + component_distribution.log_prob(jnp.expand_dims(x, axis=-1)),
        axis=-1,
    )


def test_api_contract_mixture_quantile_backend():
    backend = _build_backend(
        weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
        loc=jnp.array([-1.0, 1.0], dtype=jnp.float32),
        scale=jnp.array([0.8, 1.2], dtype=jnp.float32),
        solver_cfg={"rtol": 1e-6, "atol": 1e-6, "max_steps": 128},
        bracket_cfg={
            "x_init_low": -8.0,
            "x_init_high": 8.0,
            "expansion_factor": 2.0,
            "max_expansions": 32,
        },
    )

    for name in ("cdf", "icdf", "log_prob"):
        assert hasattr(backend, name), f"Backend is missing required method `{name}`."
        assert callable(getattr(backend, name)), f"`{name}` must be callable."

    x_scalar = jnp.array(0.1, dtype=jnp.float32)
    x_batch = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
    u_scalar = jnp.array(0.3, dtype=jnp.float32)
    u_batch = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float32)

    cdf_scalar = backend.cdf(x_scalar)
    cdf_batch = backend.cdf(x_batch)
    icdf_scalar = backend.icdf(u_scalar)
    icdf_batch = backend.icdf(u_batch)
    log_prob_scalar = backend.log_prob(x_scalar)
    log_prob_batch = backend.log_prob(x_batch)

    assert jnp.shape(cdf_scalar) == ()
    assert jnp.shape(icdf_scalar) == ()
    assert jnp.shape(log_prob_scalar) == ()
    assert cdf_batch.shape == x_batch.shape
    assert icdf_batch.shape == u_batch.shape
    assert log_prob_batch.shape == x_batch.shape


def test_single_component_reduces_to_base_distribution():
    weights = jnp.array([1.0], dtype=jnp.float32)
    loc = jnp.array([0.4], dtype=jnp.float32)
    scale = jnp.array([1.3], dtype=jnp.float32)

    backend = _build_backend(weights=weights, loc=loc, scale=scale)
    component = Normal(loc=loc[0], scale=scale[0])

    x_interior = jnp.array([-2.0, -0.2, 0.0, 0.5, 2.0], dtype=jnp.float32)
    x_tail = jnp.array([-6.0, -5.0, 5.0, 6.0], dtype=jnp.float32)
    u_interior = jnp.array([1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-3], dtype=jnp.float32)
    u_tail = U_TAIL

    _assert_allclose(
        backend.cdf(x_interior),
        component.cdf(x_interior),
        atol=SINGLE_COMPONENT_INTERIOR_TOL,
        rtol=1e-5,
        msg="Single-component mixture CDF must match component CDF in interior.",
    )
    _assert_allclose(
        backend.cdf(x_tail),
        component.cdf(x_tail),
        atol=SINGLE_COMPONENT_TAIL_TOL,
        rtol=1e-4,
        msg="Single-component mixture CDF must match component CDF in tails.",
    )
    _assert_allclose(
        backend.log_prob(x_interior),
        component.log_prob(x_interior),
        atol=SINGLE_COMPONENT_INTERIOR_TOL,
        rtol=1e-5,
        msg="Single-component mixture log_prob must match component log_prob in interior.",
    )
    _assert_allclose(
        backend.icdf(u_interior),
        component.icdf(u_interior),
        atol=SINGLE_COMPONENT_INTERIOR_TOL,
        rtol=1e-5,
        msg="Single-component mixture icdf must match component icdf in interior.",
    )
    _assert_allclose(
        backend.icdf(u_tail),
        component.icdf(u_tail),
        atol=SINGLE_COMPONENT_TAIL_TOL,
        rtol=1e-4,
        msg="Single-component mixture icdf must match component icdf in tails.",
    )


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_roundtrip_cdf_icdf_dense_grid(case):
    backend = _build_backend(weights=case["weights"], loc=case["loc"], scale=case["scale"])

    x = backend.icdf(U_DENSE)
    recovered = backend.cdf(x)

    max_error = _max_abs_error(recovered, U_DENSE)
    assert max_error < INTERIOR_TOL, (
        f"Interior roundtrip error exceeded tolerance: {max_error} >= {INTERIOR_TOL}"
    )


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_tail_inputs_remain_finite(case):
    backend = _build_backend(weights=case["weights"], loc=case["loc"], scale=case["scale"])

    x_tail = backend.icdf(U_TAIL)
    u_tail_recovered = backend.cdf(x_tail)
    logp_tail = backend.log_prob(x_tail)

    assert jnp.all(jnp.isfinite(x_tail)), "Tail icdf outputs must be finite."
    assert jnp.all(jnp.isfinite(u_tail_recovered)), "Tail cdf(icdf(u)) must be finite."
    assert jnp.all(jnp.isfinite(logp_tail)), "Tail log_prob values must be finite."

    tail_error = _max_abs_error(u_tail_recovered, U_TAIL)
    assert tail_error < TAIL_TOL, f"Tail roundtrip error exceeded tolerance: {tail_error} >= {TAIL_TOL}"


@pytest.mark.parametrize("case", MIXTURE_CASES)
def test_icdf_is_monotone(case):
    backend = _build_backend(weights=case["weights"], loc=case["loc"], scale=case["scale"])

    x = backend.icdf(U_DENSE)
    diffs = jnp.diff(x)
    assert jnp.all(diffs >= -1e-6), "icdf(u) must be nondecreasing for sorted u."


def test_jit_and_vmap_compatibility():
    case = dict(
        weights=jnp.array([0.7, 0.3], dtype=jnp.float32),
        loc=jnp.array([-1.5, 2.0], dtype=jnp.float32),
        scale=jnp.array([0.6, 1.4], dtype=jnp.float32),
    )
    backend = _build_backend(**case)

    u = jnp.array([0.02, 0.2, 0.5, 0.8, 0.98], dtype=jnp.float32)
    x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=jnp.float32)

    jit_icdf = jax.jit(backend.icdf)
    jit_cdf = jax.jit(backend.cdf)
    jit_log_prob = jax.jit(backend.log_prob)

    icdf_out = jit_icdf(u)
    cdf_out = jit_cdf(x)
    log_prob_out = jit_log_prob(x)

    assert icdf_out.shape == u.shape
    assert cdf_out.shape == x.shape
    assert log_prob_out.shape == x.shape
    assert icdf_out.dtype == u.dtype, "icdf output dtype should follow u dtype."
    assert cdf_out.dtype == x.dtype, "cdf output dtype should follow x dtype."

    vmapped_icdf = jax.vmap(lambda ui: backend.icdf(ui))(u)
    vmapped_cdf = jax.vmap(lambda xi: backend.cdf(xi))(x)
    _assert_allclose(vmapped_icdf, backend.icdf(u), atol=1e-6, rtol=1e-6)
    _assert_allclose(vmapped_cdf, backend.cdf(x), atol=1e-6, rtol=1e-6)


def test_bracket_expansion_handles_difficult_mixture():
    backend = _build_backend(
        weights=jnp.array([0.4, 0.6], dtype=jnp.float32),
        loc=jnp.array([25.0, 35.0], dtype=jnp.float32),
        scale=jnp.array([0.8, 1.2], dtype=jnp.float32),
        solver_cfg={"rtol": 1e-6, "atol": 1e-6, "max_steps": 256},
        bracket_cfg={
            "x_init_low": -1.0,
            "x_init_high": 1.0,
            "expansion_factor": 2.0,
            "max_expansions": 8,
        },
    )

    u = jnp.array([0.2, 0.5, 0.8], dtype=jnp.float32)
    x = backend.icdf(u)
    recovered = backend.cdf(x)

    assert jnp.all(jnp.isfinite(x)), "Bracket expansion path should return finite values."
    assert _max_abs_error(recovered, u) < TAIL_TOL, "Expanded brackets should still produce accurate roots."


def test_bracket_failure_reports_useful_error():
    backend = _build_backend(
        weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
        loc=jnp.array([1e6, 1e6 + 1.0], dtype=jnp.float32),
        scale=jnp.array([1.0, 1.0], dtype=jnp.float32),
        solver_cfg={"rtol": 1e-6, "atol": 1e-6, "max_steps": 32},
        bracket_cfg={
            "x_init_low": -1.0,
            "x_init_high": 1.0,
            "expansion_factor": 2.0,
            "max_expansions": 2,
        },
    )

    with pytest.raises(Exception) as excinfo:
        backend.icdf(jnp.array([0.5], dtype=jnp.float32))

    message = str(excinfo.value).lower()
    assert (
        "bracket" in message or "conver" in message or "root" in message
    ), "Failure should explicitly mention bracketing/convergence/root-finding."


def test_optional_icdf_with_status_reports_convergence_when_available():
    backend = _build_backend(
        weights=jnp.array([0.6, 0.4], dtype=jnp.float32),
        loc=jnp.array([-1.0, 1.5], dtype=jnp.float32),
        scale=jnp.array([1.0, 0.7], dtype=jnp.float32),
    )
    if not hasattr(backend, "icdf_with_status"):
        pytest.skip("icdf_with_status is optional and not implemented.")

    u = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float32)
    x, converged, n_steps = backend.icdf_with_status(u)

    assert x.shape == u.shape
    assert converged.shape == u.shape
    assert n_steps.shape == u.shape
    assert jnp.all(converged), "All diagnostics flags should indicate convergence for this easy case."
    assert jnp.all(n_steps > 0), "Diagnostics iteration counts should be positive."


@pytest.mark.parametrize(
    "weights",
    [
        jnp.array([-0.1, 1.1], dtype=jnp.float32),
        jnp.array([0.2, 0.2], dtype=jnp.float32),
    ],
    ids=["negative_weight", "not_normalized"],
)
def test_invalid_weights_are_rejected(weights):
    loc = jnp.array([-1.0, 1.0], dtype=jnp.float32)
    scale = jnp.array([1.0, 1.0], dtype=jnp.float32)

    try:
        backend = _build_backend(weights=weights, loc=loc, scale=scale)
    except Exception:
        return

    with pytest.raises(Exception):
        backend.cdf(jnp.array(0.0, dtype=jnp.float32))


def test_invalid_u_values_outside_unit_interval_are_handled_or_rejected():
    case = dict(
        weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
        loc=jnp.array([-1.5, 2.5], dtype=jnp.float32),
        scale=jnp.array([0.9, 1.1], dtype=jnp.float32),
    )
    backend = _build_backend(**case)
    invalid_u = jnp.array([-1e-3, 1.001], dtype=jnp.float32)

    try:
        out = backend.icdf(invalid_u)
    except Exception as exc:
        msg = str(exc).lower()
        assert (
            "u" in msg or "[0, 1]" in msg or "unit" in msg or "range" in msg
        ), "Out-of-range u error should be explicit."
        return

    assert jnp.all(jnp.isfinite(out)), "If invalid u is handled, outputs must remain finite."
    clipped = backend.icdf(jnp.clip(invalid_u, 0.0, 1.0))
    _assert_allclose(
        out,
        clipped,
        atol=1e-6,
        rtol=1e-6,
        msg="Out-of-range u handling should be deterministic (match clipped behavior).",
    )


def test_endpoint_u_values_are_clamped_and_finite():
    case = dict(
        weights=jnp.array([0.5, 0.5], dtype=jnp.float32),
        loc=jnp.array([-2.0, 2.0], dtype=jnp.float32),
        scale=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    backend = _build_backend(**case)

    u = jnp.array([0.0, 1.0], dtype=jnp.float32)
    x = backend.icdf(u)

    assert jnp.all(jnp.isfinite(x)), "Endpoint u values should produce finite outputs via internal clamping."
    assert jnp.all(jnp.diff(x) >= 0.0), "icdf(0) should not exceed icdf(1)."


def test_oracle_cdf_and_log_prob_match_weighted_component_formulas():
    case = dict(
        weights=jnp.array([0.25, 0.75], dtype=jnp.float32),
        loc=jnp.array([-0.5, 1.5], dtype=jnp.float32),
        scale=jnp.array([0.7, 1.3], dtype=jnp.float32),
    )
    backend = _build_backend(**case)
    component_distribution = Normal(loc=case["loc"], scale=case["scale"])
    x = jnp.array([-2.0, -0.4, 0.0, 0.7, 2.5], dtype=jnp.float32)

    expected_cdf = _oracle_mixture_cdf(x, case["weights"], component_distribution)
    expected_log_prob = _oracle_mixture_log_prob(x, case["weights"], component_distribution)

    _assert_allclose(
        backend.cdf(x),
        expected_cdf,
        atol=2e-6,
        rtol=1e-5,
        msg="Mixture cdf(x) must equal weighted component CDF sum.",
    )
    _assert_allclose(
        backend.log_prob(x),
        expected_log_prob,
        atol=2e-6,
        rtol=1e-5,
        msg="Mixture log_prob(x) must equal logsumexp(log w + component log_prob).",
    )
