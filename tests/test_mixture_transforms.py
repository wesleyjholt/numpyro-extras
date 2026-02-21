"""Tests for Task 09 NumPyro transform wrappers.

These tests define the required behavior for:
1. ``UniformToMixtureTransform``
2. ``NormalToMixtureTransform``
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions import Normal
from numpyro.distributions.constraints import _Real, unit_interval


UNIFORM_INTERIOR_RTOL = 1e-5
UNIFORM_TAIL_RTOL = 1e-4
NORMAL_INTERIOR_RTOL = 2e-5
NORMAL_TAIL_RTOL = 2e-4
JACOBIAN_FD_REL_TOL = 5e-3


class MockInterpolator:
    """Deterministic smooth bijection used to test transform wrappers."""

    def __init__(self, *, scale=1.7, shift=-0.35, clip_u_eps=1e-12):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        self.shift = jnp.asarray(shift, dtype=jnp.float32)
        self.clip_u_eps = jnp.asarray(clip_u_eps, dtype=jnp.float32)

    def _clip_u(self, u):
        return jnp.clip(jnp.asarray(u, dtype=jnp.float32), self.clip_u_eps, 1.0 - self.clip_u_eps)

    def icdf(self, u):
        u = self._clip_u(u)
        logit_u = jnp.log(u) - jnp.log1p(-u)
        return self.shift + self.scale * logit_u

    def cdf(self, x):
        x = jnp.asarray(x, dtype=jnp.float32)
        return jax.nn.sigmoid((x - self.shift) / self.scale)

    def dxdu(self, u):
        u = self._clip_u(u)
        return self.scale / (u * (1.0 - u))

    def log_abs_dxdu(self, u):
        u = self._clip_u(u)
        return jnp.log(self.scale) - jnp.log(u) - jnp.log1p(-u)


@pytest.fixture(scope="module")
def mock_interpolator():
    return MockInterpolator()


def _assert_allclose(actual, expected, *, rtol, atol=1e-7, msg=""):
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
        err_msg=msg,
    )


def _require_transform_classes():
    try:
        module = importlib.import_module("numpyro_extras.transforms")
    except Exception as exc:  # pragma: no cover - exercised before implementation exists.
        pytest.fail(
            "Required module `numpyro_extras.transforms` is missing or failed to import: "
            f"{type(exc).__name__}: {exc}"
        )

    missing = [
        name
        for name in ("UniformToMixtureTransform", "NormalToMixtureTransform")
        if not hasattr(module, name)
    ]
    if missing:
        pytest.fail(
            "Expected transform classes to exist in `numpyro_extras.transforms`: "
            f"missing {missing}."
        )

    return module.UniformToMixtureTransform, module.NormalToMixtureTransform


def _instantiate_uniform_transform(interpolator):
    UniformToMixtureTransform, _ = _require_transform_classes()
    attempts = (
        lambda: UniformToMixtureTransform(
            interpolator=interpolator,
            transform_cfg={"clip_u_eps": 1e-10, "validate_args": False},
        ),
        lambda: UniformToMixtureTransform(interpolator, transform_cfg={"clip_u_eps": 1e-10, "validate_args": False}),
        lambda: UniformToMixtureTransform(interpolator=interpolator),
        lambda: UniformToMixtureTransform(interpolator),
    )
    errors = []
    for build in attempts:
        try:
            return build()
        except TypeError as exc:
            errors.append(str(exc))

    pytest.fail(
        "Could not instantiate `UniformToMixtureTransform` with expected constructor patterns. "
        f"TypeErrors: {errors}"
    )


def _instantiate_normal_transform(interpolator):
    _, NormalToMixtureTransform = _require_transform_classes()
    attempts = (
        lambda: NormalToMixtureTransform(
            interpolator=interpolator,
            transform_cfg={"clip_u_eps": 1e-10, "validate_args": False},
        ),
        lambda: NormalToMixtureTransform(interpolator, transform_cfg={"clip_u_eps": 1e-10, "validate_args": False}),
        lambda: NormalToMixtureTransform(interpolator=interpolator),
        lambda: NormalToMixtureTransform(interpolator),
    )
    errors = []
    for build in attempts:
        try:
            return build()
        except TypeError as exc:
            errors.append(str(exc))

    pytest.fail(
        "Could not instantiate `NormalToMixtureTransform` with expected constructor patterns. "
        f"TypeErrors: {errors}"
    )


def test_uniform_transform_roundtrip(mock_interpolator):
    transform = _instantiate_uniform_transform(mock_interpolator)

    key = jax.random.PRNGKey(0)
    u_random = jax.random.uniform(key, shape=(256,), minval=1e-7, maxval=1.0 - 1e-7)
    u_grid = jnp.linspace(1e-8, 1.0 - 1e-8, 257, dtype=jnp.float32)
    u = jnp.concatenate([u_grid, u_random], axis=0)

    recovered = transform._inverse(transform(u))
    errors = jnp.abs(recovered - u)

    interior_mask = (u >= 1e-3) & (u <= 1.0 - 1e-3)
    tail_mask = ~interior_mask

    interior_max = float(jnp.max(errors[interior_mask]))
    tail_max = float(jnp.max(errors[tail_mask]))

    assert interior_max < UNIFORM_INTERIOR_RTOL, (
        "Uniform transform interior roundtrip exceeded tolerance: "
        f"{interior_max} >= {UNIFORM_INTERIOR_RTOL}"
    )
    assert tail_max < UNIFORM_TAIL_RTOL, (
        "Uniform transform tail roundtrip exceeded tolerance: "
        f"{tail_max} >= {UNIFORM_TAIL_RTOL}"
    )


def test_normal_transform_roundtrip(mock_interpolator):
    transform = _instantiate_normal_transform(mock_interpolator)

    key = jax.random.PRNGKey(1)
    z_random = jnp.clip(jax.random.normal(key, shape=(256,), dtype=jnp.float32) * 2.0, -7.5, 7.5)
    z_grid = jnp.linspace(-8.0, 8.0, 257, dtype=jnp.float32)
    z = jnp.concatenate([z_grid, z_random], axis=0)

    recovered = transform._inverse(transform(z))
    errors = jnp.abs(recovered - z)

    interior_mask = jnp.abs(z) <= 4.0
    tail_mask = ~interior_mask

    interior_max = float(jnp.max(errors[interior_mask]))
    tail_max = float(jnp.max(errors[tail_mask]))

    assert interior_max < NORMAL_INTERIOR_RTOL, (
        "Normal transform interior roundtrip exceeded tolerance: "
        f"{interior_max} >= {NORMAL_INTERIOR_RTOL}"
    )
    assert tail_max < NORMAL_TAIL_RTOL, (
        "Normal transform tail roundtrip exceeded tolerance: "
        f"{tail_max} >= {NORMAL_TAIL_RTOL}"
    )


def test_uniform_log_det_matches_finite_difference(mock_interpolator):
    transform = _instantiate_uniform_transform(mock_interpolator)

    u = jnp.linspace(1e-3, 1.0 - 1e-3, 121, dtype=jnp.float32)
    h = jnp.asarray(1e-4, dtype=jnp.float32)

    y_plus = transform(jnp.clip(u + h, 0.0, 1.0))
    y_minus = transform(jnp.clip(u - h, 0.0, 1.0))
    fd_abs_dy_du = jnp.abs((y_plus - y_minus) / (2.0 * h))

    y = transform(u)
    reported_abs_dy_du = jnp.exp(transform.log_abs_det_jacobian(u, y))

    rel_error = jnp.abs(reported_abs_dy_du - fd_abs_dy_du) / jnp.maximum(fd_abs_dy_du, 1e-12)
    rel_error_max = float(jnp.max(rel_error))

    assert rel_error_max < JACOBIAN_FD_REL_TOL, (
        "Uniform Jacobian finite-difference mismatch exceeded tolerance: "
        f"{rel_error_max} >= {JACOBIAN_FD_REL_TOL}"
    )


def test_normal_log_det_matches_chain_rule(mock_interpolator):
    transform = _instantiate_normal_transform(mock_interpolator)
    standard_normal = Normal(0.0, 1.0)

    z = jnp.concatenate(
        [
            jnp.linspace(-4.0, 4.0, 121, dtype=jnp.float32),
            jnp.array([-10.0, -8.0, 8.0, 10.0], dtype=jnp.float32),
        ],
        axis=0,
    )

    y = transform(z)
    actual = transform.log_abs_det_jacobian(z, y)

    u = standard_normal.cdf(z)
    expected = mock_interpolator.log_abs_dxdu(u) + standard_normal.log_prob(z)

    interior_mask = jnp.abs(z) <= 4.0
    _assert_allclose(
        actual[interior_mask],
        expected[interior_mask],
        rtol=2e-5,
        atol=2e-6,
        msg="Normal transform Jacobian must follow log|dy/dz| = log|dx/du| + log phi(z) in interior.",
    )

    assert jnp.all(jnp.isfinite(actual)), "Normal transform Jacobian must stay finite for extreme z values."


def test_boundary_inputs_are_stable(mock_interpolator):
    uniform_transform = _instantiate_uniform_transform(mock_interpolator)
    normal_transform = _instantiate_normal_transform(mock_interpolator)

    u_boundary = jnp.array([0.0, 1e-15, 1e-10, 0.5, 1.0 - 1e-10, 1.0 - 1e-15, 1.0], dtype=jnp.float32)
    y_u = uniform_transform(u_boundary)
    u_recovered = uniform_transform._inverse(y_u)
    ladj_u = uniform_transform.log_abs_det_jacobian(u_boundary, y_u)

    assert jnp.all(jnp.isfinite(y_u)), "Uniform transform forward outputs must be finite near boundaries."
    assert jnp.all(jnp.isfinite(u_recovered)), "Uniform transform inverse outputs must be finite near boundaries."
    assert jnp.all(jnp.isfinite(ladj_u)), "Uniform transform Jacobian must be finite near boundaries."
    assert jnp.all((u_recovered >= 0.0) & (u_recovered <= 1.0)), "Uniform inverse outputs must stay in [0, 1]."

    z_extreme = jnp.array([-12.0, -9.0, -8.0, 0.0, 8.0, 9.0, 12.0], dtype=jnp.float32)
    y_z = normal_transform(z_extreme)
    z_recovered = normal_transform._inverse(y_z)
    ladj_z = normal_transform.log_abs_det_jacobian(z_extreme, y_z)

    assert jnp.all(jnp.isfinite(y_z)), "Normal transform forward outputs must be finite for extreme z inputs."
    assert jnp.all(jnp.isfinite(z_recovered)), "Normal transform inverse outputs must be finite for extreme z inputs."
    assert jnp.all(jnp.isfinite(ladj_z)), "Normal transform Jacobian must be finite for extreme z inputs."


def test_jit_and_vmap_compatibility(mock_interpolator):
    uniform_transform = _instantiate_uniform_transform(mock_interpolator)
    normal_transform = _instantiate_normal_transform(mock_interpolator)

    u = jnp.linspace(0.01, 0.99, 64, dtype=jnp.float32)
    z = jnp.linspace(-5.0, 5.0, 64, dtype=jnp.float32)

    uniform_y = uniform_transform(u)
    normal_y = normal_transform(z)

    _assert_allclose(jax.jit(lambda x: uniform_transform(x))(u), uniform_y, rtol=1e-6, atol=1e-7)
    _assert_allclose(jax.jit(lambda y: uniform_transform._inverse(y))(uniform_y), u, rtol=1e-5, atol=1e-6)
    _assert_allclose(
        jax.jit(lambda x, y: uniform_transform.log_abs_det_jacobian(x, y))(u, uniform_y),
        uniform_transform.log_abs_det_jacobian(u, uniform_y),
        rtol=1e-6,
        atol=1e-6,
    )

    _assert_allclose(jax.jit(lambda x: normal_transform(x))(z), normal_y, rtol=1e-6, atol=1e-7)
    _assert_allclose(
        jax.jit(lambda y: normal_transform._inverse(y))(normal_y),
        normal_transform._inverse(normal_y),
        rtol=1e-6,
        atol=1e-6,
    )
    _assert_allclose(
        jax.jit(lambda x, y: normal_transform.log_abs_det_jacobian(x, y))(z, normal_y),
        normal_transform.log_abs_det_jacobian(z, normal_y),
        rtol=1e-6,
        atol=1e-6,
    )

    batched_u = jnp.stack([u, 0.5 * u + 0.25, 0.9 * u + 0.05], axis=0)
    batched_z = jnp.stack([z, z + 0.75, z - 0.5], axis=0)

    uniform_batched_y = jax.vmap(uniform_transform)(batched_u)
    normal_batched_y = jax.vmap(normal_transform)(batched_z)

    assert uniform_batched_y.shape == batched_u.shape
    assert normal_batched_y.shape == batched_z.shape
    _assert_allclose(jax.vmap(uniform_transform._inverse)(uniform_batched_y), batched_u, rtol=1e-5, atol=1e-6)
    _assert_allclose(
        jax.vmap(normal_transform._inverse)(normal_batched_y),
        normal_transform._inverse(normal_batched_y),
        rtol=1e-6,
        atol=1e-6,
    )


def test_tree_flatten_unflatten_roundtrip(mock_interpolator):
    uniform_transform = _instantiate_uniform_transform(mock_interpolator)
    normal_transform = _instantiate_normal_transform(mock_interpolator)

    for transform, sample in (
        (uniform_transform, jnp.linspace(0.05, 0.95, 33, dtype=jnp.float32)),
        (normal_transform, jnp.linspace(-3.0, 3.0, 33, dtype=jnp.float32)),
    ):
        for method_name in ("__call__", "_inverse", "log_abs_det_jacobian"):
            assert hasattr(transform, method_name), f"Transform is missing required method `{method_name}`."
            assert callable(getattr(transform, method_name)), f"`{method_name}` must be callable."

        assert transform.domain is not None, "`domain` must be set."
        assert transform.codomain is not None, "`codomain` must be set."

        children, aux_data = transform.tree_flatten()
        rebuilt_from_method = transform.__class__.tree_unflatten(aux_data, children)

        expected = transform(sample)
        rebuilt_output = rebuilt_from_method(sample)
        _assert_allclose(rebuilt_output, expected, rtol=1e-6, atol=1e-7)

        leaves, treedef = jax.tree_util.tree_flatten(transform)
        rebuilt_from_jax = jax.tree_util.tree_unflatten(treedef, leaves)
        _assert_allclose(rebuilt_from_jax(sample), expected, rtol=1e-6, atol=1e-7)

    assert uniform_transform.domain is unit_interval
    assert isinstance(normal_transform.domain, _Real)
