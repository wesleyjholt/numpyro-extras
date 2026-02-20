import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions import Exponential, Logistic, Normal
from numpyro.distributions.constraints import _Real, unit_interval

from numpyro_extras.transforms import (
    NormalToDistributionTransform,
    UniformToDistributionTransform,
)


RTOL = 1e-5
ATOL = 1e-6
ROUNDTRIP_RTOL = 1e-3
ROUNDTRIP_ATOL = 5e-5
UNIFORM_X = jnp.array([1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-4])
UNIFORM_EDGE_X = jnp.array([1e-8, 1e-6, 1 - 1e-6, 1 - 1e-8])
NORMAL_X = jnp.array([-4.0, -2.0, -0.1, 0.0, 0.1, 2.0, 4.0])
NORMAL_EDGE_X = jnp.array([-6.0, -5.0, 5.0, 6.0])

DIST_FACTORIES = [
    pytest.param(lambda: Normal(loc=0.7, scale=1.3), id="normal"),
    pytest.param(lambda: Logistic(loc=-0.5, scale=2.0), id="logistic"),
    pytest.param(lambda: Exponential(rate=1.7), id="exponential"),
]


def _assert_allclose(actual, expected, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


def _assert_allclose_roundtrip(actual, expected):
    _assert_allclose(actual, expected, rtol=ROUNDTRIP_RTOL, atol=ROUNDTRIP_ATOL)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_forward_inverse_roundtrip(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    y = transform(UNIFORM_X)
    recovered_x = transform._inverse(y)

    _assert_allclose_roundtrip(recovered_x, UNIFORM_X)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_inverse_forward_roundtrip(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    y = dist.icdf(UNIFORM_X)
    recovered_y = transform(transform._inverse(y))

    _assert_allclose_roundtrip(recovered_y, y)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_log_abs_det_jacobian_matches_log_prob(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    y = transform(UNIFORM_X)
    actual = transform.log_abs_det_jacobian(UNIFORM_X, y)
    expected = dist.log_prob(y)

    _assert_allclose(actual, expected)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_domain_codomain_and_parameter_metadata(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    assert transform.domain is unit_interval
    assert type(transform.codomain) is type(dist.support)
    assert transform.codomain.event_dim == dist.support.event_dim

    expected_names = sorted(dist.get_args().keys())
    assert transform.distribution_parameter_names == expected_names

    expected_values = [dist.get_args()[name] for name in expected_names]
    assert len(transform.distribution_parameter_values) == len(expected_values)
    for actual, expected in zip(transform.distribution_parameter_values, expected_values, strict=True):
        _assert_allclose(actual, expected)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_tree_flatten_unflatten_roundtrip(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    children, aux_data = transform.tree_flatten()
    rebuilt_from_method = UniformToDistributionTransform.tree_unflatten(aux_data, children)
    _assert_allclose(rebuilt_from_method(UNIFORM_X), transform(UNIFORM_X))

    leaves, treedef = jax.tree_util.tree_flatten(transform)
    rebuilt_from_jax = jax.tree_util.tree_unflatten(treedef, leaves)
    _assert_allclose(rebuilt_from_jax(UNIFORM_X), transform(UNIFORM_X))


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_jit_and_vmap_compatibility(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    y = transform(UNIFORM_X)
    _assert_allclose(jax.jit(lambda x: transform(x))(UNIFORM_X), y)
    _assert_allclose(jax.jit(lambda z: transform._inverse(z))(y), UNIFORM_X)
    _assert_allclose(
        jax.jit(lambda x, y_val: transform.log_abs_det_jacobian(x, y_val))(UNIFORM_X, y),
        transform.log_abs_det_jacobian(UNIFORM_X, y),
    )

    batched_x = jnp.stack([UNIFORM_X, UNIFORM_X * 0.8 + 0.1], axis=0)
    batched_y = jax.vmap(transform)(batched_x)
    _assert_allclose(jax.vmap(transform._inverse)(batched_y), batched_x)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_monotonicity(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    y = transform(UNIFORM_X)
    assert jnp.all(jnp.diff(y) >= -ATOL)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_uniform_edge_domain_stability(dist_factory):
    dist = dist_factory()
    transform = UniformToDistributionTransform(dist)

    y = transform(UNIFORM_EDGE_X)
    recovered = transform._inverse(y)
    ladj = transform.log_abs_det_jacobian(UNIFORM_EDGE_X, y)

    assert jnp.all(~jnp.isnan(y))
    assert jnp.all(~jnp.isnan(recovered))
    assert jnp.all(~jnp.isnan(ladj))
    assert jnp.all((recovered >= 0.0) & (recovered <= 1.0))


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_forward_inverse_roundtrip(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)

    y = transform(NORMAL_X)
    recovered_x = transform._inverse(y)

    _assert_allclose_roundtrip(recovered_x, NORMAL_X)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_inverse_forward_roundtrip(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)
    standard_normal = Normal(0.0, 1.0)

    y = dist.icdf(standard_normal.cdf(NORMAL_X))
    recovered_y = transform(transform._inverse(y))

    _assert_allclose_roundtrip(recovered_y, y)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_log_abs_det_jacobian_matches_formula(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)
    standard_normal = Normal(0.0, 1.0)

    y = transform(NORMAL_X)
    actual = transform.log_abs_det_jacobian(NORMAL_X, y)
    expected = dist.log_prob(y) - standard_normal.log_prob(NORMAL_X)

    _assert_allclose(actual, expected)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_domain_codomain_and_parameter_metadata(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)

    assert isinstance(transform.domain, _Real)
    assert type(transform.codomain) is type(dist.support)
    assert transform.codomain.event_dim == dist.support.event_dim

    expected_names = sorted(dist.get_args().keys())
    assert transform.distribution_parameter_names == expected_names

    expected_values = [dist.get_args()[name] for name in expected_names]
    assert len(transform.distribution_parameter_values) == len(expected_values)
    for actual, expected in zip(transform.distribution_parameter_values, expected_values, strict=True):
        _assert_allclose(actual, expected)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_tree_flatten_unflatten_roundtrip(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)

    children, aux_data = transform.tree_flatten()
    rebuilt_from_method = NormalToDistributionTransform.tree_unflatten(aux_data, children)
    assert isinstance(rebuilt_from_method._standard_normal, Normal)
    _assert_allclose(rebuilt_from_method(NORMAL_X), transform(NORMAL_X))

    leaves, treedef = jax.tree_util.tree_flatten(transform)
    rebuilt_from_jax = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt_from_jax._standard_normal, Normal)
    _assert_allclose(rebuilt_from_jax(NORMAL_X), transform(NORMAL_X))


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_jit_and_vmap_compatibility(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)

    y = transform(NORMAL_X)
    _assert_allclose(jax.jit(lambda x: transform(x))(NORMAL_X), y)
    _assert_allclose_roundtrip(jax.jit(lambda z: transform._inverse(z))(y), NORMAL_X)
    _assert_allclose(
        jax.jit(lambda x, y_val: transform.log_abs_det_jacobian(x, y_val))(NORMAL_X, y),
        transform.log_abs_det_jacobian(NORMAL_X, y),
    )

    batched_x = jnp.stack([NORMAL_X, NORMAL_X + 0.5], axis=0)
    batched_y = jax.vmap(transform)(batched_x)
    _assert_allclose_roundtrip(jax.vmap(transform._inverse)(batched_y), batched_x)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_monotonicity(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)

    y = transform(NORMAL_X)
    assert jnp.all(jnp.diff(y) >= -ATOL)


@pytest.mark.parametrize("dist_factory", DIST_FACTORIES)
def test_normal_tail_stability(dist_factory):
    dist = dist_factory()
    transform = NormalToDistributionTransform(dist)

    y = transform(NORMAL_EDGE_X)
    recovered = transform._inverse(y)
    ladj = transform.log_abs_det_jacobian(NORMAL_EDGE_X, y)

    assert jnp.all(~jnp.isnan(y))
    assert jnp.all(~jnp.isnan(recovered))
    assert jnp.all(~jnp.isnan(ladj))
    assert jnp.all(jnp.diff(y) >= 0.0)


def test_broadcasting_shapes_for_uniform_and_normal_transforms():
    dist = Normal(loc=jnp.array([-1.0, 0.0, 1.0]), scale=1.0)

    uniform_transform = UniformToDistributionTransform(dist)
    x_uniform = jnp.array([[0.2], [0.8]])
    y_uniform = uniform_transform(x_uniform)
    assert y_uniform.shape == (2, 3)
    inverse_uniform = uniform_transform._inverse(y_uniform)
    assert inverse_uniform.shape == (2, 3)
    _assert_allclose(inverse_uniform, jnp.broadcast_to(x_uniform, (2, 3)))
    ladj_uniform = uniform_transform.log_abs_det_jacobian(x_uniform, y_uniform)
    assert ladj_uniform.shape == (2, 3)

    normal_transform = NormalToDistributionTransform(dist)
    x_normal = jnp.array([[-1.0], [0.5]])
    y_normal = normal_transform(x_normal)
    assert y_normal.shape == (2, 3)
    inverse_normal = normal_transform._inverse(y_normal)
    assert inverse_normal.shape == (2, 3)
    _assert_allclose(inverse_normal, jnp.broadcast_to(x_normal, (2, 3)))
    ladj_normal = normal_transform.log_abs_det_jacobian(x_normal, y_normal)
    assert ladj_normal.shape == (2, 3)
