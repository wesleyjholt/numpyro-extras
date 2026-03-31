"""Step 04 manifest for Task 09 NumPyro transform wrappers.

Required module:
- `numpyro_extras.transforms` (expected file: `src/numpyro_extras/transforms.py`)

Required public API:
- `UniformToDistributionTransform`
- `NormalToDistributionTransform`

Behavioral pass gates:
- forward/inverse roundtrip for uniform and normal base transforms
- Jacobian correctness (finite-difference + chain-rule checks)
- boundary/clipping stability for u in {0, 1} and extreme normal inputs
- JAX `jit` + `vmap` compatibility and JAX-native traced paths
- pytree flatten/unflatten roundtrip
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from types import ModuleType, SimpleNamespace

import jax
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


MODULE_NAME = "numpyro_extras.transforms"
UNIFORM_TRANSFORM_NAME = "UniformToDistributionTransform"
NORMAL_TRANSFORM_NAME = "NormalToDistributionTransform"


@dataclasses.dataclass(frozen=True)
class _DeterministicInterpolator:
    scale: float = 1.7
    shift: float = -0.35
    eps: float = 1e-15

    def _clip_u(self, u):
        u = jnp.asarray(u, dtype=jnp.float64)
        return jnp.clip(u, self.eps, 1.0 - self.eps)

    def icdf(self, u):
        u = self._clip_u(u)
        return self.scale * (jnp.log(u) - jnp.log1p(-u)) + self.shift

    def cdf(self, x):
        x = jnp.asarray(x, dtype=jnp.float64)
        u = jax.nn.sigmoid((x - self.shift) / self.scale)
        return self._clip_u(u)

    def dxdu(self, u):
        u = self._clip_u(u)
        return self.scale / (u * (1.0 - u))

    def log_abs_dxdu(self, u):
        u = self._clip_u(u)
        return jnp.log(jnp.abs(self.scale)) - jnp.log(u) - jnp.log1p(-u)


def _import_transforms_module() -> ModuleType:
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.transforms` "
                "(expected at `src/numpyro_extras/transforms.py`)."
            )
        raise


def _get_transform_types():
    module = _import_transforms_module()
    uniform_type = getattr(module, UNIFORM_TRANSFORM_NAME, None)
    normal_type = getattr(module, NORMAL_TRANSFORM_NAME, None)

    if uniform_type is None:
        pytest.fail(
            "Missing `UniformToDistributionTransform` in `numpyro_extras.transforms`."
        )
    if normal_type is None:
        pytest.fail("Missing `NormalToDistributionTransform` in `numpyro_extras.transforms`.")
    if not callable(uniform_type):
        pytest.fail("`UniformToDistributionTransform` must be callable.")
    if not callable(normal_type):
        pytest.fail("`NormalToDistributionTransform` must be callable.")
    return uniform_type, normal_type


def _make_transform_cfg(**overrides):
    cfg = dict(
        clip_u_eps=1e-10,
        validate_args=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _build_uniform_transform(interpolator, transform_cfg):
    uniform_type, _ = _get_transform_types()
    try:
        return uniform_type(interpolator=interpolator, transform_cfg=transform_cfg)
    except TypeError:
        try:
            return uniform_type(interpolator, transform_cfg)
        except TypeError as exc:
            pytest.fail(
                "Expected constructor signature "
                "`UniformToDistributionTransform(interpolator=..., transform_cfg=...)` "
                f"or positional equivalent. Received TypeError: {exc}"
            )


def _build_normal_transform(interpolator, transform_cfg):
    _, normal_type = _get_transform_types()
    standard_normal = dist.Normal(
        loc=jnp.asarray(0.0, dtype=jnp.float64),
        scale=jnp.asarray(1.0, dtype=jnp.float64),
    )

    call_attempts = (
        lambda: normal_type(interpolator=interpolator, transform_cfg=transform_cfg),
        lambda: normal_type(
            interpolator=interpolator,
            transform_cfg=transform_cfg,
            standard_normal=standard_normal,
        ),
        lambda: normal_type(interpolator, transform_cfg),
        lambda: normal_type(interpolator, transform_cfg, standard_normal),
        lambda: normal_type(interpolator, standard_normal, transform_cfg),
        lambda: normal_type(interpolator, standard_normal),
    )
    for call in call_attempts:
        try:
            return call()
        except TypeError:
            continue

    pytest.fail(
        "Expected constructor signature "
        "`NormalToDistributionTransform(interpolator=..., transform_cfg=...)` "
        "or an equivalent form optionally accepting a standard normal distribution."
    )


@pytest.fixture
def deterministic_interpolator() -> _DeterministicInterpolator:
    return _DeterministicInterpolator()


def test_transform_api_contract(deterministic_interpolator):
    uniform_type, normal_type = _get_transform_types()

    uniform_sig = inspect.signature(uniform_type)
    normal_sig = inspect.signature(normal_type)
    assert "interpolator" in uniform_sig.parameters, (
        "`UniformToDistributionTransform` must accept `interpolator`."
    )
    assert "interpolator" in normal_sig.parameters, (
        "`NormalToDistributionTransform` must accept `interpolator`."
    )
    assert "transform_cfg" in uniform_sig.parameters, (
        "`UniformToDistributionTransform` must accept `transform_cfg`."
    )
    assert "transform_cfg" in normal_sig.parameters, (
        "`NormalToDistributionTransform` must accept `transform_cfg`."
    )

    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    uniform_transform = _build_uniform_transform(
        deterministic_interpolator, transform_cfg
    )
    normal_transform = _build_normal_transform(deterministic_interpolator, transform_cfg)

    for transform in (uniform_transform, normal_transform):
        assert callable(getattr(transform, "__call__", None))
        assert callable(getattr(transform, "_inverse", None))
        assert callable(getattr(transform, "log_abs_det_jacobian", None))
        assert callable(getattr(transform, "tree_flatten", None))
        assert callable(getattr(type(transform), "tree_unflatten", None))

    assert uniform_transform.domain is constraints.unit_interval, (
        "Uniform transform domain must be `constraints.unit_interval`."
    )
    assert uniform_transform.codomain is constraints.real, (
        "Uniform transform codomain must be `constraints.real`."
    )
    assert normal_transform.domain is constraints.real, (
        "Normal transform domain must be `constraints.real`."
    )
    assert normal_transform.codomain is constraints.real, (
        "Normal transform codomain must be `constraints.real`."
    )

    u_scalar = jnp.asarray(0.73, dtype=jnp.float64)
    z_scalar = jnp.asarray(-1.25, dtype=jnp.float64)
    y_u_scalar = uniform_transform(u_scalar)
    y_z_scalar = normal_transform(z_scalar)
    assert jnp.shape(y_u_scalar) == ()
    assert jnp.shape(y_z_scalar) == ()
    assert jnp.shape(uniform_transform._inverse(y_u_scalar)) == ()
    assert jnp.shape(normal_transform._inverse(y_z_scalar)) == ()
    assert jnp.shape(uniform_transform.log_abs_det_jacobian(u_scalar, y_u_scalar)) == ()
    assert jnp.shape(normal_transform.log_abs_det_jacobian(z_scalar, y_z_scalar)) == ()

    u_batch = jnp.array([0.2, 0.5, 0.8], dtype=jnp.float64)
    z_batch = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float64)
    y_u_batch = uniform_transform(u_batch)
    y_z_batch = normal_transform(z_batch)
    assert y_u_batch.shape == u_batch.shape
    assert y_z_batch.shape == z_batch.shape
    assert uniform_transform._inverse(y_u_batch).shape == u_batch.shape
    assert normal_transform._inverse(y_z_batch).shape == z_batch.shape
    assert uniform_transform.log_abs_det_jacobian(u_batch, y_u_batch).shape == u_batch.shape
    assert normal_transform.log_abs_det_jacobian(z_batch, y_z_batch).shape == z_batch.shape


def test_uniform_transform_roundtrip(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    uniform_transform = _build_uniform_transform(deterministic_interpolator, transform_cfg)

    key = jax.random.PRNGKey(0)
    u_grid = jnp.linspace(1e-10, 1.0 - 1e-10, 2049, dtype=jnp.float64)
    u_random = jax.random.uniform(
        key,
        shape=(2049,),
        minval=1e-10,
        maxval=1.0 - 1e-10,
        dtype=jnp.float64,
    )
    u = jnp.concatenate([u_grid, u_random], axis=0)
    u_roundtrip = uniform_transform._inverse(uniform_transform(u))
    err = jnp.abs(u_roundtrip - u)

    interior = (u > 1e-4) & (u < 1.0 - 1e-4)
    interior_max_err = float(jnp.max(err[interior]))
    tail_max_err = float(jnp.max(err[~interior]))
    assert interior_max_err < 1e-5, (
        f"Uniform roundtrip interior tolerance violated: max_err={interior_max_err:.3e}"
    )
    assert tail_max_err < 1e-4, (
        f"Uniform roundtrip tail tolerance violated: max_err={tail_max_err:.3e}"
    )


def test_normal_transform_roundtrip(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    normal_transform = _build_normal_transform(deterministic_interpolator, transform_cfg)
    standard_normal = dist.Normal(
        loc=jnp.asarray(0.0, dtype=jnp.float64),
        scale=jnp.asarray(1.0, dtype=jnp.float64),
    )

    key = jax.random.PRNGKey(1)
    z_grid = jnp.linspace(-6.5, 6.5, 2049, dtype=jnp.float64)
    z_random = jax.random.normal(key, shape=(2049,), dtype=jnp.float64) * 2.4
    z = jnp.concatenate([z_grid, z_random], axis=0)
    z_roundtrip = normal_transform._inverse(normal_transform(z))
    z_expected = standard_normal.icdf(
        jnp.clip(standard_normal.cdf(z), transform_cfg.clip_u_eps, 1.0 - transform_cfg.clip_u_eps)
    )
    err = jnp.abs(z_roundtrip - z_expected)

    interior = jnp.abs(z) <= 4.0
    interior_max_err = float(jnp.max(err[interior]))
    tail_max_err = float(jnp.max(err[~interior]))
    assert interior_max_err < 2e-5, (
        f"Normal roundtrip interior tolerance violated: max_err={interior_max_err:.3e}"
    )
    assert tail_max_err < 2e-4, (
        f"Normal roundtrip tail tolerance violated: max_err={tail_max_err:.3e}"
    )


def test_uniform_log_det_matches_finite_difference(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    uniform_transform = _build_uniform_transform(deterministic_interpolator, transform_cfg)

    u = jnp.linspace(2e-4, 1.0 - 2e-4, 2001, dtype=jnp.float64)
    h = jnp.asarray(2e-6, dtype=jnp.float64)

    y = uniform_transform(u)
    log_det = uniform_transform.log_abs_det_jacobian(u, y)

    y_plus = uniform_transform(u + h)
    y_minus = uniform_transform(u - h)
    dy_du_fd = (y_plus - y_minus) / (2.0 * h)

    model_abs_jac = jnp.exp(log_det)
    fd_abs_jac = jnp.abs(dy_du_fd)
    rel_err = jnp.abs(model_abs_jac - fd_abs_jac) / jnp.maximum(fd_abs_jac, 1e-12)
    max_rel_err = float(jnp.max(rel_err))

    assert jnp.all(jnp.isfinite(log_det)), "Uniform log-det Jacobian must be finite."
    assert jnp.all(jnp.isfinite(dy_du_fd)), "Finite-difference derivative must be finite."
    assert max_rel_err < 5e-3, (
        "Uniform Jacobian finite-difference agreement violated: "
        f"max_rel_err={max_rel_err:.3e}"
    )


def test_normal_log_det_matches_chain_rule(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    normal_transform = _build_normal_transform(deterministic_interpolator, transform_cfg)
    standard_normal = dist.Normal(
        loc=jnp.asarray(0.0, dtype=jnp.float64),
        scale=jnp.asarray(1.0, dtype=jnp.float64),
    )

    z = jnp.linspace(-6.0, 6.0, 2001, dtype=jnp.float64)
    y = normal_transform(z)
    log_det = normal_transform.log_abs_det_jacobian(z, y)

    u = standard_normal.cdf(z)
    u = jnp.clip(u, transform_cfg.clip_u_eps, 1.0 - transform_cfg.clip_u_eps)
    expected = deterministic_interpolator.log_abs_dxdu(u) + standard_normal.log_prob(z)
    max_abs_err = float(jnp.max(jnp.abs(log_det - expected)))

    assert jnp.all(jnp.isfinite(log_det)), "Normal log-det Jacobian must be finite."
    assert max_abs_err < 1e-7, (
        f"Normal chain-rule Jacobian mismatch: max_abs_err={max_abs_err:.3e}"
    )


def test_boundary_inputs_are_stable(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-10)
    uniform_transform = _build_uniform_transform(deterministic_interpolator, transform_cfg)
    normal_transform = _build_normal_transform(deterministic_interpolator, transform_cfg)
    standard_normal = dist.Normal(
        loc=jnp.asarray(0.0, dtype=jnp.float64),
        scale=jnp.asarray(1.0, dtype=jnp.float64),
    )

    eps = transform_cfg.clip_u_eps
    u_probe = jnp.array(
        [
            0.0,
            eps * 0.1,
            eps,
            1.0 - eps,
            1.0 - eps * 0.1,
            1.0,
        ],
        dtype=jnp.float64,
    )
    clipped_u = jnp.clip(u_probe, eps, 1.0 - eps)

    y_uniform = uniform_transform(u_probe)
    u_back = uniform_transform._inverse(y_uniform)
    ladj_uniform = uniform_transform.log_abs_det_jacobian(u_probe, y_uniform)
    expected_y_uniform = deterministic_interpolator.icdf(clipped_u)

    assert jnp.all(jnp.isfinite(y_uniform)), "Uniform forward map produced non-finite values."
    assert jnp.all(jnp.isfinite(u_back)), "Uniform inverse map produced non-finite values."
    assert jnp.all(jnp.isfinite(ladj_uniform)), "Uniform Jacobian produced non-finite values."
    assert jnp.allclose(u_back, clipped_u, atol=1e-9, rtol=1e-9), (
        "Uniform boundary handling must match deterministic clipping policy."
    )
    assert jnp.allclose(y_uniform, expected_y_uniform, atol=1e-12, rtol=1e-12), (
        "Uniform wrapper forward behavior must match interpolator outputs exactly "
        "after clipping."
    )

    z_extreme = jnp.array([-12.0, -10.0, -8.0, 8.0, 10.0, 12.0], dtype=jnp.float64)
    y_normal = normal_transform(z_extreme)
    z_back = normal_transform._inverse(y_normal)
    ladj_normal = normal_transform.log_abs_det_jacobian(z_extreme, y_normal)
    expected_y_normal = deterministic_interpolator.icdf(
        jnp.clip(standard_normal.cdf(z_extreme), eps, 1.0 - eps)
    )

    assert jnp.all(jnp.isfinite(y_normal)), "Normal forward map produced non-finite values."
    assert jnp.all(jnp.isfinite(z_back)), "Normal inverse map produced non-finite values."
    assert jnp.all(jnp.isfinite(ladj_normal)), "Normal Jacobian produced non-finite values."
    assert jnp.allclose(y_normal, expected_y_normal, atol=1e-12, rtol=1e-12), (
        "Normal wrapper must preserve interpolator boundary/tail semantics "
        "without extra wrapper-level switching."
    )


def test_jit_and_vmap_compatibility(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    uniform_transform = _build_uniform_transform(deterministic_interpolator, transform_cfg)
    normal_transform = _build_normal_transform(deterministic_interpolator, transform_cfg)

    u = jnp.linspace(1e-8, 1.0 - 1e-8, 1024, dtype=jnp.float64)
    z = jnp.linspace(-5.5, 5.5, 1024, dtype=jnp.float64)

    uniform_forward_jit = jax.jit(uniform_transform.__call__)
    uniform_inverse_jit = jax.jit(uniform_transform._inverse)
    uniform_ladj_jit = jax.jit(
        lambda x: uniform_transform.log_abs_det_jacobian(x, uniform_transform(x))
    )

    normal_forward_jit = jax.jit(normal_transform.__call__)
    normal_inverse_jit = jax.jit(normal_transform._inverse)
    normal_ladj_jit = jax.jit(
        lambda x: normal_transform.log_abs_det_jacobian(x, normal_transform(x))
    )

    y_u = uniform_forward_jit(u)
    u_back = uniform_inverse_jit(y_u)
    ladj_u = uniform_ladj_jit(u)
    y_z = normal_forward_jit(z)
    z_back = normal_inverse_jit(y_z)
    ladj_z = normal_ladj_jit(z)

    assert y_u.shape == u.shape
    assert u_back.shape == u.shape
    assert ladj_u.shape == u.shape
    assert y_z.shape == z.shape
    assert z_back.shape == z.shape
    assert ladj_z.shape == z.shape
    assert jnp.all(jnp.isfinite(y_u))
    assert jnp.all(jnp.isfinite(u_back))
    assert jnp.all(jnp.isfinite(ladj_u))
    assert jnp.all(jnp.isfinite(y_z))
    assert jnp.all(jnp.isfinite(z_back))
    assert jnp.all(jnp.isfinite(ladj_z))

    y_u_vmap = jax.vmap(lambda ui: uniform_transform(ui))(u)
    u_back_vmap = jax.vmap(lambda yi: uniform_transform._inverse(yi))(y_u)
    y_z_vmap = jax.vmap(lambda zi: normal_transform(zi))(z)
    z_back_vmap = jax.vmap(lambda yi: normal_transform._inverse(yi))(y_z)
    assert jnp.allclose(y_u_vmap, y_u, atol=1e-9, rtol=1e-9)
    assert jnp.allclose(u_back_vmap, u_back, atol=1e-9, rtol=1e-9)
    assert jnp.allclose(y_z_vmap, y_z, atol=1e-9, rtol=1e-9)
    assert jnp.allclose(z_back_vmap, z_back, atol=1e-9, rtol=1e-9)

    uniform_forward_jaxpr = jax.make_jaxpr(uniform_transform.__call__)(u)
    uniform_inverse_jaxpr = jax.make_jaxpr(uniform_transform._inverse)(y_u)
    uniform_ladj_jaxpr = jax.make_jaxpr(
        lambda x: uniform_transform.log_abs_det_jacobian(x, uniform_transform(x))
    )(u)
    normal_forward_jaxpr = jax.make_jaxpr(normal_transform.__call__)(z)
    normal_inverse_jaxpr = jax.make_jaxpr(normal_transform._inverse)(y_z)
    normal_ladj_jaxpr = jax.make_jaxpr(
        lambda x: normal_transform.log_abs_det_jacobian(x, normal_transform(x))
    )(z)

    assert len(uniform_forward_jaxpr.jaxpr.eqns) > 0
    assert len(uniform_inverse_jaxpr.jaxpr.eqns) > 0
    assert len(uniform_ladj_jaxpr.jaxpr.eqns) > 0
    assert len(normal_forward_jaxpr.jaxpr.eqns) > 0
    assert len(normal_inverse_jaxpr.jaxpr.eqns) > 0
    assert len(normal_ladj_jaxpr.jaxpr.eqns) > 0


def test_tree_flatten_unflatten_roundtrip(deterministic_interpolator):
    transform_cfg = _make_transform_cfg(clip_u_eps=1e-12)
    uniform_transform = _build_uniform_transform(deterministic_interpolator, transform_cfg)
    normal_transform = _build_normal_transform(deterministic_interpolator, transform_cfg)

    transform_specs = (
        (uniform_transform, jnp.linspace(1e-7, 1.0 - 1e-7, 257, dtype=jnp.float64)),
        (normal_transform, jnp.linspace(-5.0, 5.0, 257, dtype=jnp.float64)),
    )
    for transform, x in transform_specs:
        leaves, treedef = jax.tree_util.tree_flatten(transform)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        y_original = transform(x)
        y_rebuilt = rebuilt(x)
        inv_original = transform._inverse(y_original)
        inv_rebuilt = rebuilt._inverse(y_rebuilt)
        ladj_original = transform.log_abs_det_jacobian(x, y_original)
        ladj_rebuilt = rebuilt.log_abs_det_jacobian(x, y_rebuilt)

        assert jnp.allclose(y_rebuilt, y_original, atol=1e-12, rtol=1e-12), (
            "Pytree roundtrip changed forward-map behavior."
        )
        assert jnp.allclose(inv_rebuilt, inv_original, atol=1e-12, rtol=1e-12), (
            "Pytree roundtrip changed inverse-map behavior."
        )
        assert jnp.allclose(ladj_rebuilt, ladj_original, atol=1e-12, rtol=1e-12), (
            "Pytree roundtrip changed Jacobian behavior."
        )
