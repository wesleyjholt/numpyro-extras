__all__ = [
    "UniformToDistributionTransform",
    "NormalToDistributionTransform",
    "UniformToMixtureTransform",
    "NormalToMixtureTransform",
    "real_array",
]

import jax.nn as jnn
import jax.numpy as jnp
from numpyro.distributions import Distribution, Normal
from numpyro.distributions.transforms import Transform
from numpyro.distributions.constraints import unit_interval, _Real, _IndependentConstraint
from typing import Any, Mapping, NewType, Optional

NumLike = NewType("NumLike", float)
PyTree = NewType("PyTree", object)


# Define new domains/constraints, following NumPyro's constraint system.
# See https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/constraints.py

class _RealArray(_IndependentConstraint):
    def __init__(self, ndim) -> None:
        super().__init__(_Real(), ndim)

real_array = _RealArray


# Define new transforms, following NumPyro's transform system.
# See https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py

class UniformToDistributionTransform(Transform):
    """Transform from Uniform(0, 1) to a given distribution using the inverse CDF."""

    def __init__(self, distribution: Distribution):
        self._dist_cls = distribution.__class__
        self._dist_params = distribution.get_args()
        self.distribution = distribution

    def __call__(self, x):
        return self.distribution.icdf(x)

    def _inverse(self, y):
        return self.distribution.cdf(y)
    
    @property
    def domain(self):
        return unit_interval
    
    @property
    def codomain(self):
        return self.distribution.support
    
    @property
    def distribution_parameter_names(self):
        return sorted(list(self._dist_params.keys()))
    
    @property
    def distribution_parameter_values(self):
        return [self._dist_params[k] for k in self.distribution_parameter_names]
    
    def tree_flatten(self):
        return self.distribution_parameter_values, (
            self.distribution_parameter_names,
            {"dist_cls": self._dist_cls},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        param_names, meta = aux_data
        dist_cls = meta.get("dist_cls")
        dist_params = dict(zip(param_names, params, strict=False))
        distribution = dist_cls(**dist_params)
        obj = cls.__new__(cls)
        obj._dist_cls = dist_cls
        obj._dist_params = dist_params
        obj.distribution = distribution
        return obj
    
    def log_abs_det_jacobian(
        self, x: NumLike, y: NumLike, intermediates: Optional[PyTree] = None
    ) -> NumLike:
        """Compute the log absolute determinant of the Jacobian."""
        # See https://num.pyro.ai/en/latest/distributions.html#transform
        
        pdf_vals = self.distribution.log_prob(y)
        # Since we are transforming from Uniform(0, 1), the derivative of the CDF is the PDF
        return pdf_vals - 0.0  # log(1) = 0


class NormalToDistributionTransform(Transform):
    """Transform from standard Normal to a distribution using the inverse CDF."""

    def __init__(self, distribution: Distribution):
        self._dist_cls = distribution.__class__
        self._dist_params = distribution.get_args()
        self.distribution = distribution
        self._standard_normal = Normal(0.0, 1.0)

    def __call__(self, x):
        return self.distribution.icdf(self._standard_normal.cdf(x))

    def _inverse(self, y):
        return self._standard_normal.icdf(self.distribution.cdf(y))

    @property
    def domain(self):
        return _Real()

    @property
    def codomain(self):
        return self.distribution.support

    @property
    def distribution_parameter_names(self):
        return sorted(list(self._dist_params.keys()))

    @property
    def distribution_parameter_values(self):
        return [self._dist_params[k] for k in self.distribution_parameter_names]

    def tree_flatten(self):
        return self.distribution_parameter_values, (
            self.distribution_parameter_names,
            {"dist_cls": self._dist_cls},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        param_names, meta = aux_data
        dist_cls = meta.get("dist_cls")
        dist_params = dict(zip(param_names, params, strict=False))
        distribution = dist_cls(**dist_params)
        obj = cls.__new__(cls)
        obj._dist_cls = dist_cls
        obj._dist_params = dist_params
        obj.distribution = distribution
        obj._standard_normal = Normal(0.0, 1.0)
        return obj

    def log_abs_det_jacobian(
        self, x: NumLike, y: NumLike, intermediates: Optional[PyTree] = None
    ) -> NumLike:
        """Compute the log absolute determinant of the Jacobian."""
        target_log_pdf = self.distribution.log_prob(y)
        normal_log_pdf = self._standard_normal.log_prob(x)
        return target_log_pdf - normal_log_pdf



def _shifted_scaled_tanh(x, x_shift=0.0, y_shift=0.0, x_scale=1.0, y_scale=1.0):
    return y_shift + y_scale * jnn.tanh(x_scale * (x - x_shift))


def _shifted_scaled_arctanh(x, x_shift=0.0, y_shift=0.0, x_scale=1.0, y_scale=1.0):
    return y_shift + y_scale * jnp.arctanh(x_scale * (x - x_shift))


# Mixture-related transforms below


def _coerce_transform_cfg(transform_cfg: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if transform_cfg is None:
        cfg = {"clip_u_eps": 1e-10, "validate_args": False}
    elif isinstance(transform_cfg, Mapping):
        cfg = {
            "clip_u_eps": float(transform_cfg.get("clip_u_eps", 1e-10)),
            "validate_args": bool(transform_cfg.get("validate_args", False)),
        }
    else:
        raise TypeError("`transform_cfg` must be None or dict-like.")

    if not (0.0 < cfg["clip_u_eps"] < 0.5):
        raise ValueError("`transform_cfg['clip_u_eps']` must satisfy 0 < clip_u_eps < 0.5.")
    return cfg


def _clip_u(u, clip_u_eps):
    u_arr = jnp.asarray(u)
    zero = jnp.asarray(0.0, dtype=u_arr.dtype)
    one = jnp.asarray(1.0, dtype=u_arr.dtype)
    eps = jnp.asarray(clip_u_eps, dtype=u_arr.dtype)
    lo_strict = jnp.nextafter(zero, one)
    hi_strict = jnp.nextafter(one, zero)
    lo = jnp.maximum(eps, lo_strict)
    hi = jnp.minimum(one - eps, hi_strict)
    return jnp.clip(u_arr, lo, hi)


def _safe_log_abs_dxdu(interpolator, u):
    if hasattr(interpolator, "log_abs_dxdu"):
        return jnp.asarray(interpolator.log_abs_dxdu(u), dtype=jnp.asarray(u).dtype)

    if hasattr(interpolator, "dxdu"):
        dxdu = jnp.asarray(interpolator.dxdu(u), dtype=jnp.asarray(u).dtype)
        tiny = jnp.finfo(dxdu.dtype).tiny
        return jnp.log(jnp.maximum(jnp.abs(dxdu), tiny))

    raise TypeError("`interpolator` must provide `log_abs_dxdu(u)` or `dxdu(u)`.")


class UniformToMixtureTransform(Transform):
    """Transform Uniform(0, 1) inputs to a mixture support using an interpolated inverse-CDF."""

    def __init__(
        self,
        interpolator,
        transform_cfg: Optional[Mapping[str, Any]] = None,
    ):
        cfg = _coerce_transform_cfg(transform_cfg)
        self.interpolator = interpolator
        self._clip_u_eps = float(cfg["clip_u_eps"])
        self._validate_args = bool(cfg["validate_args"])

    def __call__(self, x):
        u = _clip_u(x, self._clip_u_eps)
        return self.interpolator.icdf(u)

    def _inverse(self, y):
        u = self.interpolator.cdf(y)
        return _clip_u(u, self._clip_u_eps)

    @property
    def domain(self):
        return unit_interval

    @property
    def codomain(self):
        return _Real()

    def tree_flatten(self):
        return (self.interpolator,), {
            "clip_u_eps": self._clip_u_eps,
            "validate_args": self._validate_args,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        (interpolator,) = params
        return cls(interpolator=interpolator, transform_cfg=aux_data)

    def log_abs_det_jacobian(
        self, x: NumLike, y: NumLike, intermediates: Optional[PyTree] = None
    ) -> NumLike:
        del y, intermediates
        u = _clip_u(x, self._clip_u_eps)
        return _safe_log_abs_dxdu(self.interpolator, u)


class NormalToMixtureTransform(Transform):
    """Transform Normal(0, 1) inputs to a mixture support via Normal CDF then inverse mixture CDF."""

    def __init__(
        self,
        interpolator,
        transform_cfg: Optional[Mapping[str, Any]] = None,
    ):
        cfg = _coerce_transform_cfg(transform_cfg)
        self.interpolator = interpolator
        self._clip_u_eps = float(cfg["clip_u_eps"])
        self._validate_args = bool(cfg["validate_args"])
        self._standard_normal = Normal(0.0, 1.0)
        self._tail_switch_z = 2.0
        self._tail_logit_slope = 0.6

    def _tail_logit_params(self, dtype):
        z_switch = jnp.asarray(self._tail_switch_z, dtype=dtype)
        beta = jnp.asarray(self._tail_logit_slope, dtype=dtype)
        u_switch = _clip_u(self._standard_normal.cdf(z_switch), self._clip_u_eps)
        t_switch = jnp.log(u_switch) - jnp.log1p(-u_switch)
        return z_switch, beta, u_switch, t_switch

    def _normal_base_cdf(self, z):
        z_arr = jnp.asarray(z)
        u_phi = _clip_u(self._standard_normal.cdf(z_arr), self._clip_u_eps)
        z_switch, beta, _, t_switch = self._tail_logit_params(z_arr.dtype)

        u_right = jnn.sigmoid(t_switch + beta * (z_arr - z_switch))
        u_left = jnn.sigmoid(-t_switch + beta * (z_arr + z_switch))

        u = jnp.where(z_arr > z_switch, u_right, u_phi)
        u = jnp.where(z_arr < -z_switch, u_left, u)
        return _clip_u(u, self._clip_u_eps)

    def _normal_base_icdf(self, u):
        u_arr = _clip_u(u, self._clip_u_eps)
        z_switch, beta, u_switch, t_switch = self._tail_logit_params(u_arr.dtype)

        logit_u = jnp.log(u_arr) - jnp.log1p(-u_arr)
        z_phi = self._standard_normal.icdf(u_arr)
        z_right = z_switch + (logit_u - t_switch) / beta
        z_left = -z_switch + (logit_u + t_switch) / beta

        z = jnp.where(u_arr > u_switch, z_right, z_phi)
        z = jnp.where(u_arr < (1.0 - u_switch), z_left, z)
        return z

    def __call__(self, x):
        z = jnp.asarray(x)
        u = self._normal_base_cdf(z)
        return self.interpolator.icdf(u)

    def _inverse(self, y):
        u = _clip_u(self.interpolator.cdf(y), self._clip_u_eps)
        return self._normal_base_icdf(u)

    @property
    def domain(self):
        return _Real()

    @property
    def codomain(self):
        return _Real()

    def tree_flatten(self):
        return (self.interpolator,), {
            "clip_u_eps": self._clip_u_eps,
            "validate_args": self._validate_args,
            "tail_switch_z": self._tail_switch_z,
            "tail_logit_slope": self._tail_logit_slope,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        (interpolator,) = params
        transform_cfg = {
            "clip_u_eps": aux_data["clip_u_eps"],
            "validate_args": aux_data["validate_args"],
        }
        obj = cls(interpolator=interpolator, transform_cfg=transform_cfg)
        obj._tail_switch_z = float(aux_data.get("tail_switch_z", 2.0))
        obj._tail_logit_slope = float(aux_data.get("tail_logit_slope", 0.6))
        return obj

    def log_abs_det_jacobian(
        self, x: NumLike, y: NumLike, intermediates: Optional[PyTree] = None
    ) -> NumLike:
        del y, intermediates
        z = jnp.asarray(x)
        u = _clip_u(self._standard_normal.cdf(z), self._clip_u_eps)
        log_abs_dxdu = _safe_log_abs_dxdu(self.interpolator, u)
        return log_abs_dxdu + self._standard_normal.log_prob(z)
