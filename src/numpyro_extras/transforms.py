__all__ = ["UniformToDistributionTransform", "NormalToDistributionTransform", "real_array"]

import jax.nn as jnn
import jax.numpy as jnp
from numpyro.distributions import Distribution, Normal
from numpyro.distributions.transforms import Transform
from numpyro.distributions.constraints import unit_interval, _Real, _IndependentConstraint
from typing import NewType, Optional

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