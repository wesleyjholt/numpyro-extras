# numpyro-extras
A collection of nice-to-have utilities that build off the NumPyro probabilistic programming language

## Distribution Transform Builder

```python
import jax.numpy as jnp
from numpyro.distributions import Normal
from numpyro_extras.distribution_transform_builder import (
    DistributionTransformBuildConfig,
    build_distribution_transform,
)

target = Normal(loc=0.0, scale=1.0)

cfg = DistributionTransformBuildConfig()
result = build_distribution_transform(
    base="normal",
    distribution=target,
    build_cfg=cfg,
)
transform = result.transform
diagnostics = result.diagnostics

z = jnp.linspace(-3.0, 3.0, 9)
y = transform(z)
```

`numpyro-extras` does not synthesize a generic CDF for arbitrary NumPyro
distributions. The target distribution is expected to provide a working `cdf`
and `log_prob` implementation, which typically comes from NumPyro itself. The
package's role here is to build and cache quantile-based transforms on top of
that interface.
