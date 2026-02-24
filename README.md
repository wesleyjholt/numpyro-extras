# numpyro-extras
A collection of nice-to-have utilities that build off the NumPyro probabilistic programming language

## Mixture Transform Builder

```python
import jax.numpy as jnp
from numpyro.distributions import Categorical, MixtureSameFamily, Normal
from numpyro_extras.mixture_transform_builder import (
    MixtureTransformBuildConfig,
    build_mixture_transform,
)

mixture = MixtureSameFamily(
    Categorical(probs=jnp.array([0.55, 0.45])),
    Normal(loc=jnp.array([-1.5, 2.0]), scale=jnp.array([0.9, 1.2])),
)

cfg = MixtureTransformBuildConfig()
result = build_mixture_transform(
    base="normal",
    mixture=mixture,
    build_cfg=cfg,
)
transform = result.transform
diagnostics = result.diagnostics

z = jnp.linspace(-3.0, 3.0, 9)
y = transform(z)
```
