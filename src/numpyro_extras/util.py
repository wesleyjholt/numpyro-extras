__all__ = ["random_sample_like", "build_ppl_utilities"]

import numpyro
from numpyro.distributions import Distribution
import jax
from jax import tree
import jax.random as jr
import equinox as eqx
from typing import Optional, Any, Callable, Dict, Mapping, MutableMapping
from jaxtyping import PRNGKeyArray, PyTree 
from functools import partial
from dataclasses import dataclass


def _shifted_scaled_tanh(x, x_shift=0.0, y_shift=0.0, x_scale=1.0, y_scale=1.0):
    return y_shift + y_scale * jnn.tanh(x_scale * (x - x_shift))


def _shifted_scaled_arctanh(x, x_shift=0.0, y_shift=0.0, x_scale=1.0, y_scale=1.0):
    return y_shift + y_scale * jnp.arctanh(x_scale * (x - x_shift))


def random_sample_like(
    x: PyTree, 
    dist: Distribution, 
    key: PRNGKeyArray, 
    sample_shape: Optional[tuple] = ()
):
    """Generate a random sample from a distribution with the same PyTree structure and array leaf shapes as x.
    
    Arguments
    ---------
    x : PyTree
        The PyTree whose structure and leaf shapes are to be mimicked.
    dist : numpyro.distributions.Distribution
        A Numpyro distribution object that has a `sample` method.
    key : jax.random.PRNGKey
        A PRNG key for random number generation.
    sample_shape : tuple, optional
        The shape of the samples to be drawn from the distribution. Default is (). The shape of 
        each leaf will be extended by this shape.
    
    Returns
    -------
    PyTree
        A PyTree with the same structure and leaf shapes as `x`, where each leaf is a random sample
        drawn from the specified distribution.
    """
    inexact, other = eqx.partition(x, eqx.is_inexact_array)  # All leaves in `inexact` will be replaced with samples
    leaves, treedef = tree.flatten(inexact)
    keys = jr.split(key, len(leaves))
    sampled_leaves = [dist.sample(k, sample_shape=sample_shape + leaf.shape) for k, leaf in zip(keys, leaves)]
    sampled_inexact = tree.unflatten(treedef, sampled_leaves)
    return eqx.combine(sampled_inexact, other)

def _merge_model_kwargs(
    base: Optional[Mapping[str, Any]], overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    if overrides:
        merged.update(overrides)
    return merged


def _split_params_and_extras(
    x: MutableMapping[str, Any], parameter_names: list[str]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    params: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    for name, value in x.items():
        if name in parameter_names:
            params[name] = value
        else:
            extras[name] = value
    return params, extras


def _fill_missing_params(
    x: MutableMapping[str, Any], defaults: Mapping[str, Any], parameter_names: list[str]
) -> list[str]:
    keys_to_remove: list[str] = []
    for name in parameter_names:
        if name not in x:
            x[name] = defaults[name]
            keys_to_remove.append(name)
    return keys_to_remove


@dataclass(frozen=True)
class PPLUtilityGroup:
    """Utility group for a single NumPyro model (e.g., joint, prior, or likelihood).

    When used for a likelihood (from split-mode build_ppl_utilities), all callables
    accept an additional positional argument z (prior RV values) after the primary
    input; z's space matches the primary input (unconstrained or constrained).
    """

    parameter_names: list[str]
    valid_unconstrained_inputs: Dict[str, Any]
    valid_constrained_inputs: Dict[str, Any]

    log_probability_from_unconstrained_rvs: Callable[[PyTree], Any]
    log_probability_from_constrained_rvs: Callable[[PyTree], Any]

    tempered_log_probability_from_unconstrained_rvs: Callable[[PyTree], Any]
    tempered_log_probability_from_constrained_rvs: Callable[[PyTree], Any]

    log_probability_from_unconstrained_rvs_with_trace: Callable[[PyTree], Any]
    tempered_log_probability_from_unconstrained_rvs_with_trace: Callable[[PyTree], Any]

    constrain: Callable[[PyTree], Any]
    unconstrain: Callable[[PyTree], Any]

    simulate: Callable[..., Any]


def _build_utility_group(
    model: Callable[..., Any],
    init_model_kwargs: Mapping[str, Any],
    init_rng_key: jax.Array,
    tempering_param_kwarg: str = "tempering_param",
    jit_compile: bool = False,
    conditioned_on_z: bool = False,
    z_init: Optional[PyTree] = None,
    prior_group: Optional[PPLUtilityGroup] = None,
) -> PPLUtilityGroup:
    """Build the full utility-function family for one NumPyro callable.

    When conditioned_on_z=True, the model is called as model(z, **model_kwargs) and all
    helpers accept an additional positional argument z (prior RV values). z_init must
    be provided for initialization. prior_group is required when conditioned_on_z to
    convert z between constrained/unconstrained space to match each helper's convention.
    """

    if conditioned_on_z and z_init is None:
        raise ValueError("z_init is required when conditioned_on_z=True")
    if conditioned_on_z and prior_group is None:
        raise ValueError("prior_group is required when conditioned_on_z=True")

    model_args: tuple = (z_init,) if conditioned_on_z else ()
    init_params, potential_fn_gen, postprocess_fn_gen, _ = numpyro.infer.util.initialize_model(
        init_rng_key,
        model,
        model_args=model_args,
        model_kwargs=dict(init_model_kwargs or {}),
        dynamic_args=True,
    )

    parameter_names = list(init_params.z.keys())
    default_unconstrained = init_params.z
    default_constrained = numpyro.infer.util.constrain_fn(
        model, model_args, dict(init_model_kwargs or {}), init_params.z
    )

    if conditioned_on_z:
        pg = prior_group

        def _potential_fn_for(z: PyTree, model_kwargs: Mapping[str, Any]) -> Callable[[PyTree], Any]:
            return potential_fn_gen(z, **dict(model_kwargs))

        def log_probability_from_unconstrained_rvs(x: PyTree, z: PyTree, **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            z_c = pg.constrain(z, **mk)
            return -_potential_fn_for(z_c, mk)(x)

        def tempered_log_probability_from_unconstrained_rvs(
            x: PyTree, z: PyTree, *, temperature: float = 1.0, **model_kwargs: Any
        ):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            mk[tempering_param_kwarg] = temperature
            z_c = pg.constrain(z, **mk)
            return -_potential_fn_for(z_c, mk)(x)

        def _constrain_raw(x: PyTree, z: PyTree, model_kwargs: Mapping[str, Any]):
            postprocess_fn = postprocess_fn_gen(z, **dict(model_kwargs))
            return postprocess_fn(x)

        def _unconstrain_raw(theta: Mapping[str, Any], z: PyTree, model_kwargs: Mapping[str, Any]):
            return numpyro.infer.util.unconstrain_fn(
                model, (z,), dict(model_kwargs), theta
            )

        def constrain(x: MutableMapping[str, Any], z: PyTree, **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            z_c = pg.constrain(z, **mk)
            x = dict(x)
            keys_to_remove = _fill_missing_params(x, default_unconstrained, parameter_names)
            params, extras = _split_params_and_extras(x, parameter_names)
            constrained_params = _constrain_raw(params, z_c, mk)
            constrained_params.update(extras)
            for k in keys_to_remove:
                constrained_params.pop(k, None)
            return constrained_params

        def unconstrain(theta: MutableMapping[str, Any], z: PyTree, **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            theta = dict(theta)
            keys_to_remove = _fill_missing_params(theta, default_constrained, parameter_names)
            params, extras = _split_params_and_extras(theta, parameter_names)
            unconstrained_params = _unconstrain_raw(params, z, mk)
            unconstrained_params.update(extras)
            for k in keys_to_remove:
                unconstrained_params.pop(k, None)
            return unconstrained_params

        def log_probability_from_constrained_rvs(theta: PyTree, z: PyTree, **model_kwargs: Any):
            x = unconstrain(theta, z, **model_kwargs)
            z_u = pg.unconstrain(z, **model_kwargs)
            return log_probability_from_unconstrained_rvs(x, z_u, **model_kwargs)

        def tempered_log_probability_from_constrained_rvs(
            theta: PyTree, z: PyTree, *, temperature: float = 1.0, **model_kwargs: Any
        ):
            x = unconstrain(theta, z, **model_kwargs)
            z_u = pg.unconstrain(z, **model_kwargs)
            return tempered_log_probability_from_unconstrained_rvs(
                x, z_u, temperature=temperature, **model_kwargs
            )

        def simulate(rng_key: jax.Array, z: PyTree, **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            seeded = numpyro.handlers.seed(model, rng_key)
            return seeded(z, **mk)

        def log_probability_from_unconstrained_rvs_with_trace(
            x: PyTree, z: PyTree, **model_kwargs: Any
        ):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            z_c = pg.constrain(z, **mk)
            substituted_model = numpyro.handlers.substitute(
                model, substitute_fn=partial(numpyro.infer.util._unconstrain_reparam, x)
            )
            log_prob, trace = numpyro.infer.util.log_density(
                substituted_model, (z_c,), mk, {}
            )
            return log_prob, trace

        def tempered_log_probability_from_unconstrained_rvs_with_trace(
            x: PyTree, z: PyTree, *, temperature: float = 1.0, **model_kwargs: Any
        ):
            model_kwargs = {**model_kwargs, tempering_param_kwarg: temperature}
            return log_probability_from_unconstrained_rvs_with_trace(x, z, **model_kwargs)

    else:

        def _potential_fn_for(model_kwargs: Mapping[str, Any]) -> Callable[[PyTree], Any]:
            return potential_fn_gen(**dict(model_kwargs))

        def log_probability_from_unconstrained_rvs(x: PyTree, **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            return -_potential_fn_for(mk)(x)

        def tempered_log_probability_from_unconstrained_rvs(
            x: PyTree, *, temperature: float = 1.0, **model_kwargs: Any
        ):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            mk[tempering_param_kwarg] = temperature
            return -_potential_fn_for(mk)(x)

        def _constrain_raw(x: PyTree, model_kwargs: Mapping[str, Any]):
            postprocess_fn = postprocess_fn_gen(**dict(model_kwargs))
            return postprocess_fn(x)

        def _unconstrain_raw(theta: Mapping[str, Any], model_kwargs: Mapping[str, Any]):
            return numpyro.infer.util.unconstrain_fn(
                model, (), dict(model_kwargs), theta
            )

        def constrain(x: MutableMapping[str, Any], **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            x = dict(x)
            keys_to_remove = _fill_missing_params(x, default_unconstrained, parameter_names)
            params, extras = _split_params_and_extras(x, parameter_names)
            constrained_params = _constrain_raw(params, mk)
            constrained_params.update(extras)
            for k in keys_to_remove:
                constrained_params.pop(k, None)
            return constrained_params

        def unconstrain(theta: MutableMapping[str, Any], **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            theta = dict(theta)
            keys_to_remove = _fill_missing_params(theta, default_constrained, parameter_names)
            params, extras = _split_params_and_extras(theta, parameter_names)
            unconstrained_params = _unconstrain_raw(params, mk)
            unconstrained_params.update(extras)
            for k in keys_to_remove:
                unconstrained_params.pop(k, None)
            return unconstrained_params

        def log_probability_from_constrained_rvs(theta: PyTree, **model_kwargs: Any):
            x = unconstrain(theta, **model_kwargs)
            return log_probability_from_unconstrained_rvs(x, **model_kwargs)

        def tempered_log_probability_from_constrained_rvs(
            theta: PyTree, *, temperature: float = 1.0, **model_kwargs: Any
        ):
            x = unconstrain(theta, **model_kwargs)
            return tempered_log_probability_from_unconstrained_rvs(
                x, temperature=temperature, **model_kwargs
            )

        def simulate(rng_key: jax.Array, **model_kwargs: Any):
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            seeded = numpyro.handlers.seed(model, rng_key)
            return seeded(**mk)

        def log_probability_from_unconstrained_rvs_with_trace(
            x: PyTree, **model_kwargs: Any
        ):
            substituted_model = numpyro.handlers.substitute(
                model, substitute_fn=partial(numpyro.infer.util._unconstrain_reparam, x)
            )
            mk = _merge_model_kwargs(init_model_kwargs, model_kwargs)
            log_prob, trace = numpyro.infer.util.log_density(
                substituted_model, (), mk, {}
            )
            return log_prob, trace

        def tempered_log_probability_from_unconstrained_rvs_with_trace(
            x: PyTree, *, temperature: float = 1.0, **model_kwargs: Any
        ):
            model_kwargs = {**model_kwargs, tempering_param_kwarg: temperature}
            return log_probability_from_unconstrained_rvs_with_trace(x, **model_kwargs)

    if jit_compile:
        log_probability_from_unconstrained_rvs = jax.jit(log_probability_from_unconstrained_rvs)
        tempered_log_probability_from_unconstrained_rvs = jax.jit(
            tempered_log_probability_from_unconstrained_rvs
        )
        log_probability_from_constrained_rvs = jax.jit(log_probability_from_constrained_rvs)
        tempered_log_probability_from_constrained_rvs = jax.jit(
            tempered_log_probability_from_constrained_rvs
        )
        log_probability_from_unconstrained_rvs_with_trace = eqx.filter_jit(
            log_probability_from_unconstrained_rvs_with_trace
        )
        tempered_log_probability_from_unconstrained_rvs_with_trace = eqx.filter_jit(
            tempered_log_probability_from_unconstrained_rvs_with_trace
        )
        constrain = jax.jit(constrain)
        unconstrain = jax.jit(unconstrain)

    return PPLUtilityGroup(
        parameter_names=parameter_names,
        valid_unconstrained_inputs=default_unconstrained,
        valid_constrained_inputs=default_constrained,
        log_probability_from_unconstrained_rvs=log_probability_from_unconstrained_rvs,
        log_probability_from_constrained_rvs=log_probability_from_constrained_rvs,
        tempered_log_probability_from_unconstrained_rvs=tempered_log_probability_from_unconstrained_rvs,
        tempered_log_probability_from_constrained_rvs=tempered_log_probability_from_constrained_rvs,
        log_probability_from_unconstrained_rvs_with_trace=log_probability_from_unconstrained_rvs_with_trace,
        tempered_log_probability_from_unconstrained_rvs_with_trace=tempered_log_probability_from_unconstrained_rvs_with_trace,
        constrain=constrain,
        unconstrain=unconstrain,
        simulate=simulate,
    )


@dataclass(frozen=True)
class PPLUtilities:
    parameter_names: list[str]
    valid_unconstrained_inputs: Dict[str, Any]
    valid_constrained_inputs: Dict[str, Any]

    log_probability_from_unconstrained_rvs: Callable[[PyTree], Any]
    log_probability_from_constrained_rvs: Callable[[PyTree], Any]

    tempered_log_probability_from_unconstrained_rvs: Callable[[PyTree], Any]
    tempered_log_probability_from_constrained_rvs: Callable[[PyTree], Any]

    log_probability_from_unconstrained_rvs_with_trace: Callable[[PyTree], Any]
    tempered_log_probability_from_unconstrained_rvs_with_trace: Callable[[PyTree], Any]

    constrain: Callable[[PyTree], Any]
    unconstrain: Callable[[PyTree], Any]

    simulate: Callable[..., Any]

    prior: Optional[PPLUtilityGroup] = None
    likelihood: Optional[PPLUtilityGroup] = None


def build_ppl_utilities(
    model: Callable[..., Any] = None,
    prior: Callable[..., Any] = None,
    likelihood: Callable[..., Any] = None,
    init_model_kwargs: Mapping[str, Any] = None,
    init_rng_key: Optional[jax.Array] = None,
    tempering_param_kwarg: str = "tempering_param",
    jit_compile: bool = False,
) -> PPLUtilities:
    """Build log-prob, constrain/unconstrain, and simulation utilities from a NumPyro model.

    Valid call patterns:
    - Pass `model` only.
    - Pass both `prior` and `likelihood` (with `model=None`); the joint is composed
      internally by running prior then likelihood under a shared trace.

    Assumptions:
    - `model` is called as `model(**model_kwargs)`.
    - `prior` is called as `prior(**prior_kwargs)`.
    - `likelihood` is called as `likelihood(**likelihood_kwargs)`.
    - `model`/composed-joint accept a `tempering_param` kwarg (name configurable via
      `tempering_param_kwarg`) when tempered log-prob utilities are used.
    - `init_model_kwargs` must contain a valid set of kwargs for initializing the model,
      but per-call overrides are allowed by passing kwargs to the returned functions.

    Returns utilities that operate on a parameter dict-like pytree keyed by latent site names.
    When split callables are provided, nested `prior` and `likelihood` groups with the
    same utility families are added (accessible as `ppl_utils.prior` and
    `ppl_utils.likelihood`).

    For the likelihood group, all helpers accept an additional positional argument `z`
    (prior RV values). The space of `z` matches the primary input: unconstrained-space
    helpers (e.g. ``log_probability_from_unconstrained_rvs(x, z, ...)``) expect
    unconstrained `z`; constrained-space helpers (e.g. ``log_probability_from_constrained_rvs(theta, z, ...)``)
    expect constrained `z`. The prior NumPyro callable is ``prior(**kwargs)``; the
    likelihood is ``likelihood(z, **kwargs)``.
    """

    if init_rng_key is None:
        init_rng_key = jr.key(0)

    uses_model = model is not None
    uses_split = prior is not None or likelihood is not None

    if uses_model and uses_split:
        raise ValueError(
            "Specify either `model` OR (`prior` and `likelihood`), but not both."
        )
    if not uses_model and not uses_split:
        raise ValueError(
            "Specify either `model` OR both `prior` and `likelihood`."
        )
    if not uses_model and (prior is None or likelihood is None):
        raise ValueError(
            "When `model` is omitted, both `prior` and `likelihood` are required."
        )

    if uses_model:
        joint_model = model
    else:
        def _composed_joint_model(**model_kwargs: Any):
            prior_trace = numpyro.handlers.trace(prior).get_trace(**model_kwargs)
            z = {
                name: site["value"]
                for name, site in prior_trace.items()
                if site["type"] == "sample" and not site.get("is_observed", False)
            }
            likelihood(z, **model_kwargs)

        joint_model = _composed_joint_model

    joint = _build_utility_group(
        joint_model,
        init_model_kwargs=init_model_kwargs,
        init_rng_key=init_rng_key,
        tempering_param_kwarg=tempering_param_kwarg,
        jit_compile=jit_compile,
    )

    prior_group: Optional[PPLUtilityGroup] = None
    likelihood_group: Optional[PPLUtilityGroup] = None
    if not uses_model:
        prior_group = _build_utility_group(
            prior,
            init_model_kwargs=init_model_kwargs,
            init_rng_key=init_rng_key,
            tempering_param_kwarg=tempering_param_kwarg,
            jit_compile=jit_compile,
        )
        likelihood_group = _build_utility_group(
            likelihood,
            init_model_kwargs=init_model_kwargs,
            init_rng_key=init_rng_key,
            tempering_param_kwarg=tempering_param_kwarg,
            jit_compile=jit_compile,
            conditioned_on_z=True,
            z_init=prior_group.valid_constrained_inputs,
            prior_group=prior_group,
        )

    return PPLUtilities(
        parameter_names=joint.parameter_names,
        valid_unconstrained_inputs=joint.valid_unconstrained_inputs,
        valid_constrained_inputs=joint.valid_constrained_inputs,
        log_probability_from_unconstrained_rvs=joint.log_probability_from_unconstrained_rvs,
        log_probability_from_constrained_rvs=joint.log_probability_from_constrained_rvs,
        tempered_log_probability_from_unconstrained_rvs=joint.tempered_log_probability_from_unconstrained_rvs,
        tempered_log_probability_from_constrained_rvs=joint.tempered_log_probability_from_constrained_rvs,
        log_probability_from_unconstrained_rvs_with_trace=joint.log_probability_from_unconstrained_rvs_with_trace,
        tempered_log_probability_from_unconstrained_rvs_with_trace=joint.tempered_log_probability_from_unconstrained_rvs_with_trace,
        constrain=joint.constrain,
        unconstrain=joint.unconstrain,
        simulate=joint.simulate,
        prior=prior_group,
        likelihood=likelihood_group,
    )