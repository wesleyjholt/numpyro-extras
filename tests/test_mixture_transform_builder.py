"""Step 05 manifest for Task 10 integration + factory validation.

Required module:
- `numpyro_extras.mixture_transform_builder`
  (expected file: `src/numpyro_extras/mixture_transform_builder.py`)

Required public API:
- `build_mixture_transform(...)`
- optional: convenience builders for uniform/normal base transforms

Behavioral pass gates:
- factory accepts mixture object and explicit parameter modes
- builder returns transform type matching requested base
- default end-to-end roundtrip stays finite and within tolerance
- distributional sanity vs direct mixture sampling
- config propagation and diagnostics contracts
- invalid inputs fail fast with stage-specific context
- JAX `jit` + `vmap` runtime compatibility without host fallback
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import jax
import numpyro.distributions as dist
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


MODULE_NAME = "numpyro_extras.mixture_transform_builder"
TRANSFORMS_MODULE_NAME = "numpyro_extras.transforms"
BUILD_FN_NAME = "build_mixture_transform"
UNIFORM_TRANSFORM_NAME = "UniformToMixtureTransform"
NORMAL_TRANSFORM_NAME = "NormalToMixtureTransform"

MIXTURE_SHAPE_CASES = [
    pytest.param(
        dict(weights=[0.50, 0.50], loc=[-2.2, 1.8], scale=[0.8, 1.1]),
        id="balanced_bimodal",
    ),
    pytest.param(
        dict(weights=[0.99, 0.01], loc=[-1.5, 5.8], scale=[1.0, 0.65]),
        id="dominant_component",
    ),
    pytest.param(
        dict(weights=[0.45, 0.55], loc=[0.0, 0.12], scale=[0.95, 1.05]),
        id="near_overlapping",
    ),
]


@dataclasses.dataclass(frozen=True)
class _BuilderPayload:
    transform: object
    diagnostics: object | None
    metadata: object | None
    raw_result: object


def _import_builder_module() -> ModuleType:
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.mixture_transform_builder` "
                "(expected at `src/numpyro_extras/mixture_transform_builder.py`)."
            )
        raise


def _import_transforms_module() -> ModuleType:
    try:
        return importlib.import_module(TRANSFORMS_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name in {TRANSFORMS_MODULE_NAME, "numpyro_extras"}:
            pytest.fail(
                "Missing module `numpyro_extras.transforms` "
                "(expected at `src/numpyro_extras/transforms.py`)."
            )
        raise


def _get_builder():
    module = _import_builder_module()
    builder = getattr(module, BUILD_FN_NAME, None)
    if builder is None:
        pytest.fail(
            "Missing `build_mixture_transform` in "
            "`numpyro_extras.mixture_transform_builder`."
        )
    if not callable(builder):
        pytest.fail("`build_mixture_transform` must be callable.")
    return builder


def _expected_transform_type(base: str):
    module = _import_transforms_module()
    if base == "uniform":
        expected = getattr(module, UNIFORM_TRANSFORM_NAME, None)
        if expected is None:
            pytest.fail(
                "Missing `UniformToMixtureTransform` in `numpyro_extras.transforms`."
            )
        return expected

    if base == "normal":
        expected = getattr(module, NORMAL_TRANSFORM_NAME, None)
        if expected is None:
            pytest.fail(
                "Missing `NormalToMixtureTransform` in `numpyro_extras.transforms`."
            )
        return expected

    raise AssertionError(f"Unexpected base: {base}")


def _make_build_cfg(**overrides):
    cfg = dict(
        solver_cfg=SimpleNamespace(
            maxiter=256,
            rtol=1e-8,
            atol=1e-8,
            bracket_cfg=SimpleNamespace(
                x_init_low=-8.0,
                x_init_high=8.0,
                expansion_factor=2.0,
                max_expansions=32,
            ),
        ),
        knot_cfg=SimpleNamespace(num_knots=257, u_eps=1e-6),
        interp_cfg=SimpleNamespace(
            interior_method="akima",
            clip_u_eps=1e-10,
            safe_arctanh_eps=1e-7,
        ),
        tail_cfg=SimpleNamespace(
            enforce_c1_stitch=True,
            min_tail_scale=1e-8,
        ),
        transform_cfg=SimpleNamespace(
            clip_u_eps=1e-10,
            validate_args=False,
        ),
    )
    for key, value in overrides.items():
        if (
            key in cfg
            and isinstance(cfg[key], SimpleNamespace)
            and isinstance(value, Mapping)
        ):
            merged = vars(cfg[key]).copy()
            merged.update(value)
            cfg[key] = SimpleNamespace(**merged)
            continue
        cfg[key] = value
    return SimpleNamespace(**cfg)


def _cfg_to_kwargs(build_cfg) -> dict[str, object]:
    if build_cfg is None:
        return {}
    if isinstance(build_cfg, Mapping):
        return dict(build_cfg)
    if isinstance(build_cfg, SimpleNamespace):
        return vars(build_cfg).copy()
    return {
        name: getattr(build_cfg, name)
        for name in dir(build_cfg)
        if not name.startswith("_")
    }


def _make_mixture_distribution(weights, loc, scale):
    weights = jnp.asarray(weights, dtype=jnp.float64)
    loc = jnp.asarray(loc, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    return dist.MixtureSameFamily(
        dist.Categorical(probs=weights),
        dist.Normal(loc=loc, scale=scale),
    )


def _is_signature_type_error(exc: TypeError) -> bool:
    message = str(exc).lower()
    signature_tokens = (
        "unexpected keyword",
        "positional argument",
        "required positional",
        "got multiple values",
        "takes ",
        "missing ",
    )
    return any(token in message for token in signature_tokens)


def _call_builder(
    *,
    base: str,
    base_kwargs_variants: list[dict[str, object]],
    build_cfg=None,
    extra_kwargs: Mapping[str, object] | None = None,
):
    builder = _get_builder()
    extra_kwargs = dict(extra_kwargs or {})
    cfg_kwargs = _cfg_to_kwargs(build_cfg)
    attempts = []
    errors: list[str] = []

    for base_kwargs in base_kwargs_variants:
        root_kwargs = {"base": base, **base_kwargs, **extra_kwargs}
        attempts.append(root_kwargs)
        if build_cfg is not None:
            attempts.append({**root_kwargs, **cfg_kwargs})
            for cfg_name in ("build_cfg", "config", "cfg", "mixture_transform_cfg"):
                attempts.append({**root_kwargs, cfg_name: build_cfg})
                attempts.append({**root_kwargs, cfg_name: cfg_kwargs})

    for kwargs in attempts:
        try:
            return builder(**kwargs)
        except TypeError as exc:
            if not _is_signature_type_error(exc):
                raise
            errors.append(str(exc))
            continue

    signature = inspect.signature(builder)
    attempts_preview = [sorted(kwargs.keys()) for kwargs in attempts[:4]]
    pytest.fail(
        "Unable to call `build_mixture_transform` with expected contract variants. "
        "Expected support for mixture-object mode and explicit-params mode with "
        f"`base='uniform'|'normal'`. Signature={signature}. "
        f"Attempted-key-sets={attempts_preview}. "
        f"Recent TypeErrors={errors[-3:]}."
    )


def _build_from_mixture(
    *,
    base: str,
    mixture_distribution,
    build_cfg=None,
    extra_kwargs: Mapping[str, object] | None = None,
):
    return _call_builder(
        base=base,
        base_kwargs_variants=[
            {"mixture_distribution": mixture_distribution},
            {"mixture": mixture_distribution},
        ],
        build_cfg=build_cfg,
        extra_kwargs=extra_kwargs,
    )


def _build_from_explicit(
    *,
    base: str,
    weights,
    loc,
    scale,
    build_cfg=None,
    extra_kwargs: Mapping[str, object] | None = None,
):
    weights = jnp.asarray(weights, dtype=jnp.float64)
    loc = jnp.asarray(loc, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    component_distribution = dist.Normal(loc=loc, scale=scale)
    component_params = {"loc": loc, "scale": scale}
    return _call_builder(
        base=base,
        base_kwargs_variants=[
            {"weights": weights, "component_distribution": component_distribution},
            {
                "weights": weights,
                "component_family": dist.Normal,
                "component_params": component_params,
            },
            {
                "weights": weights,
                "component_cls": dist.Normal,
                "component_params": component_params,
            },
            {
                "weights": weights,
                "component_distribution_cls": dist.Normal,
                "component_params": component_params,
            },
            {"weights": weights, "loc": loc, "scale": scale, "component_family": dist.Normal},
            {"weights": weights, "loc": loc, "scale": scale, "component_cls": dist.Normal},
        ],
        build_cfg=build_cfg,
        extra_kwargs=extra_kwargs,
    )


def _extract_payload(result) -> _BuilderPayload:
    if isinstance(result, Mapping):
        transform = result.get("transform")
        diagnostics = result.get("diagnostics")
        metadata = result.get("metadata", result.get("meta", result.get("build_info")))
        if transform is None:
            pytest.fail(
                "Builder must return a mapping containing `transform`, or return a "
                "transform object directly."
            )
        return _BuilderPayload(
            transform=transform,
            diagnostics=diagnostics,
            metadata=metadata,
            raw_result=result,
        )

    transform = getattr(result, "transform", None)
    diagnostics = getattr(result, "diagnostics", None)
    metadata = getattr(result, "metadata", getattr(result, "meta", None))
    if transform is None and callable(result):
        transform = result
    if transform is None:
        pytest.fail(
            "Builder return value must expose a transform via `.transform` or be "
            "the transform object itself."
        )
    return _BuilderPayload(
        transform=transform,
        diagnostics=diagnostics,
        metadata=metadata,
        raw_result=result,
    )


def _assert_transform_contract(transform, *, base: str):
    for method_name in ("__call__", "_inverse", "log_abs_det_jacobian"):
        method = getattr(transform, method_name, None)
        assert callable(method), f"Transform missing callable `{method_name}`."

    if base == "uniform":
        x_scalar = jnp.asarray(0.73, dtype=jnp.float64)
    else:
        x_scalar = jnp.asarray(-1.25, dtype=jnp.float64)
    y_scalar = transform(x_scalar)
    x_roundtrip = transform._inverse(y_scalar)
    ladj = transform.log_abs_det_jacobian(x_scalar, y_scalar)
    assert jnp.shape(y_scalar) == (), "Transform forward must accept scalar input."
    assert jnp.shape(x_roundtrip) == (), "Transform inverse must accept scalar input."
    assert jnp.shape(ladj) == (), "Transform Jacobian must accept scalar input."


def _diagnostic_handles(payload: _BuilderPayload) -> list[object]:
    handles = []
    for candidate in (
        payload.diagnostics,
        payload.metadata,
        getattr(payload.transform, "diagnostics", None),
        getattr(payload.transform, "metadata", None),
        getattr(payload.raw_result, "diagnostics", None),
        getattr(payload.raw_result, "metadata", None),
    ):
        if candidate is not None:
            handles.append(candidate)

    if isinstance(payload.raw_result, Mapping):
        for key in ("diagnostics", "metadata", "meta", "build_info", "config"):
            value = payload.raw_result.get(key)
            if value is not None:
                handles.append(value)
    return handles


def _to_plain_tree(value, *, _depth: int = 0):
    if _depth > 8:
        return repr(value)
    if dataclasses.is_dataclass(value):
        return _to_plain_tree(dataclasses.asdict(value), _depth=_depth + 1)
    if isinstance(value, SimpleNamespace):
        return _to_plain_tree(vars(value), _depth=_depth + 1)
    if isinstance(value, Mapping):
        return {
            str(k): _to_plain_tree(v, _depth=_depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_plain_tree(v, _depth=_depth + 1) for v in value]
    return value


def _tree_contains_key_fragment(tree, fragment: str) -> bool:
    fragment = fragment.lower()
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            if fragment in str(key).lower():
                return True
            if _tree_contains_key_fragment(value, fragment):
                return True
        return False
    if isinstance(tree, list):
        return any(_tree_contains_key_fragment(v, fragment) for v in tree)
    return False


def _tree_contains_bool(tree, target: bool) -> bool:
    if isinstance(tree, Mapping):
        return any(_tree_contains_bool(v, target) for v in tree.values())
    if isinstance(tree, list):
        return any(_tree_contains_bool(v, target) for v in tree)
    return isinstance(tree, bool) and tree is target


def _tree_contains_number(tree, target: float, *, atol: float, rtol: float) -> bool:
    if isinstance(tree, Mapping):
        return any(
            _tree_contains_number(v, target, atol=atol, rtol=rtol)
            for v in tree.values()
        )
    if isinstance(tree, list):
        return any(
            _tree_contains_number(v, target, atol=atol, rtol=rtol) for v in tree
        )
    if isinstance(tree, (int, float)):
        return abs(float(tree) - target) <= (atol + rtol * abs(target))
    return False


def _assert_roundtrip_and_finiteness(transform, *, base: str):
    if base == "uniform":
        x = jnp.linspace(1e-10, 1.0 - 1e-10, 2049, dtype=jnp.float64)
        interior = (x > 1e-4) & (x < 1.0 - 1e-4)
    else:
        x = jnp.linspace(-6.5, 6.5, 2049, dtype=jnp.float64)
        interior = jnp.abs(x) <= 4.0

    y = transform(x)
    x_roundtrip = transform._inverse(y)
    ladj = transform.log_abs_det_jacobian(x, y)
    err = jnp.abs(x_roundtrip - x)

    assert jnp.all(jnp.isfinite(y)), "Forward path produced NaN/Inf values."
    assert jnp.all(jnp.isfinite(x_roundtrip)), "Inverse path produced NaN/Inf values."
    assert jnp.all(jnp.isfinite(ladj)), "Jacobian path produced NaN/Inf values."
    interior_max_err = float(jnp.max(err[interior]))
    tail_max_err = float(jnp.max(err[~interior]))
    assert interior_max_err < 2e-4, (
        "Default end-to-end roundtrip interior tolerance violated: "
        f"max_err={interior_max_err:.3e}"
    )
    assert tail_max_err < 1e-3, (
        f"Default end-to-end roundtrip tail tolerance violated: max_err={tail_max_err:.3e}"
    )


@pytest.mark.parametrize("base", ["uniform", "normal"])
def test_builder_returns_expected_transform_type(base):
    builder = _get_builder()
    signature = inspect.signature(builder)
    assert "base" in signature.parameters, (
        "`build_mixture_transform` must accept `base`."
    )

    weights = [0.55, 0.45]
    loc = [-1.5, 2.0]
    scale = [0.9, 1.2]
    mixture = _make_mixture_distribution(weights=weights, loc=loc, scale=scale)

    object_payload = _extract_payload(
        _build_from_mixture(base=base, mixture_distribution=mixture)
    )
    explicit_payload = _extract_payload(
        _build_from_explicit(base=base, weights=weights, loc=loc, scale=scale)
    )
    expected_type = _expected_transform_type(base)

    assert isinstance(object_payload.transform, expected_type), (
        f"`build_mixture_transform(base='{base}')` returned unexpected transform type."
    )
    assert isinstance(explicit_payload.transform, expected_type), (
        "Explicit-params input mode must return the same transform class as "
        "mixture-object input mode."
    )
    _assert_transform_contract(object_payload.transform, base=base)
    _assert_transform_contract(explicit_payload.transform, base=base)

    probe = (
        jnp.linspace(1e-6, 1.0 - 1e-6, 257, dtype=jnp.float64)
        if base == "uniform"
        else jnp.linspace(-4.5, 4.5, 257, dtype=jnp.float64)
    )
    y_object = object_payload.transform(probe)
    y_explicit = explicit_payload.transform(probe)
    assert jnp.allclose(y_object, y_explicit, atol=5e-5, rtol=5e-5), (
        "Mixture-object and explicit-params builder modes should agree numerically "
        "for the same mixture configuration."
    )

    handles = _diagnostic_handles(object_payload)
    assert handles, (
        "Builder result must expose diagnostics/metadata handles "
        "(for configs, knots, and approximation summaries)."
    )


def test_builder_default_roundtrip_uniform_base():
    payload = _extract_payload(
        _build_from_mixture(
            base="uniform",
            mixture_distribution=_make_mixture_distribution(
                weights=[0.55, 0.45],
                loc=[-1.5, 2.0],
                scale=[0.9, 1.2],
            ),
        )
    )
    _assert_roundtrip_and_finiteness(payload.transform, base="uniform")


def test_builder_default_roundtrip_normal_base():
    payload = _extract_payload(
        _build_from_mixture(
            base="normal",
            mixture_distribution=_make_mixture_distribution(
                weights=[0.55, 0.45],
                loc=[-1.5, 2.0],
                scale=[0.9, 1.2],
            ),
        )
    )
    _assert_roundtrip_and_finiteness(payload.transform, base="normal")


@pytest.mark.parametrize("base", ["uniform", "normal"])
@pytest.mark.parametrize("case", MIXTURE_SHAPE_CASES)
def test_end_to_end_finite_for_multiple_mixture_shapes(base, case):
    payload = _extract_payload(
        _build_from_mixture(
            base=base,
            mixture_distribution=_make_mixture_distribution(
                weights=case["weights"],
                loc=case["loc"],
                scale=case["scale"],
            ),
        )
    )
    if base == "uniform":
        x = jnp.linspace(1e-8, 1.0 - 1e-8, 513, dtype=jnp.float64)
    else:
        x = jnp.linspace(-5.0, 5.0, 513, dtype=jnp.float64)

    y = payload.transform(x)
    x_back = payload.transform._inverse(y)
    ladj = payload.transform.log_abs_det_jacobian(x, y)
    assert y.shape == x.shape, "Forward map must preserve batch shape."
    assert x_back.shape == x.shape, "Inverse map must preserve batch shape."
    assert ladj.shape == x.shape, "Jacobian map must preserve batch shape."
    assert jnp.all(jnp.isfinite(y)), "Forward map must be finite for all shape cases."
    assert jnp.all(jnp.isfinite(x_back)), "Inverse map must be finite for all shape cases."
    assert jnp.all(jnp.isfinite(ladj)), "Jacobian map must be finite for all shape cases."


def test_distributional_sanity_against_direct_mixture_sampling():
    weights = jnp.array([0.8, 0.2], dtype=jnp.float64)
    loc = jnp.array([-1.25, 3.25], dtype=jnp.float64)
    scale = jnp.array([0.85, 1.55], dtype=jnp.float64)
    mixture = _make_mixture_distribution(weights=weights, loc=loc, scale=scale)

    payload = _extract_payload(
        _build_from_mixture(base="normal", mixture_distribution=mixture)
    )
    transform = payload.transform

    n = 120_000
    key_base, key_direct = jax.random.split(jax.random.PRNGKey(314159), 2)
    z = jax.random.normal(key_base, shape=(n,), dtype=jnp.float64)
    direct = mixture.sample(key_direct, sample_shape=(n,))
    from_transform = transform(z)

    q = jnp.asarray([0.1, 0.25, 0.5, 0.75, 0.9], dtype=jnp.float64)
    q_direct = jnp.quantile(direct, q)
    q_transform = jnp.quantile(from_transform, q)
    q_abs_err = jnp.abs(q_transform - q_direct)
    max_q_abs_err = float(jnp.max(q_abs_err))

    assert jnp.all(jnp.isfinite(from_transform)), (
        "Transform-generated samples must be finite in distributional sanity checks."
    )
    assert max_q_abs_err < 6e-2, (
        "Central quantile agreement vs direct mixture sampling violated: "
        f"max_abs_err={max_q_abs_err:.3e}"
    )


def test_custom_config_propagates_through_pipeline():
    mixture = _make_mixture_distribution(
        weights=[0.55, 0.45],
        loc=[-1.5, 2.0],
        scale=[0.9, 1.2],
    )
    custom_cfg = _make_build_cfg(
        knot_cfg={"num_knots": 129},
        interp_cfg={"clip_u_eps": 3e-9, "safe_arctanh_eps": 2e-7},
        tail_cfg={"enforce_c1_stitch": True, "min_tail_scale": 7e-8},
        transform_cfg={"clip_u_eps": 9e-11},
    )

    payload = _extract_payload(
        _build_from_mixture(
            base="normal",
            mixture_distribution=mixture,
            build_cfg=custom_cfg,
        )
    )
    handles = _diagnostic_handles(payload)
    assert handles, (
        "Builder must expose diagnostics/metadata handles so downstream users can "
        "inspect effective config and approximation diagnostics."
    )

    tree = _to_plain_tree(handles)
    assert _tree_contains_number(tree, 129, atol=0.0, rtol=0.0), (
        "Expected `knot_cfg.num_knots` value not found in diagnostics/config handles."
    )
    assert _tree_contains_number(tree, 9e-11, atol=0.0, rtol=1e-5), (
        "Expected custom `transform_cfg.clip_u_eps` value not found in diagnostics."
    )
    assert _tree_contains_bool(tree, True), (
        "Expected `enforce_c1_stitch=True` marker in diagnostics/config handles."
    )
    assert _tree_contains_key_fragment(tree, "stitch"), (
        "Tail/interpolation diagnostics must include stitch provenance."
    )
    assert _tree_contains_key_fragment(tree, "slope") or _tree_contains_key_fragment(
        tree, "boundary"
    ), (
        "Stitch provenance diagnostics must include boundary slope-source context."
    )

    invalid_cfg = _make_build_cfg(interp_cfg={"clip_u_eps": -1e-3})
    with pytest.raises(
        (ValueError, TypeError),
        match=r"(?i)clip|eps|interp|positive|invalid",
    ):
        _build_from_mixture(
            base="uniform",
            mixture_distribution=mixture,
            build_cfg=invalid_cfg,
        )


def test_invalid_inputs_fail_fast_with_context():
    with pytest.raises(
        (ValueError, TypeError),
        match=r"(?i)weight|prob|sum|simplex|non[- ]?negative",
    ):
        _build_from_explicit(
            base="uniform",
            weights=[-0.15, 1.15],
            loc=[-1.0, 2.0],
            scale=[0.9, 1.1],
        )

    with pytest.raises(
        (ValueError, TypeError),
        match=r"(?i)shape|size|broadcast|component|parameter",
    ):
        _build_from_explicit(
            base="normal",
            weights=[0.5, 0.5],
            loc=[-1.0, 0.0, 2.0],
            scale=[0.9, 1.1],
        )

    strict_cfg = _make_build_cfg(
        solver_cfg={
            "bracket_cfg": SimpleNamespace(
                x_init_low=0.0,
                x_init_high=0.0,
                expansion_factor=2.0,
                max_expansions=0,
            )
        }
    )
    with pytest.raises(
        (ValueError, RuntimeError),
        match=r"(?i)bracket|root|converg|monoton|knot",
    ) as exc_info:
        payload = _extract_payload(
            _build_from_mixture(
                base="uniform",
                mixture_distribution=_make_mixture_distribution(
                    weights=[0.5, 0.5],
                    loc=[-9.5, 10.5],
                    scale=[0.7, 0.9],
                ),
                build_cfg=strict_cfg,
            )
        )
        _ = payload.transform(jnp.asarray(0.999, dtype=jnp.float64))

    exc = exc_info.value
    for attr_name in ("diagnostics", "context", "failure_reason"):
        value = getattr(exc, attr_name, None)
        if value is None:
            continue
        failure_tree = _to_plain_tree(value)
        assert _tree_contains_key_fragment(failure_tree, "reason") or (
            _tree_contains_key_fragment(failure_tree, "stage")
        ), (
            "Failure diagnostics should include reason or stage context when available."
        )
        break


@pytest.mark.parametrize("base", ["uniform", "normal"])
def test_builder_jit_vmap_and_batch_shape_contract(base):
    payload = _extract_payload(
        _build_from_mixture(
            base=base,
            mixture_distribution=_make_mixture_distribution(
                weights=[0.55, 0.45],
                loc=[-1.5, 2.0],
                scale=[0.9, 1.2],
            ),
        )
    )
    transform = payload.transform
    if base == "uniform":
        x = jnp.linspace(1e-8, 1.0 - 1e-8, 1024, dtype=jnp.float64)
    else:
        x = jnp.linspace(-5.0, 5.0, 1024, dtype=jnp.float64)

    forward_jit = jax.jit(transform.__call__)
    inverse_jit = jax.jit(transform._inverse)
    ladj_jit = jax.jit(
        lambda t: transform.log_abs_det_jacobian(t, transform(t))
    )

    y = forward_jit(x)
    x_back = inverse_jit(y)
    ladj = ladj_jit(x)
    assert y.shape == x.shape
    assert x_back.shape == x.shape
    assert ladj.shape == x.shape
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(x_back))
    assert jnp.all(jnp.isfinite(ladj))

    y_vmap = jax.vmap(lambda xi: transform(xi))(x)
    x_back_vmap = jax.vmap(lambda yi: transform._inverse(yi))(y)
    assert jnp.allclose(y_vmap, y, atol=1e-9, rtol=1e-9), (
        "`vmap` forward path should match batched forward path."
    )
    assert jnp.allclose(x_back_vmap, x_back, atol=1e-9, rtol=1e-9), (
        "`vmap` inverse path should match batched inverse path."
    )

    x_batch = jnp.reshape(x[:512], (16, 32))
    y_batch = transform(x_batch)
    x_back_batch = transform._inverse(y_batch)
    ladj_batch = transform.log_abs_det_jacobian(x_batch, y_batch)
    assert y_batch.shape == x_batch.shape, "Forward path must preserve 2D batch shapes."
    assert x_back_batch.shape == x_batch.shape, (
        "Inverse path must preserve 2D batch shapes."
    )
    assert ladj_batch.shape == x_batch.shape, "Jacobian path must preserve 2D shapes."

    # If runtime path falls back to NumPy host ops, tracer conversion will fail.
    forward_jaxpr = jax.make_jaxpr(transform.__call__)(x)
    inverse_jaxpr = jax.make_jaxpr(transform._inverse)(y)
    ladj_jaxpr = jax.make_jaxpr(
        lambda t: transform.log_abs_det_jacobian(t, transform(t))
    )(x)
    assert len(forward_jaxpr.jaxpr.eqns) > 0
    assert len(inverse_jaxpr.jaxpr.eqns) > 0
    assert len(ladj_jaxpr.jaxpr.eqns) > 0
