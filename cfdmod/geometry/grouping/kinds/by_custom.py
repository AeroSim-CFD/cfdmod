"""User-defined grouping via a Python callback.

Escape hatch for grouping logic that none of the built-in kinds express
naturally (e.g. semantic labels from a CAD attribute, K-means on a
derived feature, geodesic distance from a seed face).

Two ways to specify the callback:

- An importable dotted path string (``"my_pkg.my_module.my_func"``),
  which round-trips cleanly through YAML/JSON serialisation.
- A direct ``Callable``, useful in notebooks; the model attempts to
  derive a dotted path on serialisation but raises on lambdas or local
  functions where no such path exists.

Standardised callback signature::

    def callback(
        mesh: LnasFormat,
        candidate_idxs: np.ndarray,   # int64, sorted parent indices
        params: dict[str, Any],       # spec.params, untouched
    ) -> dict[str, np.ndarray]:
        '''Return {group_name: parent_triangle_indices_int64}.

        Indices must lie within ``candidate_idxs`` (so ``restrict_to``
        is honored). The driver sorts and de-duplicates per-group
        indices before merging into the result.
        '''
"""

from __future__ import annotations

import importlib
from typing import Annotated, Any, Literal

import numpy as np
from lnas import LnasFormat
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer


class CustomGrouping(BaseModel):
    """Grouping defined by a user-supplied Python callback.

    Args:
        kind: Discriminator literal, always ``"by_custom"``.
        callback: Either an importable dotted path string
            (``"my_pkg.my_func"``) or a Python callable matching the
            signature documented in the module docstring.
        params: User-defined parameters passed verbatim to the
            callback. Keep JSON/YAML-safe if you intend to persist the
            chain via :func:`dump_groupings`.
        restrict_to: Optional list of earlier group names; when set,
            the callback receives only triangles in (the union of)
            those groups as ``candidate_idxs``.
    """

    kind: Literal["by_custom"] = "by_custom"
    callback: Annotated[
        Any,
        Field(description="Importable dotted path string or Python callable."),
    ]
    params: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Untyped params passed to the callback."),
    ]
    restrict_to: Annotated[
        list[str] | None,
        Field(None, description="Optional list of earlier group names to restrict to."),
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("callback", mode="before")
    def _validate_callback(cls, v: Any) -> Any:
        if isinstance(v, str):
            if "." not in v:
                raise ValueError(
                    f"callback string must be a dotted import path "
                    f"('module.func'), got {v!r}"
                )
            return v
        if callable(v):
            return v
        raise ValueError(
            "callback must be a dotted import path string or a Python callable; "
            f"got {type(v).__name__}"
        )

    @model_serializer(mode="wrap")
    def _serialise(self, handler) -> dict[str, Any]:
        data = handler(self)
        cb = self.callback
        if isinstance(cb, str):
            return data
        mod = getattr(cb, "__module__", None)
        qn = getattr(cb, "__qualname__", None)
        if mod is None or qn is None or "<" in qn:
            raise ValueError(
                f"CustomGrouping callback {cb!r} is not importable by a stable "
                "dotted name (lambda or local function); cannot serialise. "
                "Pass a module-level function or a 'module.func' string."
            )
        data["callback"] = f"{mod}.{qn}"
        return data


def _resolve_callback(cb: Any):
    if callable(cb):
        return cb
    module_name, _, fn_name = cb.rpartition(".")
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"CustomGrouping callback path {cb!r} did not resolve to a callable"
        )
    return fn


def apply_by_custom(
    spec: CustomGrouping,
    mesh: LnasFormat,
    allowed: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Resolve the callback, run it, validate and normalise its output."""
    n_parent = int(mesh.geometry.triangles.shape[0])

    if allowed is not None:
        cand = np.asarray(allowed, dtype=np.int64)
    else:
        cand = np.arange(n_parent, dtype=np.int64)

    if cand.size == 0:
        return {}

    callback = _resolve_callback(spec.callback)
    raw = callback(mesh, cand, spec.params)

    if not isinstance(raw, dict):
        raise TypeError(
            f"CustomGrouping callback must return dict[str, np.ndarray]; "
            f"got {type(raw).__name__}"
        )

    out: dict[str, np.ndarray] = {}
    for name, idxs in raw.items():
        if not isinstance(name, str):
            raise TypeError(
                f"CustomGrouping group names must be str; got {type(name).__name__}"
            )
        arr = np.unique(np.asarray(idxs, dtype=np.int64).ravel())
        if arr.size > 0:
            if arr[0] < 0 or arr[-1] >= n_parent:
                raise ValueError(
                    f"CustomGrouping group {name!r}: triangle index out of range "
                    f"[0, {n_parent}); first/last={int(arr[0])}/{int(arr[-1])}"
                )
            in_cand = np.isin(arr, cand, assume_unique=True)
            if not in_cand.all():
                outside = arr[~in_cand]
                raise ValueError(
                    f"CustomGrouping group {name!r}: {outside.size} index/indices "
                    f"outside the candidate set (restrict_to violation); "
                    f"first offender = {int(outside[0])}"
                )
        out[name] = arr
    return out
