"""Regroup orchestration.

``run_regroup`` is the entry point used by the CLI and by callers that
already have a loaded mesh. It expands any
:class:`cfdmod.regroup.parameters.BySizeRoundedPerComponent` specs into
plain :class:`cfdmod.geometry.grouping.GroupingSpec` chains, runs the
grouping, builds the regrouped LnasFormat, and writes the new geometry
plus the rewritten timeseries.
"""

from __future__ import annotations

import pathlib

import numpy as np
from lnas import LnasFormat

from cfdmod.geometry.grouping import (
    ByDivisionsGrouping,
    ByZoningGrouping,
    GroupingSpec,
    apply_groupings,
)
from cfdmod.geometry.grouping.kinds.by_divisions import _intervals_from_count
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.io.mesh import load_mesh
from cfdmod.io.xdmf import (
    write_processing_metadata,
    write_temporal_xdmf,
)
from cfdmod.logger import logger
from cfdmod.regroup.functions import (
    apply_regroup_to_timeseries,
    build_regrouped_mesh,
    build_regroup_mapping,
    build_sliced_regrouped_mesh,
)
from cfdmod.regroup.parameters import (
    BySizeRoundedPerComponent,
    RegroupConfig,
    RegroupSpec,
)

__all__ = ["expand_regroup_chain", "run_regroup"]


def _restricted_centroids(
    mesh: LnasFormat,
    parent_idxs: np.ndarray,
) -> np.ndarray:
    triangles = mesh.geometry.triangle_vertices  # (n_tri, 3, 3)
    return np.mean(triangles[parent_idxs], axis=1)  # (k, 3)


def _rounded_division_count(extent: float, target: float, min_n: int) -> int:
    """``max(min_n, round(extent / target))``; handles the degenerate extent=0 case."""
    if target <= 0:
        raise ValueError(f"target size must be > 0, got {target}")
    if extent <= 0.0:
        return max(min_n, 1)
    # int(x + 0.5) avoids banker's rounding so 0.5 -> 1 (matches user expectation
    # that a "barely there" division still counts as one cell).
    n = int(extent / target + 0.5)
    return max(min_n, n)


_IntervalsTriple = tuple[list[float], list[float], list[float]]


def _expand_one_per_component(
    spec: BySizeRoundedPerComponent,
    mesh: LnasFormat,
    prior_chain: list[GroupingSpec],
    transformation: TransformationConfig | None,
) -> tuple[
    list[GroupingSpec],
    set[str],
    dict[str, _IntervalsTriple],
    dict[str, np.ndarray],
]:
    """Expand a fan-out spec into one ``ByDivisionsGrouping`` per parent group.

    Returns ``(new_specs, consumed_parent_names)``. The consumed names
    are the parent groups whose triangles have been subdivided; the
    caller drops them from the final grouping so the leaf cells (which
    overlap with the parents) are the only output groups.
    """
    if not prior_chain:
        raise ValueError(
            "BySizeRoundedPerComponent requires at least one preceding grouping spec "
            "to define the components to fan out over."
        )

    prior = build_regroup_mapping(mesh, prior_chain, transformation)

    if spec.restrict_to is not None:
        missing = [n for n in spec.restrict_to if n not in prior.groups]
        if missing:
            raise ValueError(
                f"BySizeRoundedPerComponent: restrict_to references unknown "
                f"groups {missing}. Available: {sorted(prior.groups)}"
            )
        parent_names = [n for n in spec.restrict_to]
    else:
        parent_names = list(prior.groups.keys())

    # Compute centroids in the same (transformed) frame the binning will see.
    if transformation is not None:
        mesh_for_binning = mesh.copy()
        mesh_for_binning.geometry.apply_transformation(
            transformation.get_geometry_transformation()
        )
    else:
        mesh_for_binning = mesh

    expanded: list[GroupingSpec] = []
    consumed: set[str] = set()
    parent_intervals: dict[str, _IntervalsTriple] = {}
    parent_triangles: dict[str, np.ndarray] = {}
    for parent_name in parent_names:
        parent_idxs = np.asarray(prior.groups[parent_name], dtype=np.int64)
        if parent_idxs.size == 0:
            continue
        consumed.add(parent_name)
        parent_triangles[parent_name] = parent_idxs
        cents = _restricted_centroids(mesh_for_binning, parent_idxs)
        lo = cents.min(axis=0)
        hi = cents.max(axis=0)

        if spec.target_size_x is not None:
            n_x: int | None = _rounded_division_count(
                float(hi[0] - lo[0]), spec.target_size_x, spec.min_n_div
            )
        else:
            n_x = None
        if spec.target_size_y is not None:
            n_y: int | None = _rounded_division_count(
                float(hi[1] - lo[1]), spec.target_size_y, spec.min_n_div
            )
        else:
            n_y = None
        if spec.target_size_z is not None:
            n_z: int | None = _rounded_division_count(
                float(hi[2] - lo[2]), spec.target_size_z, spec.min_n_div
            )
        else:
            n_z = None

        sub_template = spec.name_template.replace("{parent}", parent_name)
        expanded.append(
            ByDivisionsGrouping(
                n_div_x=n_x,
                n_div_y=n_y,
                n_div_z=n_z,
                name_template=sub_template,
                restrict_to=[parent_name],
            )
        )
        parent_intervals[parent_name] = (
            _intervals_from_count(float(lo[0]), float(hi[0]), n_x),
            _intervals_from_count(float(lo[1]), float(hi[1]), n_y),
            _intervals_from_count(float(lo[2]), float(hi[2]), n_z),
        )

        logger.info(
            f"  expand[{parent_name}]: extents=({hi[0] - lo[0]:.3f}, "
            f"{hi[1] - lo[1]:.3f}, {hi[2] - lo[2]:.3f}) -> "
            f"n_div=({n_x}, {n_y}, {n_z})"
        )

    return expanded, consumed, parent_intervals, parent_triangles


def expand_regroup_chain(
    chain: list[RegroupSpec],
    mesh: LnasFormat,
    transformation: TransformationConfig | None,
) -> tuple[
    list[GroupingSpec],
    set[str],
    dict[str, _IntervalsTriple],
    dict[str, np.ndarray],
]:
    """Resolve any regroup-local specs into plain ``GroupingSpec`` entries.

    Returns ``(expanded_specs, consumed_group_names, parent_intervals,
    parent_triangles)``. ``consumed`` names are intermediate parent
    groups that ``BySizeRoundedPerComponent`` has fanned out over;
    ``parent_intervals`` and ``parent_triangles`` carry the per-parent
    cut planes and triangle indices needed to drive the ``"sliced"``
    aggregation mode (empty dicts when no fan-out happened).
    """
    expanded: list[GroupingSpec] = []
    consumed: set[str] = set()
    parent_intervals: dict[str, _IntervalsTriple] = {}
    parent_triangles: dict[str, np.ndarray] = {}
    for spec in chain:
        if isinstance(spec, BySizeRoundedPerComponent):
            new_specs, new_consumed, new_intervals, new_triangles = (
                _expand_one_per_component(spec, mesh, expanded, transformation)
            )
            expanded.extend(new_specs)
            consumed |= new_consumed
            parent_intervals.update(new_intervals)
            parent_triangles.update(new_triangles)
        else:
            expanded.append(spec)
    return expanded, consumed, parent_intervals, parent_triangles


def _resolve_global_intervals(
    expanded: list[GroupingSpec],
    grouping,
    mesh: LnasFormat,
) -> tuple[_IntervalsTriple, np.ndarray] | None:
    """For a chain whose only zoning is a single trailing ByZoning/Divisions,
    return ``((x, y, z), all_grouped_triangle_idxs)``; else None.

    Used by sliced mode when the user wrote a one-shot zoning config (not a
    BySizeRoundedPerComponent fan-out). Picks the last zoning-like spec.
    """
    last_zoning: ByZoningGrouping | None = None
    for spec in expanded:
        if isinstance(spec, ByZoningGrouping):
            last_zoning = spec
        elif isinstance(spec, ByDivisionsGrouping):
            # ByDivisionsGrouping resolves to a ByZoningGrouping; bbox-derived.
            cents = mesh.geometry.triangle_vertices.mean(axis=1)
            allowed = np.concatenate(
                [grouping.groups[n] for n in (spec.restrict_to or [])]
            ) if spec.restrict_to else np.arange(cents.shape[0], dtype=np.int64)
            if allowed.size == 0:
                continue
            cand = cents[allowed]
            lo = cand.min(axis=0)
            hi = cand.max(axis=0)
            last_zoning = ByZoningGrouping(
                x_intervals=_intervals_from_count(float(lo[0]), float(hi[0]), spec.n_div_x),
                y_intervals=_intervals_from_count(float(lo[1]), float(hi[1]), spec.n_div_y),
                z_intervals=_intervals_from_count(float(lo[2]), float(hi[2]), spec.n_div_z),
            )
    if last_zoning is None:
        return None
    all_idxs = np.unique(
        np.concatenate([np.asarray(idxs, dtype=np.int64) for idxs in grouping.groups.values()])
    ) if grouping.groups else np.array([], dtype=np.int64)
    return (
        (
            list(last_zoning.x_intervals),
            list(last_zoning.y_intervals),
            list(last_zoning.z_intervals),
        ),
        all_idxs,
    )


def _record_processing_metadata(
    output_h5: pathlib.Path,
    group: str,
    cfg: RegroupConfig,
    expanded: list[GroupingSpec],
    *,
    input_geometry: pathlib.Path | None,
    input_timeseries: pathlib.Path,
) -> None:
    config_dict = {
        "regroup": {
            "groupings": [s.model_dump(mode="python") for s in cfg.groupings],
            "expanded_groupings": [s.model_dump(mode="python") for s in expanded],
            "aggregation": cfg.aggregation,
            "timeseries_group": cfg.timeseries_group,
            "transformation": (
                cfg.transformation.model_dump(mode="python")
                if cfg.transformation is not None
                else None
            ),
            "unassigned_policy": cfg.unassigned_policy,
        }
    }
    extra = {"input_timeseries": str(input_timeseries)}
    if input_geometry is not None:
        extra["input_geometry"] = str(input_geometry)
    write_processing_metadata(output_h5, group, config=config_dict, extra=extra)


def run_regroup(
    cfg: RegroupConfig,
    geometry: pathlib.Path | LnasFormat,
    timeseries: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Run the full regroup pipeline and write outputs to ``output_dir``.

    Outputs:
        - ``geometry.lnas`` (always); ``geometry.stl`` if ``cfg.output_geometry_format
          == "lnas_and_stl"``.
        - ``{cfg.timeseries_group}.regrouped.h5`` and a sibling ``.xdmf``.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries = pathlib.Path(timeseries)

    input_geometry_path = (
        pathlib.Path(geometry) if not isinstance(geometry, LnasFormat) else None
    )
    mesh = load_mesh(geometry)

    expanded, consumed, parent_intervals, parent_triangles = expand_regroup_chain(
        cfg.groupings, mesh, cfg.transformation
    )
    if not expanded:
        raise ValueError(
            "regroup: chain expanded to zero specs (no work to do); check the config."
        )

    grouping = build_regroup_mapping(mesh, expanded, cfg.transformation)
    if consumed:
        kept = {n: idxs for n, idxs in grouping.groups.items() if n not in consumed}
        from cfdmod.geometry.grouping import GroupingResult

        grouping = GroupingResult(
            parent_n_triangles=grouping.parent_n_triangles, groups=kept
        )
        logger.info(
            f"regroup: dropped {len(consumed)} consumed parent group(s): "
            f"{sorted(consumed)}"
        )
    if not grouping.groups and cfg.unassigned_policy == "drop":
        raise ValueError(
            "regroup: chain produced zero non-empty groups and unassigned_policy='drop'."
        )

    if cfg.aggregation == "sliced":
        if not parent_intervals:
            # No fan-out happened. Try to extract a single trailing zoning to
            # use as the global cut. Fall back to error otherwise.
            global_resolved = _resolve_global_intervals(expanded, grouping, mesh)
            if global_resolved is None:
                raise ValueError(
                    "regroup (sliced): could not resolve cut intervals from the chain. "
                    "Use a chain with BySizeRoundedPerComponent or a trailing "
                    "ByZoningGrouping / ByDivisionsGrouping."
                )
            global_intervals, global_idxs = global_resolved
            parent_intervals = {"*": global_intervals}
            parent_triangles = {"*": global_idxs}

        logger.info(
            f"regroup (sliced): slicing {len(parent_intervals)} parent "
            f"component(s) along axis-aligned cut planes..."
        )
        new_lnas, regroup_index = build_sliced_regrouped_mesh(
            mesh,
            grouping,
            parent_intervals=parent_intervals,
            parent_triangles=parent_triangles,
            unassigned_policy=cfg.unassigned_policy,
        )
    else:
        new_lnas, regroup_index = build_regrouped_mesh(
            mesh,
            grouping,
            aggregation=cfg.aggregation,
            unassigned_policy=cfg.unassigned_policy,
        )

    out_lnas = output_dir / "geometry.lnas"
    new_lnas.to_file(out_lnas)
    if cfg.output_geometry_format == "lnas_and_stl":
        new_lnas.geometry.export_stl(output_dir / "geometry.stl")
    logger.info(
        f"regroup: wrote geometry to {out_lnas} ({new_lnas.geometry.triangles.shape[0]} "
        f"triangle(s), {len(new_lnas.surfaces)} surface(s))"
    )

    out_h5 = output_dir / f"{cfg.timeseries_group}.regrouped.h5"
    apply_regroup_to_timeseries(
        timeseries,
        out_h5,
        group=cfg.timeseries_group,
        regroup_index=regroup_index,
        new_triangles=new_lnas.geometry.triangles,
        new_vertices=new_lnas.geometry.vertices,
    )

    out_xdmf = out_h5.with_suffix(".xdmf")
    write_temporal_xdmf(out_h5, out_xdmf, cfg.timeseries_group)

    _record_processing_metadata(
        out_h5,
        cfg.timeseries_group,
        cfg,
        expanded,
        input_geometry=input_geometry_path,
        input_timeseries=timeseries,
    )

    logger.info(f"regroup: wrote timeseries to {out_h5} (+ sibling XDMF)")
