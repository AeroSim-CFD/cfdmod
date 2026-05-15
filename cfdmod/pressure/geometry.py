"""Geometry utilities for the pressure module.

Builds :class:`GeometryData` and :class:`ProcessedEntity` over the
triangle-grouping pipeline (:mod:`cfdmod.geometry.grouping`). Each
geometry slice carries:

- ``mesh``: the body's or surface's :class:`LnasGeometry` sub-mesh.
- ``triangles_idxs``: the parent-mesh triangle indices that compose
  ``mesh`` (in ascending parent-index order).
- ``grouping``: the :class:`GroupingResult` produced by applying
  ``spec_chain`` to a transformed copy of the parent mesh. The chain
  always begins with a :class:`BySurfaceGrouping` whose only set is
  ``body_label`` and is followed by a single :class:`ByZoningGrouping`
  with ``restrict_to=[body_label]`` and
  ``name_template="{idx}-<body_label>"``. This composition reproduces
  the legacy ``f"{region_int}-{body_id}"`` region-label scheme with
  byte-exact output: cells are named identically and
  :func:`tabulate_geometry_data` emits the same DataFrame shape.
- ``body_label``: the surface-set name (Cf/Cm bodies use the body name;
  Ce uses the surface label).

The lone exception to "every triangle gets a region cell" is the
"unbinned" case: triangles in the body mesh whose centroid does not
fall in any zoning cell are emitted with the legacy sentinel label
``f"-1-{body_label}"`` so downstream consumers (statistics, lever-arm
resolution) keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lnas import LnasFormat, LnasGeometry

from cfdmod.geometry.grouping import (
    BySurfaceGrouping,
    ByZoningGrouping,
    GroupingResult,
    GroupingSpec,
    apply_groupings,
)
from cfdmod.io.geometry.region_meshing import create_regions_mesh
from cfdmod.io.geometry.transformation_config import TransformationConfig
from cfdmod.logger import logger
from cfdmod.pressure.parameters import BodyConfig, CeConfig, MomentBodyConfig, ZoningModel

# ---------------------------------------------------------------------------
# Shared geometry data containers
# ---------------------------------------------------------------------------


@dataclass
class GeometryData:
    """Per-body / per-surface geometry slice driven by a grouping chain.

    ``grouping`` is computed against a *transformed* copy of the parent
    mesh (the same transformation later applied to the working geometry
    in :func:`tabulate_geometry_data`), so spatial cells are evaluated
    in the same frame the legacy code did.
    """

    mesh: LnasGeometry
    triangles_idxs: np.ndarray
    grouping: GroupingResult
    spec_chain: list[GroupingSpec]
    body_label: str


@dataclass
class ProcessedEntity:
    mesh: LnasGeometry


# ---------------------------------------------------------------------------
# Internal helpers (legacy behaviour preserved exactly)
# ---------------------------------------------------------------------------


def _build_chain(
    body_label: str,
    sfc_list: list[str],
    zoning: ZoningModel,
) -> list[GroupingSpec]:
    """Assemble the canonical ``BySurface -> ByZoning`` chain for one body.

    The zoning ``name_template`` uses ``"{idx}-<body_label>"`` so the
    resulting region labels match the legacy format byte-for-byte.
    """
    return [
        BySurfaceGrouping(sets={body_label: list(sfc_list)}),
        ByZoningGrouping(
            x_intervals=list(zoning.x_intervals),
            y_intervals=list(zoning.y_intervals),
            z_intervals=list(zoning.z_intervals),
            name_template="{idx}-" + body_label,
            restrict_to=[body_label],
        ),
    ]


def _apply_chain_in_transformed_frame(
    chain: list[GroupingSpec],
    mesh: LnasFormat,
    transformation: TransformationConfig,
) -> GroupingResult:
    """Apply ``chain`` to a transformed copy of ``mesh``.

    Surfaces and triangle indices are unchanged by the geometric
    transformation; only vertex coordinates move, which is what the
    zoning binner consumes. This mirrors the legacy
    ``get_region_indexing`` flow.
    """
    mesh_for_binning = mesh.copy()
    mesh_for_binning.geometry.apply_transformation(transformation.get_geometry_transformation())
    return apply_groupings(mesh_for_binning, chain)


def _zoning_spec_of(geom_data: GeometryData) -> ByZoningGrouping:
    """Extract the single ByZoningGrouping from ``geom_data.spec_chain``."""
    for spec in geom_data.spec_chain:
        if isinstance(spec, ByZoningGrouping):
            return spec
    raise ValueError(
        "GeometryData.spec_chain must contain a ByZoningGrouping; "
        f"found {[type(s).__name__ for s in geom_data.spec_chain]}"
    )


# ---------------------------------------------------------------------------
# Tabulation + region helpers (public)
# ---------------------------------------------------------------------------


def get_region_definition_dataframe(geom_dict: dict[str, GeometryData]) -> pd.DataFrame:
    """Enumerate all zoning cells (including empty ones) per body.

    Columns: ``x_min, x_max, y_min, y_max, z_min, z_max, region_idx``.
    ``region_idx`` is suffixed with ``-{sfc_id}`` to match the legacy
    region-label format.
    """
    dfs = []
    for sfc_id, geom_data in geom_dict.items():
        spec = _zoning_spec_of(geom_data)
        df = _zoning_regions_df(spec)
        df["region_idx"] = df["region_idx"].astype(str) + f"-{sfc_id}"
        dfs.append(df)
    return pd.concat(dfs)


def _zoning_regions_df(spec: ByZoningGrouping) -> pd.DataFrame:
    """All Cartesian cells of ``spec`` as a DataFrame (linear-index ordered)."""
    from cfdmod.geometry.grouping.kinds.by_zoning import _regions

    rows = {
        "x_min": [],
        "x_max": [],
        "y_min": [],
        "y_max": [],
        "z_min": [],
        "z_max": [],
        "region_idx": [],
    }
    for linear, _ix, _iy, _iz, lo, hi in _regions(spec):
        rows["x_min"].append(lo[0])
        rows["x_max"].append(hi[0])
        rows["y_min"].append(lo[1])
        rows["y_max"].append(hi[1])
        rows["z_min"].append(lo[2])
        rows["z_max"].append(hi[2])
        rows["region_idx"].append(linear)
    return pd.DataFrame(rows)


def get_indexing_mask(mesh: LnasGeometry, df_regions: pd.DataFrame) -> np.ndarray:
    """Index each triangle in the mesh into the respective region.

    Kept for backwards compatibility with helpers in
    :mod:`cfdmod.pressure.run` (e.g. ``_bbox_corners_xy_cases``) that
    consume a legacy ``df_regions`` table directly. Triangles whose
    centroid does not fall in any region get ``-1``.
    """
    triangles = mesh.triangle_vertices
    centroids = np.mean(triangles, axis=1)
    triangles_region = np.full((triangles.shape[0],), -1, dtype=np.int32)

    for _, region in df_regions.iterrows():
        ll = np.array([region["x_min"], region["y_min"], region["z_min"]])
        ur = np.array([region["x_max"], region["y_max"], region["z_max"]])
        in_idx = np.all(
            np.logical_and(centroids >= ll, centroids < ur),
            axis=1,
        )
        triangles_region[in_idx] = region["region_idx"]

    return triangles_region


def tabulate_geometry_data(
    geom_dict: dict[str, GeometryData],
    mesh_areas: np.ndarray,
    mesh_normals: np.ndarray,
    transformation: TransformationConfig,  # kept for API compatibility, unused
) -> pd.DataFrame:
    """Long-form table with one row per body triangle.

    Columns: ``region_idx, point_idx, area, n_x, n_y, n_z``.

    Triangles in a body whose centroid falls outside every zoning cell
    are tagged with the legacy sentinel label ``f"-1-{body_label}"``.
    The dict key is expected to equal ``geom_data.body_label`` (the
    factories in this module enforce this; the long-form region label
    becomes ``f"{cell_idx}-{body_label}"``).
    """
    del transformation  # binning was already done when GeometryData was built

    dfs = []
    for _sfc_id, geom_data in geom_dict.items():
        labels = _per_triangle_region_labels(geom_data)

        df = pd.DataFrame()
        df["region_idx"] = labels
        df["point_idx"] = geom_data.triangles_idxs
        df["area"] = mesh_areas[geom_data.triangles_idxs].copy()
        df["n_x"] = mesh_normals[geom_data.triangles_idxs, 0].copy()
        df["n_y"] = mesh_normals[geom_data.triangles_idxs, 1].copy()
        df["n_z"] = mesh_normals[geom_data.triangles_idxs, 2].copy()
        dfs.append(df)
    return pd.concat(dfs)


def _per_triangle_region_labels(geom_data: GeometryData) -> np.ndarray:
    """Per-triangle ``region_idx`` strings for ``geom_data.triangles_idxs``.

    Triangles in no zoning cell get ``f"-1-{body_label}"``. The output
    length matches ``len(geom_data.triangles_idxs)`` and the order
    matches that array.
    """
    body_label = geom_data.body_label
    body_suffix = f"-{body_label}"
    parent_labels = geom_data.grouping.to_region_idx(unassigned="")

    labels = np.empty(len(geom_data.triangles_idxs), dtype=object)
    for i, parent_tri in enumerate(geom_data.triangles_idxs):
        lbl = parent_labels[int(parent_tri)]
        # to_region_idx joins membership with '|'. The cell label, when
        # present, is the piece that ends with body_suffix and is NOT
        # the bare body_label itself.
        cell = None
        for piece in lbl.split("|"):
            if piece.endswith(body_suffix) and piece != body_label:
                cell = piece
                break
        labels[i] = cell if cell is not None else f"-1{body_suffix}"
    return labels


def build_geometry_data(
    body_label: str,
    sfc_list: list[str],
    zoning: ZoningModel,
    mesh: LnasFormat,
    transformation: TransformationConfig | None = None,
) -> GeometryData:
    """Low-level builder for a GeometryData slice.

    Used by :func:`get_geometry_data` and :func:`get_ce_geometry_data`,
    and by tests that want to construct a GeometryData from a single
    surface set + a :class:`ZoningModel` directly. ``transformation``
    defaults to identity when omitted.
    """
    sfcs = list(sfc_list) if sfc_list else list(mesh.surfaces.keys())
    geom, geometry_idx = mesh.geometry_from_list_surfaces(surfaces_names=sfcs)
    chain = _build_chain(body_label=body_label, sfc_list=sfcs, zoning=zoning)
    grouping = _apply_chain_in_transformed_frame(
        chain, mesh, transformation or TransformationConfig()
    )
    return GeometryData(
        mesh=geom,
        triangles_idxs=geometry_idx,
        grouping=grouping,
        spec_chain=chain,
        body_label=body_label,
    )


def get_geometry_data(
    body_cfg: BodyConfig | MomentBodyConfig,
    sfc_list: list[str],
    mesh: LnasFormat,
    transformation: TransformationConfig,
) -> GeometryData:
    """Build a Cf/Cm GeometryData via the grouping pipeline.

    Honors ``body_cfg.groupings`` when set (the explicit YAML chain);
    otherwise builds the canonical ``[BySurface, ByZoning(sub_bodies)]``
    chain via ``BodyConfig.resolved_groupings``.

    ``transformation`` is the same one applied to the working geometry
    by the caller -- spatial cells are evaluated in that frame.

    Empty ``sfc_list`` means "every surface in the mesh", matching the
    legacy convention used by the synthetic-surface code path.
    """
    sfcs = list(sfc_list) if sfc_list else list(mesh.surfaces.keys())
    geom, geometry_idx = mesh.geometry_from_list_surfaces(surfaces_names=sfcs)
    chain = body_cfg.resolved_groupings(sfcs)
    grouping = _apply_chain_in_transformed_frame(chain, mesh, transformation)
    return GeometryData(
        mesh=geom,
        triangles_idxs=geometry_idx,
        grouping=grouping,
        spec_chain=chain,
        body_label=body_cfg.name,
    )


def combine_stats_data_with_mesh(
    mesh: LnasGeometry,
    region_idx_array: np.ndarray,
    data_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Combine compiled statistical data with surface meshing by indexing regions."""
    combined_df = pd.DataFrame()
    combined_df["point_idx"] = np.arange(len(mesh.triangle_vertices))
    combined_df["region_idx"] = region_idx_array
    combined_df = pd.merge(combined_df, data_stats, on="region_idx", how="left")
    combined_df.drop(columns=["region_idx"], inplace=True)
    return combined_df


# ---------------------------------------------------------------------------
# Ce surface geometry
# ---------------------------------------------------------------------------


def _get_surface_zoning(mesh: LnasGeometry, sfc: str, config: CeConfig) -> ZoningModel:
    """Per-surface zoning rule for Ce (no_zoning / exception / global).

    Identical to the legacy implementation: resolved against the
    *un-transformed* sub-mesh's normals (so planar surfaces drop the
    binning axis along their dominant normal), then offset by 0.1 on
    the outer edges.
    """
    if sfc in config.zoning.no_zoning:  # type: ignore
        zoning = ZoningModel(**{})
    elif sfc in config.zoning.surfaces_in_exception:  # type: ignore
        zoning = [
            cfg for cfg in config.zoning.exceptions.values() if sfc in cfg.surfaces  # type: ignore
        ][0]
    else:
        zoning = config.zoning.global_zoning  # type: ignore
        if len(np.unique(np.round(mesh.normals, decimals=2), axis=0)) == 1:
            ignore_axis = np.where(np.abs(mesh.normals[0]) == np.abs(mesh.normals[0]).max())[0][0]
            zoning = zoning.ignore_axis(ignore_axis)

    return zoning.offset_limits(0.1)


def get_ce_geometry_data(
    surface_dict: dict[str, list[str]], cfg: CeConfig, mesh: LnasFormat
) -> dict[str, GeometryData]:
    """Build per-surface GeometryData for Ce via the grouping pipeline."""
    geom_dict: dict[str, GeometryData] = {}
    for sfc_lbl, sfc_list in surface_dict.items():
        if sfc_lbl in cfg.zoning.exclude:  # type: ignore
            logger.debug(f"Surface {sfc_lbl} ignored!")
            continue
        surface_geom, sfc_triangles_idxs = mesh.geometry_from_list_surfaces(
            surfaces_names=sfc_list
        )
        zoning = _get_surface_zoning(mesh=surface_geom, sfc=sfc_lbl, config=cfg)

        chain = _build_chain(body_label=sfc_lbl, sfc_list=sfc_list, zoning=zoning)
        grouping = _apply_chain_in_transformed_frame(chain, mesh, cfg.transformation)

        geom_dict[sfc_lbl] = GeometryData(
            mesh=surface_geom,
            triangles_idxs=sfc_triangles_idxs,
            grouping=grouping,
            spec_chain=chain,
            body_label=sfc_lbl,
        )
    return geom_dict


def generate_regions_mesh(
    geom_data: GeometryData, cfg: CeConfig
) -> tuple[LnasGeometry, np.ndarray]:
    """Cut the surface mesh by zoning cells, returning the region mesh and indexing."""
    spec = _zoning_spec_of(geom_data)

    transformed_surface = geom_data.mesh.copy()
    transformed_surface.apply_transformation(cfg.transformation.get_geometry_transformation())

    regions_mesh = create_regions_mesh(
        transformed_surface,
        (
            tuple(spec.x_intervals),
            tuple(spec.y_intervals),
            tuple(spec.z_intervals),
        ),
    )
    df_regions = _zoning_regions_df(spec)
    regions_mesh_triangles_indexing = get_indexing_mask(mesh=regions_mesh, df_regions=df_regions)
    regions_mesh.apply_transformation(
        cfg.transformation.get_geometry_transformation(), invert_transf=True
    )
    return regions_mesh, regions_mesh_triangles_indexing
