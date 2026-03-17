import pathlib

import lnas
import numpy as np

from cfdmod.api.geometry.STL import export_stl
from cfdmod.logger import logger
from cfdmod.use_cases.loft.functions import generate_loft_surface
from cfdmod.use_cases.loft.parameters import LoftCaseConfig


def run_loft(
    cfg: LoftCaseConfig,
    geom: lnas.LnasGeometry,
    output_path: pathlib.Path,
):
    """Orchestrate loft surface generation for all cases, writing STL files to output_path.

    Args:
        cfg (LoftCaseConfig): Loft configuration with all case definitions.
        geom (lnas.LnasGeometry): Source terrain geometry.
        output_path (pathlib.Path): Base directory where output STL files are written.
            Each case is saved to output_path/<case_label>/loft.stl.
    """
    for case_lbl, loft_params in cfg.cases.items():
        if case_lbl == "default":
            continue
        logger.info(f"Generating loft for {case_lbl}...")
        loft_geom = generate_loft_surface(
            geom=geom,
            loft_radius=loft_params.loft_radius,
            loft_z_pos=loft_params.upwind_elevation,
        )
        loft_tris = loft_geom.triangle_vertices
        u = loft_tris[:, 1] - loft_tris[:, 0]
        v = loft_tris[:, 2] - loft_tris[:, 0]
        loft_normals = np.cross(u, v)
        export_stl(output_path / f"{case_lbl}" / "loft.stl", loft_tris, loft_normals)
        logger.info(f"Generated loft for {case_lbl}!")
