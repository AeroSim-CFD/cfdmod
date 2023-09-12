import pathlib

import numpy as np
import trimesh

from cfdmod.use_cases.altimetry import SectionVertices


class AltimetrySection:
    section_label: str
    plane_origin: np.ndarray
    plane_normal: np.ndarray

    def __init__(self, label: str, plane_origin: np.ndarray, plane_normal: np.ndarray):
        self.section_label = label
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal

    def slice_surface(self, surface_mesh: trimesh.Trimesh) -> SectionVertices:
        section_slice = surface_mesh.section(
            plane_origin=self.plane_origin, plane_normal=self.plane_normal
        )

        vertices = np.array(section_slice.to_dict()["vertices"])
        section_vertices = SectionVertices(vertices, self.plane_origin, self.plane_normal)

        return section_vertices
