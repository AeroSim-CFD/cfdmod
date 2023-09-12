from __future__ import annotations

import numpy as np
import trimesh

from cfdmod.use_cases.altimetry import AltimetryShed, SectionVertices


class AltimetrySection:
    def __init__(self, label: str, plane_origin: np.ndarray, plane_normal: np.ndarray):
        self.label = label
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal
        self.section_sheds: list[AltimetryShed] = []

    @classmethod
    def from_points(cls, label: str, p0: np.ndarray, p1: np.ndarray) -> AltimetrySection:
        """Generates a new AltimetrySection from the given points by calculating plane coordinates and normal

        Args:
            label (str): Label for the new AltimetrySection
            p0 (np.ndarray): First point of the plane
            p1 (np.ndarray): Second point of the plane

        Returns:
            AltimetrySection: Object representing the new AltimetrySection
        """
        x_o = (p0[0] + p1[0]) / 2  # Index 0 for x variable
        y_o = (p0[1] + p1[1]) / 2  # Index 1 for y variable
        z_o = (p0[2] + p1[2]) / 2  # Index 2 for z variable

        x_n = p0[0] - p1[0]  # Get delta x
        y_n = p0[1] - p1[1]  # Get delta z
        x_u = x_n / (x_n**2 + y_n**2) ** 0.5 if x_n != 0 else 0.0  # Normalization
        y_u = y_n / (x_n**2 + y_n**2) ** 0.5 if y_n != 0 else 0.0  # Normalization

        plane_origin = np.array([x_o, y_o, z_o])
        plane_normal = np.array([-y_u, x_u, 0])

        return AltimetrySection(label, plane_origin, plane_normal)

    def slice_surface(self, surface_mesh: trimesh.Trimesh):
        """Slices the surface and generates section vertices

        Args:
            surface_mesh (trimesh.Trimesh): Surface mesh
        """
        section_slice = surface_mesh.section(
            plane_origin=self.plane_origin, plane_normal=self.plane_normal
        )
        vertices = np.array(section_slice.to_dict()["vertices"])
        self.section_vertices = SectionVertices(vertices, self.plane_origin, self.plane_normal)

    def include_shed(self, shed: AltimetryShed):
        """Includes a shed for plotting

        Args:
            shed (AltimetryShed): Shed object
        """
        shed.project_shed_profile(
            self.plane_origin, self.plane_normal, self.section_vertices.offset
        )
        self.section_sheds.append(shed)
