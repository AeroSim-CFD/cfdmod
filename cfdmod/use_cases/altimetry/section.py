from __future__ import annotations

import numpy as np
import trimesh

from cfdmod.use_cases.altimetry import SectionVertices, Shed

__all__ = ["AltimetrySection"]


class AltimetrySection:
    """Representation of a section of altimetric profile and the corresponding sheds cut by it"""

    def __init__(self, label: str, plane_origin: np.ndarray, plane_normal: np.ndarray):
        """Initialize an AltimetrySection from section plane description

        Args:
            label (str, optional): Label for altimetry section.
            plane_origin (np.ndarray, optional): Origin of the plane used to generate the section.
            plane_normal (np.ndarray, optional): Normal direction of the plane used to generate the section.
        """
        self.label = label
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal
        self.section_sheds: list[Shed] = []
        self.section_shed_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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
        p_n = p0[:2] - p1[:2]
        p_n /= np.linalg.norm(p_n)

        plane_origin = (p0 + p1) / 2

        # Rotation of p0_p1 direction in plane direction
        plane_normal = np.array([-p_n[1], p_n[0], 0])

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

    def include_shed(self, shed: Shed):
        """Includes a shed for plotting

        Args:
            shed (AltimetryShed): Shed object
        """
        shed_profile = self.project_shed_profile(shed)
        self.section_sheds.append(shed)
        self.section_shed_profiles[shed.shed_label] = shed_profile

    def project_shed_profile(self, shed: Shed) -> tuple[np.ndarray, np.ndarray]:
        """Project the shed into the section plane

        Args:
            shed (Shed): Shed object to be plotted in altimetric profile

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple with the projected profile in x and y coordinates
        """
        # Get shed profile limits
        projected_start = shed.start_coordinate[:2] - self.plane_origin[:2]
        projected_end = shed.end_coordinate[:2] - self.plane_origin[:2]

        projected_length = np.linalg.norm(projected_end - projected_start)
        projected_offset = np.linalg.norm(projected_start)

        direction = (
            -self.plane_normal[1] * projected_start[0] + self.plane_normal[0] * projected_start[1]
        )
        direction /= abs(direction)
        projected_offset *= direction

        max_shed_elevation = max(shed.start_coordinate[2], shed.end_coordinate[2])

        # Generate square profile from shed projected coordinates
        g_x = np.array(
            [
                projected_offset,
                projected_offset,
                projected_offset + projected_length,
                projected_offset + projected_length,
            ]
        )
        g_y = np.array(
            [
                shed.start_coordinate[2],
                max_shed_elevation + shed.height,
                max_shed_elevation + shed.height,
                shed.end_coordinate[2],
            ]
        )

        return (g_x - self.section_vertices.offset, g_y)
