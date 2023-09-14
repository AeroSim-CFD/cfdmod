from __future__ import annotations

import numpy as np
import trimesh
from pydantic import Field

from cfdmod.use_cases.altimetry import SectionVertices, Shed, ShedProfile

__all__ = ["AltimetrySection"]


class AltimetrySection:
    """Representation of a section of altimetric profile and the corresponding sheds cut by it"""

    def __init__(
        self,
        label: str = Field(..., title="Section label", description="Label for altimetry section"),
        plane_origin: np.ndarray = Field(
            ...,
            title="Plane origin",
            description="Origin of the plane used to generate the section",
        ),
        plane_normal: np.ndarray = Field(
            ...,
            title="Plane normal",
            description="Normal direction of the plane used to generate the section",
        ),
    ):
        self.label = label
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal
        self.section_sheds: list[ShedProfile] = []

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
        shed_profile = ShedProfile(
            shed=shed,
            plane_origin=self.plane_origin,
            plane_normal=self.plane_normal,
            offset=self.section_vertices.offset,
        )
        self.section_sheds.append(shed_profile)
