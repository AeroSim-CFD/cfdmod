import math

import numpy as np

__all__ = ["SectionVertices"]


class SectionVertices:
    """Object to store vertices and project them"""

    def __init__(
        self,
        vertices: np.ndarray,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
    ):
        """Initialize a section vertices object from the section vertices

        Args:
            vertices (np.ndarray, optional): Vertices generated from sectioning a surface.
            plane_origin (np.ndarray, optional): Origin of plane that defines the section.
            plane_normal (np.ndarray, optional): Normal direction of plane that defines the section.
        """
        if plane_normal[0] == 0:
            # Normal to x
            self.pos = np.array(sorted(vertices, key=lambda pos: pos[0]))
        else:
            # Not normal to x, so it can be sorted with y
            self.pos = np.array(sorted(vertices, key=lambda pos: pos[1]))

        self.project_into_plane(plane_origin, plane_normal)

    def project_into_plane(
        self,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
    ):
        """Projects the section point cloud onto the plane of the section

        Args:
            plane_origin (np.ndarray): Plane origin
            plane_normal (np.ndarray): PLane normal
        """

        def _direction_func(point: np.ndarray) -> float:
            return (
                -plane_normal[1] * point[0]
                + plane_normal[0] * point[1]
                + plane_normal[2] * point[2]
            )

        position = self.pos.copy()
        position[:, :2] -= plane_origin[:2]  # Centralize according to origin but ignore z
        distance = np.apply_along_axis(lambda x: np.linalg.norm(x[:2]), 1, position)

        direction = np.apply_along_axis(_direction_func, 1, position)
        direction /= abs(direction)

        self.projected_position = distance * direction
        self.offset = self.projected_position.min()

        # Offset section profile to 0, in the x axis, for the altimetry profile
        self.projected_position -= self.offset

        # Define plot limits
        self.minz = int(self.pos[:, 2].min() / 50) * 50
        self.maxz: int = math.ceil(self.pos[:, 2].max() / 50) * 50
        self.minx: float = self.projected_position.min()
        self.maxx: float = self.projected_position.max()
