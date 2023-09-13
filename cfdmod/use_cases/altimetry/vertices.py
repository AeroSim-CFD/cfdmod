import math

import numpy as np


class SectionVertices:
    def __init__(self, vertices: np.ndarray, plane_origin: np.ndarray, plane_normal: np.ndarray):
        """Object to store vertices and project them

        Args:
            vertices (np.ndarray): Point cloud of vertices
            plane_origin (np.ndarray): Origin of the plane used to generate the vertices
            plane_normal (np.ndarray): Normal of the plane used to generate the vertices
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
        position = self.pos.copy()
        position[:, :2] -= plane_origin[:2]  # Centralize according to origin but ignore z
        distance = np.apply_along_axis(lambda x: np.linalg.norm(x[:2]), 1, position)

        direction_func = (
            lambda x: -plane_normal[1] * x[0] + plane_normal[0] * x[1] + plane_normal[2] * x[2]
        )
        direction = np.apply_along_axis(direction_func, 1, position)
        direction /= abs(direction)

        self.projected_position = distance * direction
        self.offset = min(self.projected_position)

        # Offset section profile to 0, in the x axis, for the altimetry profile
        self.projected_position -= self.offset

        # Define plot limits
        self.minz = int(min(self.pos[:, 2]) / 50) * 50
        self.maxz = math.ceil(max(self.pos[:, 2]) / 50) * 50
        self.minx = min(self.projected_position)
        self.maxx = max(self.projected_position)
