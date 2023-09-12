import math
from typing import Tuple

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
            data = np.array(sorted(vertices, key=lambda pos: pos[0]))
        else:
            # Not normal to x, so it can be sorted with y
            data = np.array(sorted(vertices, key=lambda pos: pos[1]))
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.z = data[:, 2]
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
        projected_position = []
        for x, y, z in zip(self.x, self.y, self.z):
            # Centralize according to origin
            x -= plane_origin[0]
            y -= plane_origin[1]

            # Coordinate decomposition
            distance = (x**2 + y**2) ** 0.5
            direction = (
                -plane_normal[1] * x + plane_normal[0] * y + plane_normal[2] * z
            )  # Scalar product for direction
            direction /= abs(direction)  # Normalization
            projected_position.append(distance * direction)

        self.projected_position = np.array(projected_position)
        self.offset = min(self.projected_position)
        self.projected_position -= (
            self.offset
        )  # Offset section profile to 0, in the x axis, for the altimetry profile

        self.minz = int(min(self.z) / 50) * 50
        self.maxz = math.ceil(max(self.z) / 50) * 50
        self.minx = min(self.projected_position)
        self.maxx = max(self.projected_position)
