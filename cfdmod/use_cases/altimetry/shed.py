import numpy as np


class AltimetryShed:
    def __init__(self, p0: np.ndarray, p1: np.ndarray, height: float = 15):
        self.start_coordinate = p0
        self.end_coordinate = p1
        self.height = height

    def project_shed_profile(
        self, plane_origin: np.ndarray, plane_normal: np.ndarray, offset: float
    ):
        """Project the shed section into a plane

        Args:
            plane_origin (np.ndarray): Plane origin coordinate
            plane_normal (np.ndarray): Plane normal vector
            offset (float): Value for offseting the shed to origin
        """
        # Get shed profile limits
        x_shed_start = self.start_coordinate[0] - plane_origin[0]
        x_shed_end = self.end_coordinate[0] - plane_origin[0]
        y_shed_start = self.start_coordinate[1] - plane_origin[1]
        y_shed_end = self.end_coordinate[1] - plane_origin[1]

        shed_length: float = (
            (x_shed_start - x_shed_end) ** 2 + (y_shed_start - y_shed_end) ** 2
        ) ** 0.5
        origin_offset: float = (x_shed_start**2 + y_shed_start**2) ** 0.5

        # Projection of shed start and end positions into the slicing plane
        direction = -plane_normal[1] * x_shed_start + plane_normal[0] * y_shed_start
        direction /= abs(direction)

        origin_offset *= direction
        z_start = self.start_coordinate[2]
        z_end = self.end_coordinate[2]

        # Generate block profile from shed projected coordinates
        g_x = [
            origin_offset,
            origin_offset,
            origin_offset + shed_length,
            origin_offset + shed_length,
        ]
        g_x -= offset
        g_y = [
            z_start,
            max(z_start, z_end) + self.height,
            max(z_start, z_end) + self.height,
            z_end,
        ]

        self.profile = tuple([np.array(g_x), np.array(g_y)])
