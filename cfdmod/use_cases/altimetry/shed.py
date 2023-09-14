from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class Shed:
    """Representation of a standard shed for consulting cases"""

    start_coordinate: np.ndarray
    end_coordinate: np.ndarray
    height: float = 15


class ShedProfile:
    """Object representing the shed profile to be plotted in altimetric profile"""

    def __init__(
        self, shed: Shed, plane_origin: np.ndarray, plane_normal: np.ndarray, offset: float
    ):
        self.shed = shed
        self.profile = project_shed_profile(
            shed=shed, plane_origin=plane_origin, plane_normal=plane_normal, offset=offset
        )


def project_shed_profile(
    shed: Shed, plane_origin: np.ndarray, plane_normal: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project the shed into the section plane

    Args:
        shed (Shed): Shed object to be plotted in altimetric profile
        plane_origin (np.ndarray): Plane origin coordinate
        plane_normal (np.ndarray): Plane normal vector
        offset (float): Value for offsetting the shed to origin

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with the projected profile in x and y coordinates
    """
    # Get shed profile limits
    projected_start = shed.start_coordinate[:2] - plane_origin[:2]
    projected_end = shed.end_coordinate[:2] - plane_origin[:2]

    projected_length = np.linalg.norm(projected_end - projected_start)
    projected_offset = np.linalg.norm(projected_start)

    direction = -plane_normal[1] * projected_start[0] + plane_normal[0] * projected_start[1]
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

    return (g_x - offset, g_y)
