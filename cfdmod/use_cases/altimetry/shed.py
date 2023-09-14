from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

__all__ = ["Shed", "ShedProfile"]


@dataclass(kw_only=True)
class Shed:
    """Representation of a standard shed for consulting cases"""

    start_coordinate: np.ndarray = Field(
        ...,
        title="Start coordinate",
        description="Start coordinate of the shed/building cut by the section",
    )
    end_coordinate: np.ndarray = Field(
        ...,
        title="End coordinate",
        description="End coordinate of the shed/building cut by the section",
    )
    # height: float = Field(
    #     15.0,
    #     title="Shed height",
    #     description="Size of the shed/building in z axis."
    #     + "Used to determine the limits when plotting, connecting the shed coordinates",
    # )
    height: float = 15.0


class ShedProfile:
    """Object representing the shed profile to be plotted in altimetric profile"""

    def __init__(
        self,
        shed: Shed = Field(
            ...,
            title="Shed object",
            description="Target shed to get the profile from",
        ),
        plane_origin: np.ndarray = Field(
            ...,
            title="Plane origin",
            description="Origin of the section plane for cutting the shed/building",
        ),
        plane_normal: np.ndarray = Field(
            ...,
            title="Plane normal",
            description="Normal direction of the section plane for cutting the shed/building",
        ),
        offset: float = Field(
            ...,
            title="Offset",
            description="Offset value for translating the shed."
            + "This value comes from the offset needed to centralize"
            + "the surface according to the origin",
        ),
    ):
        self.shed = shed
        self.profile = _project_shed_profile(
            shed=shed, plane_origin=plane_origin, plane_normal=plane_normal, offset=offset
        )


def _project_shed_profile(
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
