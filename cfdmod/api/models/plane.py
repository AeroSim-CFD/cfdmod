from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfdmod.api.models import Line, Point


@dataclass
class Plane:
    plane_origin: np.ndarray
    plane_normal: np.ndarray

    @classmethod
    def create_plane_from_points(cls, p0: Point, p1: Point, p2: Point) -> Plane:
        plane_origin = (p0 + p1 + p2) / 3
        p0_p1 = p1 - p0
        p0_p2 = p2 - p0
        plane_normal = np.cross(p0_p1, p0_p2)
        plane_normal /= np.linalg.norm(plane_normal)

        return Plane(plane_origin, plane_normal)

    @classmethod
    def create_plane_from_line_and_point(cls, p0: Point, line: Line) -> Plane:
        plane_origin = (p0.coordinate + line.p0.coordinate + line.p1.coordinate) / 3
        p0_line = line.p0.coordinate - p0.coordinate
        line_dir = line.p1.coordinate - line.p0.coordinate
        plane_normal = np.cross(p0_line, line_dir)
        plane_normal /= np.linalg.norm(plane_normal)

        return Plane(plane_origin, plane_normal)
