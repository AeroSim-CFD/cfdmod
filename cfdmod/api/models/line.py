from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfdmod.api.models import Point


@dataclass
class Line:
    p0: Point
    p1: Point
    resolution: int

    @classmethod
    def create_line_from_direction(
        cls, p0: Point, direction: np.ndarray, size: float, resolution: int
    ) -> Line:
        if direction.shape != p0.coordinate.shape:
            raise ValueError("Direction dimensions does not match with coordinates dimensions")
        p1 = Point(p0.coordinate + direction * size)
        return Line(p0, p1, resolution)
