from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    coordinate: np.ndarray

    def __add__(self, point: Point) -> Point:
        return Point(self.coordinate + point.coordinate)

    def __sub__(self, point: Point) -> np.ndarray:
        return self.coordinate - point.coordinate
