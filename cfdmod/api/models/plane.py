from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    plane_normal: np.ndarray
    plane_origin: np.ndarray

    @classmethod
    def create_plane_from_points(cls) -> Plane:
        ...
