from typing import Dict, Literal, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict


class PointsInterceptor(BaseModel):
    """Model to hold points from interception between region intervals and mesh"""

    class Config:
        arbitrary_types_allowed = True

    # Dict keyed by position on respecting axis, which value is also a dictionary
    # keyed by a tuple with point indices with the coordinates of the interception
    x: Dict[float, Dict[Tuple[int, int], np.ndarray]] = {}
    y: Dict[float, Dict[Tuple[int, int], np.ndarray]] = {}
    z: Dict[float, Dict[Tuple[int, int], np.ndarray]] = {}

    def get_axis_dict(
        self, axis: Literal["x", "y", "z"]
    ) -> Dict[float, Dict[Tuple[int, int], np.ndarray]]:
        if axis == "x":
            return self.x
        elif axis == "y":
            return self.y
        elif axis == "z":
            return self.z

    def get_all_interception_points(self) -> np.ndarray:
        return np.array(
            [
                p
                for axis in [self.x, self.y, self.z]
                for interv in axis.values()
                for p in interv.values()
            ]
        )
