from typing import Annotated

import numpy as np
from lnas import TransformationsMatrix
from pydantic import BaseModel, Field


class TransformationConfig(BaseModel):
    """Geometry's transformation configurations"""

    translation: Annotated[
        tuple[float, ...],
        Field(
            (0, 0, 0),
            title="Translation",
            description="Translation values for geometry transformation",
        ),
    ]

    rotation: Annotated[
        tuple[float, ...],
        Field(
            (0, 0, 0),
            title="Rotation",
            description="Rotation angles (in radians) for geometry transformation",
        ),
    ]

    fixed_point: Annotated[
        tuple[float, ...],
        Field(
            (0, 0, 0),
            title="Fixed point",
            description="Point to use as reference to rotate and scale object",
        ),
    ]

    def __hash__(self) -> int:
        return hash((self.translation, self.rotation, self.fixed_point))

    def get_geometry_transformation(self):
        return TransformationsMatrix(
            angle=np.array(self.rotation, dtype=np.float64),
            translation=np.array(self.translation, dtype=np.float64),
            scale=np.array([1, 1, 1], dtype=np.float64),
            fixed_point=np.array(self.fixed_point, dtype=np.float64),
            always_update=False,
        )
