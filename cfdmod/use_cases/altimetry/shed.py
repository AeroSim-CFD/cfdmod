import numpy as np
from pydantic import BaseModel, Field, ConfigDict

__all__ = ["Shed"]


class Shed(BaseModel):
    """Representation of a standard shed for consulting cases"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    shed_label: str = Field(
        ...,
        title="Building label",
        description="Label of the shed/building represented by the object",
    )
    height: float = Field(
        15.0,
        title="Shed height",
        description="Size of the shed/building in z axis."
        + "Used to determine the limits when plotting, connecting the shed coordinates",
    )
