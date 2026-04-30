__all__ = ["S1Probe"]

from typing import Annotated, Sequence

from pydantic import BaseModel, Field


class S1Probe(BaseModel):
    p1: Annotated[
        Sequence[float],
        Field(..., title="Start point", description="Coordinate for the probe start point"),
    ]
    p2: Annotated[
        Sequence[float],
        Field(..., title="End point", description="Coordinate for the probe end point"),
    ]
    numPoints: Annotated[
        int,
        Field(
            ...,
            title="Number of points",
            description="Number of discrete points that define the probe",
        ),
    ]
