from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

__all__ = ["AltimetryProbe"]


class AltimetryProbe(BaseModel):
    """Probe for altimetry

    Object containing specific data for altimetry use case.
    Used for defining building position and section plane
    """

    probe_coordinate: tuple[float, float, float] = Field(
        ...,
        title="Coordinate of altimetry probe",
        description="Spatial 3D coordinate that defines probe location",
    )
    building_label: str = Field(
        ...,
        title="Building label",
        description="Label of the building being cut by the probe",
    )
    section_label: str = Field(
        ...,
        title="Section label",
        description="Label of the section defined by the probe",
    )
    probe_label: str = Field(
        ...,
        title="Probe label",
        description="Label of the probe",
    )
    case_label: str = Field(
        ...,
        title="Case label",
        description="Label of the consulting case applied to the probe."
        + "Normally this label is used to define the wind direction",
    )

    @property
    def coordinate(self) -> np.ndarray:
        return np.array(self.probe_coordinate, dtype=np.float32)

    @classmethod
    def from_csv(cls, csv_path: pathlib.Path) -> list[AltimetryProbe]:
        probes_df = pd.read_csv(csv_path)
        probes_list: list[AltimetryProbe] = []

        if not all([x in probes_df.columns for x in ["X", "Y", "Z"]]):
            raise Exception("Missing probe coordinates columns")

        for probe_data in probes_df.iterrows():
            data = probe_data[1]  # Unpack data from dataframe iterrow
            building_label = data["building"] if data["building"] else "default"
            section_label = data["section"] if data["section"] else "default"
            case_label = str(data["case"]) if str(data["case"]) else "default"
            probe_label = data["probe_name"] if data["probe_name"] else f"Probe {len(probes_list)}"
            probe_coords = (data["X"], data["Y"], data["Z"])
            probes_list.append(
                AltimetryProbe(
                    probe_coordinate=probe_coords,
                    building_label=building_label,
                    section_label=section_label,
                    probe_label=probe_label,
                    case_label=case_label,
                )
            )
        return probes_list
