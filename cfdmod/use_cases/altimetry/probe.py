from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(kw_only=True)
class AltimetryProbe:
    """Probe for altimetry

    Object containing specific data for altimetry use case. 
    Used for defining building position and section plane
    """

    coordinate: np.ndarray
    building_label: str
    section_label: str
    probe_label: str
    case_label: str

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
            probe_coords = np.array([data["X"], data["Y"], data["Z"]])
            probes_list.append(
                AltimetryProbe(
                    coordinate=probe_coords,
                    building_label=building_label,
                    section_label=section_label,
                    probe_label=probe_label,
                    case_label=case_label,
                )
            )
        return probes_list
