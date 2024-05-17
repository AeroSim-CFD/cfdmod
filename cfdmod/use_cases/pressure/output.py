from dataclasses import dataclass

import pandas as pd

from cfdmod.api.vtk.write_vtk import merge_polydata, write_polydata
from cfdmod.use_cases.pressure.geometry import ProcessedEntity
from cfdmod.use_cases.pressure.path_manager import PathManagerBody
from cfdmod.utils import create_folders_for_file


@dataclass
class CommonOutput:
    processed_entities: list[ProcessedEntity]
    excluded_entities: list[ProcessedEntity]
    data_df: pd.DataFrame
    stats_df: pd.DataFrame
    region_indexing_df: pd.DataFrame
    region_definition_df: pd.DataFrame

    def save_outputs(self, cfg_label: str, path_manager: PathManagerBody):
        # Output 1-A: Region indexing dataframe
        region_indexing_path = path_manager.get_region_indexing_path(cfg_lbl=cfg_label)
        create_folders_for_file(region_indexing_path)
        self.region_indexing_df.to_hdf(
            region_indexing_path, key="Region", mode="w", index=False, format="table"
        )
        # Output 1-B: Region definition dataframe
        region_definition_path = path_manager.get_region_definition_path(cfg_lbl=cfg_label)
        create_folders_for_file(region_definition_path)
        self.region_definition_df.to_hdf(
            region_definition_path, key="Region", mode="w", index=False, format="table"
        )

        # Output 2: Time series dataframe
        timeseries_path = path_manager.get_timeseries_path(cfg_lbl=cfg_label)
        self.data_df.to_hdf(
            timeseries_path, key="Time_series", mode="w", index=False, format="table"
        )

        # Output 3: Statistics dataframe
        stats_path = path_manager.get_stats_path(cfg_lbl=cfg_label)
        self.stats_df.to_hdf(stats_path, key="Statistics", mode="w", index=False, format="table")

        # Output 4: VTK polydata
        all_entities = self.processed_entities + self.excluded_entities
        merged_polydata = merge_polydata([entity.polydata for entity in all_entities])
        write_polydata(path_manager.get_vtp_path(cfg_lbl=cfg_label), merged_polydata)
