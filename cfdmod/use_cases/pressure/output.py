from dataclasses import dataclass

import pandas as pd

from cfdmod.api.vtk.write_vtk import merge_polydata, write_polydata
from cfdmod.use_cases.pressure.force.Cf_config import CfConfig
from cfdmod.use_cases.pressure.geometry import ProcessedEntity
from cfdmod.use_cases.pressure.moment.Cm_config import CmConfig
from cfdmod.use_cases.pressure.path_manager import PathManagerBody
from cfdmod.use_cases.pressure.shape.Ce_config import CeConfig
from cfdmod.utils import create_folders_for_file


@dataclass
class CommonOutput:
    processed_entities: list[ProcessedEntity]
    excluded_entities: list[ProcessedEntity]
    data_df: pd.DataFrame
    stats_df: pd.DataFrame
    regions_df: pd.DataFrame

    def save_outputs(
        self, cfg_label: str, cfg: CfConfig | CeConfig | CmConfig, path_manager: PathManagerBody
    ):
        # Output 1: Regions dataframe
        cfg_hash = cfg.sha256()
        regions_path = path_manager.get_regions_df_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)
        create_folders_for_file(regions_path)
        self.regions_df.to_hdf(path_or_buf=regions_path, key="Regions", mode="w", index=False)

        # Output 2: Time series dataframe
        timeseries_path = path_manager.get_timeseries_df_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)
        self.data_df.to_hdf(path_or_buf=timeseries_path, key="Time_series", mode="w", index=False)

        # Output 3: Statistics dataframe
        stats_path = path_manager.get_stats_df_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash)
        self.stats_df.to_hdf(path_or_buf=stats_path, key="Statistics", mode="w", index=False)

        # Output 4: VTK polydata
        all_entities = self.processed_entities + self.excluded_entities
        merged_polydata = merge_polydata([entity.polydata for entity in all_entities])
        write_polydata(
            path_manager.get_vtp_path(cfg_lbl=cfg_label, cfg_hash=cfg_hash), merged_polydata
        )
