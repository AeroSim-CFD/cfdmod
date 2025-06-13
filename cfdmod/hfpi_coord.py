from __future__ import annotations

import pathlib
import itertools
import pickle

from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import pathlib
from multiprocessing import Pool, cpu_count


import cfdmod.hfpi_mock as hfpi


class WindAnalysis(BaseModel):
    """Data for wind analysis and calculation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    directional_velocity_multiplier: dict[float, float]
    directional_roughness_cats: pd.DataFrame

    def S2(self, direction: float):
        # parameters from NBR 6123, mean speed of 10min
        Fr = 0.69
        p = {"I": 0.095, "II": 0.15, "III": 0.185, "IV": 0.23, "V": 0.31}
        b = {"I": 1.23, "II": 1.00, "III": 0.86, "IV": 0.71, "V": 0.50}

    def S3(self, recurrence_period: float):
        return 0.54 * (0.994 / recurrence_period) ** -0.157

    def get_U_H(self, height: float, direction: float, recurrence_period: float) -> float:
        ...
        return 45


class DimensionSpecs(BaseModel):
    base: float
    height: float


class HFPICaseParameters(BaseModel):
    direction: float
    xi: float
    recurrence_period: float

    def get_results_filename(self, base_folder: pathlib.Path):
        return base_folder / f"dir{self.direction}_xi{self.xi}_rp{self.recurrence_period}.pickle"


def solve_hfpi_case(hfpi_analysis: HFPIAnalysisFull, parameters: HFPICaseParameters):
    hfpi_solver = hfpi_analysis.generate_hfpi_solver(parameters)
    hfpi_results = hfpi_solver.solve_hfpi()
    path_save = parameters.get_results_filename(hfpi_analysis.save_folder)
    hfpi_results.save(path_save)

def _wrapper_solve_hfpi_case(args: tuple[HFPIAnalysisFull, HFPICaseParameters]):
    return solve_hfpi_case(args[0], args[1])


class HFPIAnalysisFull(BaseModel):
    wind_analytics: WindAnalysis
    dimensions: DimensionSpecs
    structural_data: hfpi.HFPIStructuralData
    directional_forces: dict[float, hfpi.HFPIForcesData]
    save_folder: pathlib.Path

    results: dict[HFPICaseParameters, hfpi.HFPIResults] = Field(default_factory=dict)

    def generate_hfpi_solver(self, parameters: HFPICaseParameters):
        forces = self.directional_forces[parameters.direction]
        if forces.is_scaled:
            raise ValueError("Forces should not be scaled before generating HFPI solver")

        dim = self.dimensions
        U_h = self.wind_analytics.get_U_H(
            dim.height, parameters.direction, parameters.recurrence_period
        )
        dim_data = hfpi.HFPIDimensionalData(
            U_H=U_h,
            xi=parameters.xi,
            base=dim.base,
            height=dim.height,
        )
        
        return hfpi.HFPISolver(
            structural_data=self.structural_data,
            forces=forces,
            dim_data=dim_data,
        )

    def generate_combined_parameters(
        self, *, directions: list[ float], xis: list[float], recurrence_periods: list[float]
    ) -> list[HFPICaseParameters]:
        cases_parameters = [
            HFPICaseParameters(
                direction=direction,
                xi=xi,
                recurrence_period=period,
            )
            for direction, xi, period in itertools.product(*[directions, xis, recurrence_periods])
        ]
        return cases_parameters

    def load_results(self, parameters: HFPICaseParameters):
        filename = parameters.get_results_filename(self.save_folder)
        self.results[parameters] = hfpi.HFPIResults.load(filename)

    def solve_all(self, parameters: list[HFPICaseParameters], max_workers: int | None = None):
        args = [(self, param) for param in parameters]
        # Avoid RAM explosion
        n_lim_workers = 10
        n_proc = min(n_lim_workers, max_workers or cpu_count())
        with Pool(processes=n_proc) as pool:
            pool.map(_wrapper_solve_hfpi_case, args)