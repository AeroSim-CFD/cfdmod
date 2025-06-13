from __future__ import annotations

import pathlib
import itertools
from cfdmod.logger import logger
import time
from typing import Callable, TypeVar

from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import pathlib
from multiprocessing import Pool, cpu_count


from cfdmod.use_cases.hfpi import solver

T = TypeVar("T")


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
    """Dimensions specification for structure"""

    base: float
    height: float


class HFPICaseParameters(BaseModel, frozen=True):
    """Parameters for an HFPI case analysis"""

    direction: float
    xi: float
    recurrence_period: float

    def get_results_filename(self, base_folder: pathlib.Path):
        return base_folder / f"dir{self.direction}_xi{self.xi}_rp{self.recurrence_period}.pickle"


def solve_hfpi_case(hfpi_analysis: HFPIAnalysisHandler, parameters: HFPICaseParameters):
    """Solve HFPI for system and save it to disk"""

    t0 = time.time()
    logger.info(f"Solving HFPI for: {parameters.json()}")
    hfpi_params = hfpi_analysis.generate_hfpi_solver_params(parameters)
    hfpi_results = solver.solve_hfpi(
        structural_data=hfpi_params.structural_data,
        dim_data=hfpi_params.dim_data,
        forces=hfpi_params.forces,
    )
    logger.info(f"Solved HFPI in {time.time()-t0:.2f}s for: {parameters.json()}!")

    path_save = parameters.get_results_filename(hfpi_analysis.save_folder)
    hfpi_results.save(path_save)
    logger.info(f"Saved HFPI results to {path_save.as_posix()}")
    return hfpi_results


def _wrapper_solve_hfpi_case(args: tuple[HFPIAnalysisHandler, HFPICaseParameters]):
    return solve_hfpi_case(args[0], args[1])


class _HFPIParams(BaseModel):
    structural_data: solver.HFPIStructuralData
    dim_data: solver.HFPIDimensionalData
    forces: solver.HFPIForcesData


class HFPIAnalysisHandler(BaseModel):
    """Full analysis for an HFPI case"""

    wind_analytics: WindAnalysis
    dimensions: DimensionSpecs
    structural_data: solver.HFPIStructuralData
    directional_forces: dict[float, solver.HFPIForcesData]
    save_folder: pathlib.Path

    results: dict[HFPICaseParameters, solver.HFPIResults] = Field(default_factory=dict)

    def generate_hfpi_solver_params(self, parameters: HFPICaseParameters):
        forces = self.directional_forces[parameters.direction]
        if forces.is_scaled:
            raise ValueError("Forces should not be scaled before generating HFPI solver")

        dim = self.dimensions
        U_h = self.wind_analytics.get_U_H(
            dim.height, parameters.direction, parameters.recurrence_period
        )
        dim_data = solver.HFPIDimensionalData(
            U_H=U_h,
            xi=parameters.xi,
            base=dim.base,
            height=dim.height,
        )

        return _HFPIParams(
            structural_data=self.structural_data,
            forces=forces,
            dim_data=dim_data,
        )

    def generate_combined_parameters(
        self, *, directions: list[float], xis: list[float], recurrence_periods: list[float]
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

    def solve_all(self, parameters: list[HFPICaseParameters], max_workers: int | None = None):
        args = [(self, param) for param in parameters]
        # Avoid RAM explosion
        n_lim_workers = 10
        n_proc = min(n_lim_workers, max_workers or cpu_count())
        with Pool(processes=n_proc) as pool:
            pool.map(_wrapper_solve_hfpi_case, args)


class HFPIFullResults(BaseModel):
    results_folder: pathlib.Path

    results: dict[HFPICaseParameters, solver.HFPIResults] = Field(default_factory=dict)

    def load_result(self, parameters: HFPICaseParameters):
        filename = parameters.get_results_filename(self.results_folder)
        self.results[parameters] = solver.HFPIResults.load(filename)

    @classmethod
    def load_all_results(cls, parameters: list[HFPICaseParameters], results_folder: pathlib.Path):
        results = cls(results_folder=results_folder)
        for p in parameters:
            results.load_result(p)
        return results

    def join_by(self, callback: Callable[[HFPICaseParameters], T]) -> dict[T, HFPIFullResults]:
        joined_values = {}
        for p in self.results:
            key = callback(p)
            if key not in joined_values:
                joined_values[key] = []
            joined_values[key].append((p, self.results[p]))
        return {
            k: HFPIFullResults(
                results_folder=self.results_folder, results={p: r for p, r in res_list}
            )
            for k, res_list in joined_values.items()
        }

    def join_by_recurrence_period(self):
        return self.join_by(lambda params: params.recurrence_period)

    def join_by_xi(self):
        return self.join_by(lambda params: params.xi)

    def filter_by_xi(self, xi: float):
        return self.join_by_xi().get(xi)

    def filter_by_recurrence_period(self, recurrence_period: float):
        return self.join_by_recurrence_period().get(recurrence_period)

    def get_max_acceleration(self):
        return max(res.get_max_acceleration() for res in self.results.values())

    def get_max_static_forces(self):
        max_static_eq = {}
        for r in self.results.values():
            r_max_static_eq = r.get_max_forces_static_eq()
            if len(max_static_eq) == 0:
                max_static_eq = r_max_static_eq
                continue
            for k in max_static_eq:
                max_static_eq[k] = max(max_static_eq[k], r_max_static_eq[k])
        return max_static_eq

    def get_max_acceleration_by_recurrence_period(self):
        res = self.join_by_recurrence_period()
        return {k: r.get_max_acceleration() for k, r in res.items()}
