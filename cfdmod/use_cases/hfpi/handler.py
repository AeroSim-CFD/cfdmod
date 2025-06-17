from __future__ import annotations

import pathlib
import itertools
from cfdmod.logger import logger
import time
from typing import Callable, TypeVar, Literal
from collections import defaultdict
import math

from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import pathlib
from multiprocessing import Pool, cpu_count

import numpy as np
from cfdmod.use_cases.hfpi import solver

T = TypeVar("T")


class WindAnalysis(BaseModel):
    """Data for wind analysis and calculation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pandas with keys: wind_direction, catI, catII, catIII, catIV, catV, Kd
    # Kd is optional and defaults to read, it defaults to one
    directional_data: pd.DataFrame

    @classmethod
    def build(cls, data_csv: pathlib.Path):
        df = pd.read_csv(data_csv, index_col=None)
        req_keys = ["wind_direction","I","II","III","IV","V"]
        if not solver._validate_keys_df(df, req_keys):
            raise KeyError("Not all required keys are in wind CSV. "
                           f"Required ones are: {req_keys}, found {list(df.columns)}")
        if("Kd" not in df.columns):
            df["Kd"] = 1
        df = df[req_keys + ["Kd"]]
        df.sort_values(by=["wind_direction"], inplace=True)
        return WindAnalysis(directional_data=df)

    def S2(self, direction: float):
        # parameters from NBR 6123, mean speed of 10min
        Fr = 0.69
        p = {"I": 0.095, "II": 0.15, "III": 0.185, "IV": 0.23, "V": 0.31}
        b = {"I": 1.23, "II": 1.00, "III": 0.86, "IV": 0.71, "V": 0.50}
        
        df = self.directional_data
        row = df.loc[df["wind_direction"] == direction].squeeze()
        sum_p = sum(row[k] * p[k] for k in p.keys())
        sum_b = sum(row[k] * b[k] for k in b.keys())
        return sum_b

    def S3(self, recurrence_period: float):
        return 0.54 * (0.994 / recurrence_period) ** -0.157
    
    def height_velocity(self, height: float, direction: float):
        z_0 = 0.05     # Roughness length for open terrain (m)
        v_b = 28       # Basic wind speed (m/s)
        c_0 = 1.0      # Orography factor
        # Eurocode's kr for z0 = 0.05
        k_r = 0.19     # Approximate kr for open terrain
        # Compute roughness factor and final wind speed
        c_r = k_r * math.log(height / z_0)

        return c_r * c_0 * v_b

    def get_U_H(self, height: float, direction: float, recurrence_period: float) -> float:
        ...
        # Just for test
        return self.height_velocity(height, direction) * self.S2(direction) * self.S3(recurrence_period)


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


def _get_global_stats_dct_float(dcts: list[dict[str, float]], stats_type: Literal["min", "max", "mean"]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)

    for d in dcts:
        for k, v in d.items():
            grouped[k].append(v)

    result: dict[str, float] = {}
    for k, values in grouped.items():
        if stats_type == "min":
            result[k] = min(values)
        elif stats_type == "max":
            result[k] = max(values)
        elif stats_type == "mean":
            result[k] = sum(values) / len(values)
        else:
            raise ValueError(f"Invalid stats_type: {stats_type!r}. Must be 'min', 'max', or 'mean'.")
    
    return result


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

    def join_by_direction(self):
        return self.join_by(lambda params: params.direction)

    def filter_by_xi(self, xi: float):
        return self.join_by_xi()[xi]

    def filter_by_recurrence_period(self, recurrence_period: float):
        return self.join_by_recurrence_period()[recurrence_period]

    def get_max_acceleration(self):
        return max(res.get_max_acceleration() for res in self.results.values())

    def get_max_acceleration_by_recurrence_period(self):
        res = self.join_by_recurrence_period()
        return {k: r.get_max_acceleration() for k, r in res.items()}

    def get_stats_global_forces_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [v.get_stats_global_forces_static_eq(stats_type) for k, v in self.results.items()]
        return _get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_moments_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [v.get_stats_global_moments_static_eq(stats_type) for k, v in self.results.items()]
        return _get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [v.static_results.get_stats_global_forces_static(stats_type) for k, v in self.results.items()]
        return _get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [v.static_results.get_stats_global_moments_static(stats_type) for k, v in self.results.items()]
        return _get_global_stats_dct_float(dcts, stats_type)

    def get_global_peaks_by_direction(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Get global peaks per direction of results
        
        Returns results as [load_type] = DataFrame["direction", stats_type]

        load_type = "forces_static", "moments_static", "forces_static_eq", "moments_static_eq"
        stats_type = "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"
        """
        res = self.join_by_direction()

        s = ["min", "max"]
        axis = ["x", "y", "z"]

        # Dict as [load_type][(stats_type, direction)] = value
        joined_res: dict[str, dict[tuple[str, float], float]] = {
            "forces_static": {},
            "moments_static": {},
            "forces_static_eq": {},
            "moments_static_eq": {},
        }
        for d, r in res.items():
            calls = [
                ("forces_static", r.get_stats_global_forces_static),
                ("moments_static", r.get_stats_global_moments_static),
                ("forces_static_eq", r.get_stats_global_forces_static_eq),
                ("moments_static_eq", r.get_stats_global_moments_static_eq),
            ]
            for name, c in calls:
                min_vals = c("min")
                max_vals = c("max")
                mean_vals = c("mean")
                for ax in axis:
                    joined_res[name][f"min_{ax}", d] = min_vals[ax]
                    joined_res[name][f"max_{ax}", d] = max_vals[ax]
                    joined_res[name][f"mean_{ax}", d] = mean_vals[ax]

        dct_dfs: dict[str, pd.DataFrame] = {}
        for k, dct_res in joined_res.items():
            all_directions = sorted(set(d for _, d in dct_res.keys()))
            dct = {"direction": all_directions}
            for stats in ("min", "max", "mean"):
                for ax in axis:
                    values = []
                    for d in all_directions:
                        v = dct_res[f"{stats}_{ax}", d]
                        values.append(v)
                    dct[f"{stats}_{ax}"] = values
            dct_dfs[k] = pd.DataFrame(dct)

        return dct_dfs
