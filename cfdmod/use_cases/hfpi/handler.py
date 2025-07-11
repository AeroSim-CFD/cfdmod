from __future__ import annotations

import itertools
import pathlib
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Callable, Literal, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from cfdmod.logger import logger
from cfdmod.use_cases.hfpi import common, dynamic, static

T = TypeVar("T")


class WindAnalysis(BaseModel):
    """Data for wind analysis and calculation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pandas with keys: wind_direction, I, II, III, IV, V, Kd
    # Kd is optional and defaults to read, it defaults to one
    directional_data: pd.DataFrame
    V0: float
    U_H_overwrite: float | None = None

    @classmethod
    def build(cls, data_csv: pathlib.Path, V0: float, U_H_overwrite: float | None = None):
        df = pd.read_csv(data_csv, index_col=None)
        req_keys = ["wind_direction", "I", "II", "III", "IV", "V"]
        if not common.validate_keys_df(df, req_keys):
            raise KeyError(
                "Not all required keys are in wind CSV. "
                f"Required ones are: {req_keys}, found {list(df.columns)}"
            )
        if "Kd" not in df.columns:
            df["Kd"] = 1
        df = df[req_keys + ["Kd"]]
        df.sort_values(by=["wind_direction"], inplace=True)
        return WindAnalysis(directional_data=df, V0=V0, U_H_overwrite=U_H_overwrite)

    def S2(self, height: float, direction: float):
        # parameters from NBR 6123, mean speed of 10min
        Fr = 0.69
        p = {"I": 0.095, "II": 0.15, "III": 0.185, "IV": 0.23, "V": 0.31}
        b = {"I": 1.23, "II": 1.00, "III": 0.86, "IV": 0.71, "V": 0.50}

        df = self.directional_data
        row = df.loc[df["wind_direction"] == direction].squeeze()
        sum_p = sum(row[k] * p[k] for k in p.keys())
        sum_b = sum(row[k] * b[k] for k in b.keys())
        return Fr * sum_b * (height / 10) ** sum_p

    def S3(self, recurrence_period: float):
        return 0.54 * (0.994 / recurrence_period) ** -0.157

    def get_U_H(
        self, height: float, direction: float, recurrence_period: float, use_kd: bool = True
    ) -> float:
        if self.U_H_overwrite is not None:
            return self.U_H_overwrite

        df = self.directional_data
        row = df.loc[df["wind_direction"] == direction].squeeze()
        V0 = self.V0
        kd = row["Kd"] if use_kd else 1
        S2 = self.S2(height, direction)
        S3 = self.S3(recurrence_period)
        return V0 * kd * S2 * S3


class DimensionSpecs(BaseModel):
    """Dimensions specification for structure"""

    base: float
    height: float


class HFPICaseParameters(BaseModel, frozen=True):
    """Parameters for an HFPI case analysis"""

    direction: float
    xi: float
    recurrence_period: float
    use_kd: bool
    structural_data: dynamic.HFPIStructuralData = Field(exclude=True)

    def get_results_filename(self, base_folder: pathlib.Path):
        return base_folder / f"dir{self.direction}_xi{self.xi}_rp{self.recurrence_period}.pickle"


def solve_hfpi_case(hfpi_analysis: MultipleAnalysisHandler, parameters: HFPICaseParameters):
    """Solve HFPI for system and save it to disk"""

    hfpi_params = hfpi_analysis.generate_hfpi_solver_params(parameters)
    floors_heights = hfpi_params.structural_data.df_floors["Z"].to_numpy()

    t0 = time.time()
    logger.info(f"Solving static for: {parameters.model_dump_json()}")
    static_results = static.solve_static_forces(
        forces=hfpi_params.forces, dim_data=hfpi_params.dim_data, floors_heights=floors_heights
    )
    logger.info(f"Solved static in {time.time()-t0:.2f}s for: {parameters.model_dump_json()}!")

    logger.info(f"Solving HFPI for: {parameters.model_dump_json()}")
    hfpi_results = dynamic.solve_hfpi(
        structural_data=hfpi_params.structural_data,
        dim_data=hfpi_params.dim_data,
        forces=hfpi_params.forces,
        xi=parameters.xi,
    )
    logger.info(f"Solved HFPI in {time.time()-t0:.2f}s for: {parameters.model_dump_json()}!")

    res = ResultType(static_res=static_results, dynamic_res=hfpi_results)
    path_save = parameters.get_results_filename(hfpi_analysis.save_folder)
    res.save(path_save)

    logger.info(f"Saved HFPI results to {path_save.as_posix()}")
    return hfpi_results


def _wrapper_solve_hfpi_case(args: tuple[MultipleAnalysisHandler, HFPICaseParameters]):
    return solve_hfpi_case(args[0], args[1])


class _HFPIParams(BaseModel):
    structural_data: dynamic.HFPIStructuralData
    dim_data: static.DimensionalData
    forces: static.StaticForcesData
    xi: float


class MultipleAnalysisHandler(BaseModel):
    """Full analysis for an HFPI case"""

    wind_analytics: WindAnalysis
    dimensions: DimensionSpecs
    directional_forces: dict[float, static.StaticForcesData]
    save_folder: pathlib.Path

    results: dict[HFPICaseParameters, ResultType] = Field(default_factory=dict)

    def generate_hfpi_solver_params(self, parameters: HFPICaseParameters):
        forces = self.directional_forces[parameters.direction]
        if forces.is_scaled:
            raise ValueError("Forces should not be scaled before generating HFPI solver")

        dim = self.dimensions
        U_h = self.wind_analytics.get_U_H(
            dim.height,
            parameters.direction,
            parameters.recurrence_period,
            use_kd=parameters.use_kd,
        )
        dim_data = static.DimensionalData(
            U_H=U_h,
            base=dim.base,
            height=dim.height,
        )

        return _HFPIParams(
            structural_data=parameters.structural_data,
            dim_data=dim_data,
            forces=forces,
            xi=parameters.xi,
        )

    def generate_combined_parameters(
        self,
        *,
        structural_data: dynamic.HFPIStructuralData,
        directions: list[float],
        xis: list[float],
        use_kd: list[bool],
        recurrence_periods: list[float],
    ) -> list[HFPICaseParameters]:
        cases_parameters = [
            HFPICaseParameters(
                direction=direction,
                recurrence_period=period,
                use_kd=kd,
                xi=xi,
                structural_data=structural_data,
            )
            for direction, xi, kd, period in itertools.product(
                *[directions, xis, use_kd, recurrence_periods]
            )
        ]
        return cases_parameters

    def solve_all(self, parameters: list[HFPICaseParameters], max_workers: int | None = None):
        args = [(self, param) for param in parameters]
        # Avoid RAM explosion
        n_lim_workers = 10
        n_proc = min(n_lim_workers, max_workers or cpu_count())
        with Pool(processes=n_proc) as pool:
            pool.map(_wrapper_solve_hfpi_case, args)

    def solve_static(self, floors_height: np.ndarray, H: float, recurrence_period: float = 50, use_kd: bool = False):
        analysis_results = {}
        for direction in self.directional_forces:
            forces = self.directional_forces[direction]
            U_h = self.wind_analytics.get_U_H(
                height=H, direction=direction, recurrence_period=recurrence_period, use_kd=use_kd,
            )
            dim_data = static.DimensionalData(
                U_H=U_h, height=self.dimensions.height, base=self.dimensions.base
            )
            res = static.solve_static_forces(
                forces=forces, dim_data=dim_data, floors_heights=floors_height
            )
            analysis_results[direction] = ResultType(static_res=res, dynamic_res=None)

        return DirectionalAnalysisResults(results=analysis_results)


class ResultType(BaseModel):
    static_res: static.StaticResults
    dynamic_res: dynamic.HFPIResults | None

    def get_stats_global_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.static_res.get_stats_global_forces_static(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.static_res.get_stats_global_moments_static(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_forces_effective(self, stats_type: Literal["min", "max", "mean"]):
        forces_eq = self.dynamic_res.get_stats_forces_static_eq(stats_type)
        forces_static = self.static_res.get_stats_forces_static(stats_type)
        return common.get_stats_among_dct([forces_eq, forces_static], stats_type)

    def get_stats_moments_effective(self, stats_type: Literal["min", "max", "mean"]):
        mom_eq = self.dynamic_res.get_stats_moments_static_eq(stats_type)
        mom_static = self.static_res.get_stats_moments_static(stats_type)
        return common.get_stats_among_dct([mom_eq, mom_static], stats_type)

    def get_stats_global_forces_effective(self, stats_type: Literal["min", "max", "mean"]):
        forces_eq = self.dynamic_res.get_stats_global_forces_static_eq(stats_type)
        forces_static = self.static_res.get_stats_global_forces_static(stats_type)
        return common.get_stats_among_dct([forces_eq, forces_static], stats_type)

    def get_stats_global_moments_effective(self, stats_type: Literal["min", "max", "mean"]):
        mom_eq = self.dynamic_res.get_stats_global_moments_static_eq(stats_type)
        mom_static = self.static_res.get_stats_global_moments_static(stats_type)
        return common.get_stats_among_dct([mom_eq, mom_static], stats_type)

    def save(self, filename: pathlib.Path):
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: pathlib.Path):
        with open(filename, "rb") as f:
            return pickle.load(f)


class DirectionalAnalysisResults(BaseModel):
    results: dict[float, ResultType] = Field(default_factory=dict)

    def join_by_direction(self):
        return {d: DirectionalAnalysisResults(results={d: res}) for d, res in self.results.items()}

    def get_stats_global_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.static_res.get_stats_global_forces_static(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.static_res.get_stats_global_moments_static(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_forces_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.dynamic_res.get_stats_global_forces_static_eq(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_global_moments_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.dynamic_res.get_stats_global_moments_static_eq(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_effective_global_forces(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.dynamic_res.get_stats_global_forces_effective(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_stats_effective_global_moments(self, stats_type: Literal["min", "max", "mean"]):
        dcts = [
            v.dynamic_res.get_stats_global_moments_effective(stats_type)
            for k, v in self.results.items()
        ]
        return common.get_global_stats_dct_float(dcts, stats_type)

    def get_global_peaks_by_direction(
        self,
        variable_types: list[Literal["static", "hfpi", "effective"]] = [
            "static",
            "hfpi",
            "effective",
        ],
    ) -> dict[str, pd.DataFrame]:
        """Get global peaks per direction of results

        Returns results as [load_type] = DataFrame["direction", stats_type]

        load_type = "forces_static", "moments_static", "forces_static_eq", "moments_static_eq"
        stats_type = "min_x", "min_y", "min_z", "max_x", "max_y", "max_z"
        """
        res = self.join_by_direction()

        axis = ["x", "y", "z"]
        vars_use: list[str] = []
        if "static" in variable_types:
            vars_use.extend(["forces_static", "moments_static"])
        if "hfpi" in variable_types:
            vars_use.extend(["forces_static_eq", "moments_static_eq"])
        if "effective" in variable_types:
            vars_use.extend(["forces_effective", "moments_effective"])

        # Dict as [load_type][(stats_type, direction)] = value
        joined_res: dict[str, dict[tuple[str, float], float]] = {k: {} for k in vars_use}

        for d, r in res.items():
            calls = {
                "forces_static": r.get_stats_global_forces_static,
                "moments_static": r.get_stats_global_moments_static,
                "forces_static_eq": r.get_stats_global_forces_static_eq,
                "moments_static_eq": r.get_stats_global_moments_static_eq,
                "forces_effective": r.get_stats_effective_global_forces,
                "moments_effective": r.get_stats_effective_global_moments,
            }
            for name in vars_use:
                c = calls[name]
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

    def get_max_acceleration(self, pos: tuple[float, float] = (0, 0), floor: int = -1):
        return max(
            res.dynamic_res.get_max_acceleration(pos, floor) for res in self.results.values()
        )

    def get_max_acceleration_by_recurrence_period(
        self, pos: tuple[float, float] = (0, 0), floor: int = -1
    ):
        res = self.join_by_recurrence_period()
        return {k: r.get_max_acceleration(pos, floor) for k, r in res.items()}


class HFPIAnalysisResults(DirectionalAnalysisResults):
    results_folder: pathlib.Path

    results: dict[HFPICaseParameters, ResultType] = Field(default_factory=dict)  # type: ignore

    def load_result(self, parameters: HFPICaseParameters):
        filename = parameters.get_results_filename(self.results_folder)
        self.results[parameters] = ResultType.load(filename)

    def __getitem__(self, k: int) -> ResultType:
        return list(self.results.values())[k]

    @classmethod
    def load_all_results(cls, parameters: list[HFPICaseParameters], results_folder: pathlib.Path):
        results = cls(results_folder=results_folder)
        for p in parameters:
            results.load_result(p)
        return results

    def join_by(self, callback: Callable[[HFPICaseParameters], T]) -> dict[T, HFPIAnalysisResults]:
        joined_values = {}
        for p in self.results:
            key = callback(p)
            if key not in joined_values:
                joined_values[key] = []
            joined_values[key].append((p, self.results[p]))
        return {
            k: HFPIAnalysisResults(
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
