from __future__ import annotations

import pathlib

from pydantic import BaseModel
import pandas as pd
import numpy as np


def _validate_keys_df(df: pd.DataFrame, keys: list[str]):
    if any(k not in df.columns for k in keys):
        return False
    return True


def read_hfpi_modes(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI modes from CSV. Expected columns:

    mode, period, wp

    It adds a column frequency=1/period
    """

    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["mode", "period", "wp"]
    if not _validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI modes CSV {csv_path.as_posix()}. Found only keys {df.columns}"
        )
    df = df[req_keys]
    df["frequency"] = 1 / df["period"]
    return df


def read_hfpi_floors_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI floors data from CSV. Expected columns:

    Z, XR, YR, XG, YG, M, I, R
    """

    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["Z", "XR", "YR", "XG", "YG", "M", "I", "R"]
    if not _validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI floors CSV {csv_path.as_posix()}. Found only keys {df.columns}"
        )
    df = df[req_keys]
    return df


def read_hfpi_floor_phi(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI floor phi from CSV. Expected columns:

    DX, DY, RZ
    """

    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["DX", "DY", "RZ"]
    if not _validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI floor phi CSV {csv_path.as_posix()}. Found only keys {df.columns}"
        )
    df = df[req_keys]
    return df


def normalize_mode_shapes(df_floors: pd.DataFrame, df_phi: pd.DataFrame):
    """Normalize mode shapes for HFPI"""

    nodes = df_floors
    M_pav = nodes["M"] * (df_phi["DX"] ** 2 + df_phi["DY"] ** 2 + (nodes["R"] * df_phi["RZ"]) ** 2)
    M_gen = sum(M_pav)
    return df_phi / (M_gen**0.5)


class HFPIStructuralData(BaseModel):
    """Structural data required to run HFPI"""

    df_modes: pd.DataFrame
    df_floors: pd.DataFrame
    df_phi_floors: list[pd.DataFrame]

    @classmethod
    def build(
        cls, modes_csv: pathlib.Path, floors_csv: pathlib.Path, phi_floors_csvs: list[pathlib.Path]
    ):
        df_modes = read_hfpi_modes(modes_csv)
        df_floors = read_hfpi_floors_data(floors_csv)
        df_phi_floors = []
        for p in phi_floors_csvs:
            df = read_hfpi_floor_phi(p)
            if len(df) != len(df_floors):
                raise ValueError(
                    f"Floor phi data at {p} has different number of floors than the floors csv at {floors_csv}. "
                    "Make sure that all phi and floor CSVs match the number of floors"
                )
            df_phi_floors.append(p)

        return HFPIStructuralData(
            df_modes=df_modes,
            df_floors=df_floors,
            df_phi_floors=df_phi_floors,
        )

    def normalize_all_mode_shapes(self):
        """Normalize mode shapes for HFPI. Alters the `df_phi_floors`"""

        self.df_phi_floors = [normalize_mode_shapes(self.df_floors, df_phi) for df_phi in self.df_phi_floors]


class HFPICaseData(BaseModel):
    U_H: float
    height: float
    base: float
    # Critical damping ratio
    xi: float = 0.02

    @property
    def q(self):
        return 0.613 * (self.U_H**2)

    @property
    def CST(self):
        return self.base / self.U_H

    @property
    def time_normalization_factor(self):
        magic_factor = 45.5871904815377 
        return magic_factor / self.U_H

    @property
    def force_normalization_factor(self):
        return self.base * self.height * self.q

    @property
    def moments_normalization_factor(self):
        return self.base * self.base * self.height * self.q


def read_hfpi_forces(hdf_path: pathlib.Path, key_name: str) -> pd.DataFrame:
    df_force = pd.read_hdf(hdf_path)
    req_keys = ["time_normalized", key_name]
    if not _validate_keys_df(df_force, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI Forces HDF {hdf_path.as_posix()}. Found only keys {df_force.columns}"
        )
    return df_force


def normalize_hfpi_forces(df_force: pd.DataFrame, key_name: str, *, force_factor: float, time_factor: float):
    df_force["time"] *= time_factor
    df_force[key_name] *= force_factor
    df_force.drop("time_normalized", axis=1, inplace=True)
    df_force.sort_values(by=["time"], inplace=True)
    return df_force


class HFPIForcesData(BaseModel):
    cf_x: pd.DataFrame
    cf_y: pd.DataFrame
    cm_z: pd.DataFrame

    @classmethod
    def build(cls, cf_x_h5: pathlib.Path, cf_y_h5: pathlib.Path, cm_z_h5: pathlib.Path):
        cf_x = read_hfpi_forces(cf_x_h5, "FX")
        cf_y = read_hfpi_forces(cf_y_h5, "FY")
        cm_z = read_hfpi_forces(cm_z_h5, "MZ")
        return HFPIForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
        )

    def get_normalized_forces(self, case_data: HFPICaseData) -> HFPIForcesData:
        time_factor = case_data.time_normalization_factor

        cf_x = normalize_hfpi_forces(self.cf_x.copy(), "FX", force_factor=case_data.force_normalization_factor, time_factor=time_factor)
        cf_y = normalize_hfpi_forces(self.cf_y.copy(), "FY", force_factor=case_data.force_normalization_factor, time_factor=time_factor)
        cm_z = normalize_hfpi_forces(self.cm_z.copy(), "MZ", force_factor=case_data.moments_normalization_factor, time_factor=time_factor)
        return HFPIForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
        )

    def get_generalized_forces(self, structural_data: HFPIStructuralData):
        modes = structural_data.df_modes["modes"]
        # I didn't understand hor to do this...

        ### Generalized forces
        # for case in cases_vel.keys():
        #     for PR in PRs:
        #         F_gen = {}
        #         T = Forces_exp[case][PR]["T"]
        #         for p in mode_numbers:
        #             df_temp = pd.DataFrame(np.zeros((T, n_floors)), columns=[f for f in range(n_floors)])
        #             for f in range(n_floors):
        #                 if f not in Forces_exp[case][PR]["FX"].columns:
        #                     df_temp[f] = 0
        #                 else:
        #                     df_temp[f] = (
        #                         Forces_exp[case][PR]["FX"][f] * phi_norm[p]["DX"][f]
        #                         + Forces_exp[case][PR]["FY"][f] * phi_norm[p]["DY"][f]
        #                         + Forces_exp[case][PR]["MZ"][f] * phi_norm[p]["RZ"][f]
        #                     )
        #             F_gen[p] = df_temp.sum(axis=1)
        #         Forces_exp[case][PR]["F_gen"] = F_gen


def second_order_backward_euler(generalized_force: np.ndarray, dt: float, wp: float, xi: float) -> np.ndarray:
    """Solve time series with backward euler method

    Returns displacement for time series
    """

    gf = generalized_force
    n_samples = len(generalized_force)
    gp = np.full((n_samples + 2), gf.mean() / (wp**2))
    gp = np.full((n_samples + 2), 0)

    a = 2 - 2 * xi * wp * dt - (wp**2) * (dt**2)
    b = -1 + 2 * xi * wp * dt
    c = dt**2

    # displacement
    for t in range(2, n_samples + 2):
        gp[t] = a * gp[t - 1] + b * gp[t - 2] + c * gf[t - 2]
    return gp[2:]


