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

    @property
    def n_modes(self):
        return len(self.df_modes)

    @property
    def n_floors(self):
        return len(self.df_floors)

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

        self.df_phi_floors = [
            normalize_mode_shapes(self.df_floors, df_phi) for df_phi in self.df_phi_floors
        ]


class HFPICaseData(BaseModel):
    """Analytical data required to analyze a given HFPI model"""
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


def read_hfpi_forces(hdf_path: pathlib.Path, scalar_key: str) -> pd.DataFrame:
    """Read forces for HFPI from path, with scalar key specified"""
    df_force = pd.read_hdf(hdf_path)
    req_keys = ["time_normalized", scalar_key]
    if not _validate_keys_df(df_force, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI Forces HDF {hdf_path.as_posix()}. Found only keys {df_force.columns}"
        )
    df_force = df_force[["time_normalized", scalar_key]]
    return df_force


def normalize_hfpi_forces(
    df_force: pd.DataFrame, key_name: str, *, force_factor: float, time_factor: float
):
    """Normalize HFPI forces by given factors"""

    df_force["time"] *= time_factor
    df_force[key_name] *= force_factor
    df_force.drop("time_normalized", axis=1, inplace=True)
    df_force.sort_values(by=["time"], inplace=True)


class HFPIForcesData(BaseModel):
    cf_x: pd.DataFrame
    cf_y: pd.DataFrame
    cm_z: pd.DataFrame

    @property
    def n_samples(self):
        return len(self.cf_x)

    @classmethod
    def build(cls, cf_x_h5: pathlib.Path, cf_y_h5: pathlib.Path, cm_z_h5: pathlib.Path):
        cf_x = read_hfpi_forces(cf_x_h5, "FX")
        cf_y = read_hfpi_forces(cf_y_h5, "FY")
        cm_z = read_hfpi_forces(cm_z_h5, "MZ")
        if len(cf_x) != len(cf_y) or len(cf_x) != len(cm_z):
            raise ValueError(
                f"Length of forces data don't match. Paths {cf_x_h5, cf_y_h5, cm_z_h5}"
            )

        return HFPIForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
        )

    def get_normalized_forces(self, case_data: HFPICaseData) -> HFPIForcesData:
        """Generate HFPI normalized forces data"""
        time_factor = case_data.time_normalization_factor

        cf_x = normalize_hfpi_forces(
            self.cf_x.copy(),
            "FX",
            force_factor=case_data.force_normalization_factor,
            time_factor=time_factor,
        )
        cf_y = normalize_hfpi_forces(
            self.cf_y.copy(),
            "FY",
            force_factor=case_data.force_normalization_factor,
            time_factor=time_factor,
        )
        cm_z = normalize_hfpi_forces(
            self.cm_z.copy(),
            "MZ",
            force_factor=case_data.moments_normalization_factor,
            time_factor=time_factor,
        )
        return HFPIForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
        )


def compute_generalized_forces(forces: HFPIForcesData, structural_data: HFPIStructuralData) -> pd.DataFrame:
    """Compute generalized forces for a structure

    Generalized forces are the acting forces in the building projected to a given mode

    Args:
        forces (HFPIForcesData): Forces to use as reference
        structural_data (HFPIStructuralData): Structural data for building

    Returns:
        pd.DataFrame: Dataframe with generalized forces by mode and by floor
    """

    F_gen: dict[int, np.ndarray] = {}
    n_floors = structural_data.n_floors
    n_modes = structural_data.n_modes
    n_samples = forces.n_samples

    cf_x, cf_y, cm_z = forces.cf_x, forces.cf_y, forces.cm_z
    for n_mode in range(n_modes):
        df_phi = structural_data.df_phi_floors[n_mode]
        f_tmp = np.zeros((n_samples, n_floors))
        for n_floor in range(n_floors):
            if n_floor not in cf_x:
                f_tmp[:, n_floor] = 0
                continue
            f_tmp[n_floor] = (
                cf_x[n_floor] * df_phi["DX"][n_floor]
                + cf_y[n_floor] * df_phi["DY"][n_floor]
                + cm_z[n_floor] * df_phi["RZ"][n_floor]
            )
        F_gen[n_mode] = f_tmp.sum(axis=1)
    df_forces_gen = pd.DataFrame(F_gen)
    return df_forces_gen


def solve_mode_general_displacement(
    gen_force: np.ndarray, dt: float, wp: float, xi: float
) -> np.ndarray:
    """Solve general displacement for a mode with backward euler method

    Returns displacement for time series for given mode
    """

    gf = gen_force
    n_samples = len(gen_force)
    gp = np.full((n_samples + 2), gf.mean() / (wp**2))
    gp = np.full((n_samples + 2), 0)

    a = 2 - 2 * xi * wp * dt - (wp**2) * (dt**2)
    b = -1 + 2 * xi * wp * dt
    c = dt**2

    # displacement
    for t in range(2, n_samples + 2):
        gp[t] = a * gp[t - 1] + b * gp[t - 2] + c * gf[t - 2]
    return gp[2:]


def solve_mode_real_displacement(
    gen_mode_displacement: np.ndarray, df_mode_phi: pd.DataFrame
) -> pd.DataFrame:
    """Solve real structure displacement for a given mode

    Args:
        gen_mode_displacement (np.ndarray): general displacement of building in given mode
        df_mode_phi (pd.DataFrame): Mode data for structure floors

    Returns:
        pd.DataFrame: Dataframe with real structure displacement in "x", "y" and "z"
    """

    n_floors = len(df_mode_phi)
    n_samples = len(gen_mode_displacement)

    disp = {}
    disp["x"] = np.zeros((n_samples, n_floors))
    disp["y"] = np.zeros((n_samples, n_floors))
    disp["z"] = np.zeros((n_samples, n_floors))

    for n_floor in range(n_floors):
        disp["x"][:, n_floor] = gen_mode_displacement * df_mode_phi["DX"][n_floor]
        disp["y"][:, n_floor] = gen_mode_displacement * df_mode_phi["DY"][n_floor]
        disp["z"][:, n_floor] = gen_mode_displacement * df_mode_phi["RZ"][n_floor]

    return pd.DataFrame(disp)


def solve_mode_static_equivalent_force(
    real_mode_displacement: np.ndarray,
    df_mode_phi: pd.DataFrame,
    df_floors: pd.DataFrame,
    wp: float,
) -> pd.DataFrame:
    """Solve static equivalent force for a given mode

    Args:
        real_mode_displacement (np.ndarray): real displacement of building in given mode
        df_mode_phi (pd.DataFrame): Mode data for structure floors
        df_floors (pd.DataFrame): Data for structure floors
        wp (float): frequency for mode

    Returns:
        pd.DataFrame: Dataframe with static equivalent forces in "x", "y" and "z"
    """

    n_floors = len(df_mode_phi)
    n_samples = len(real_mode_displacement)

    static_eq = {}
    static_eq["x"] = np.zeros((n_samples, n_floors))
    static_eq["y"] = np.zeros((n_samples, n_floors))
    static_eq["z"] = np.zeros((n_samples, n_floors))

    for n_floor in range(n_floors):
        M = df_floors["M"][n_floor]
        R = df_floors["R"][n_floor]
        force_factor = (wp**2) * M
        moment_factor = (wp**2) * M * (R**2)
        static_eq["x"][:, n_floor] = (
            real_mode_displacement * force_factor * df_mode_phi["DX"][n_floor]
        )
        static_eq["y"][:, n_floor] = (
            real_mode_displacement * force_factor * df_mode_phi["DY"][n_floor]
        )
        static_eq["z"][:, n_floor] = (
            real_mode_displacement * moment_factor * df_mode_phi["RZ"][n_floor]
        )

    return pd.DataFrame(static_eq)


def combine_modes(all_modes_df: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine separate modes to get the total values"""
    sample = all_modes_df[0]
    summed = {key: np.zeros_like(sample[key]) for key in ['x', 'y', 'z']}
    
    for df in all_modes_df:
        for key in ['x', 'y', 'z']:
            summed[key] += df[key]

    return pd.DataFrame(summed)

class HFPISolver(BaseModel):
    """Solver for full process of HFPI"""

    structural_data: HFPIStructuralData
    case_data: HFPICaseData
    forces: HFPIForcesData

    def get_general_displacement(self): ...
