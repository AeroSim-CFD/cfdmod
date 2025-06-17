from __future__ import annotations

import pathlib

from pydantic import BaseModel, ConfigDict
from typing import Literal
import pandas as pd
import numpy as np
from scipy import integrate
import pickle


def _validate_keys_df(df: pd.DataFrame, keys: list[str]):
    if any(k not in df.columns for k in keys):
        return False
    return True


def read_hfpi_modes(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI modes from CSV. Expected columns:

    mode, period

    It adds a column frequency=1/period
    """

    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["mode", "period"]
    if not _validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI modes CSV {csv_path.as_posix()}. Found only keys {df.columns}"
        )
    df = df[req_keys]
    df["frequency"] = 1 / df["period"]
    df["wp"] = 2 * np.pi * df["frequency"]
    df.sort_values(by="mode", inplace=True)
    return df


def read_hfpi_floors_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI floors data from CSV. Expected columns:

    Z, XR, YR, M, I, R: height, center of rotation, mass, moment of inertia and rotation arm
    """

    df = pd.read_csv(csv_path, index_col=None)
    # "XG", "YG", "I"
    req_keys = ["Z", "XR", "YR", "M", "I", "R"]
    if not _validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI floors CSV {csv_path.as_posix()}. Found only keys {df.columns}"
        )
    df = df[req_keys]
    df.sort_values(by="Z", inplace=True)
    return df


def read_hfpi_floor_phi(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI floor phi from CSV. Expected columns:

    DX, DY, RZ: displacement in X and Y direction and rotation in Z.
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
    dx, dy, rz = [df_phi[k].to_numpy() for k in ("DX", "DY", "RZ")]
    m, r = nodes["M"].to_numpy(), nodes["R"].to_numpy()
    M_pav = m * (dx ** 2 + dy ** 2 + (r * rz) ** 2)
    M_gen = sum(M_pav)
    df_phi[["DX", "DY", "RZ"]] /= (M_gen**0.5)


class HFPIStructuralData(BaseModel):
    """Structural data required to run HFPI"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # information about modes: how many and their natural frequency
    df_modes: pd.DataFrame
    # information about floors: height, center of rotation, mass, moment of inertia and radius
    df_floors: pd.DataFrame
    # list of modal shapes. Each shape has components of displacement X and Y and rotation Z.
    df_modal_shapes: list[pd.DataFrame]

    max_active_modes: int
    is_normalized: bool = False

    @property
    def n_modes(self):
        return min(self.max_active_modes, len(self.df_modes))

    @property
    def n_floors(self):
        return len(self.df_floors)

    @classmethod
    def build(
        cls,
        modes_csv: pathlib.Path,
        floors_csv: pathlib.Path,
        phi_floors_csvs: list[pathlib.Path],
        max_active_modes: int = 1000,
    ):
        df_modes = read_hfpi_modes(modes_csv)
        df_floors = read_hfpi_floors_data(floors_csv)
        df_phi_floors = []
        for p in phi_floors_csvs:
            df = read_hfpi_floor_phi(p)
            df_phi_floors.append(df)

        struct_data = HFPIStructuralData(
            df_modes=df_modes,
            df_floors=df_floors,
            df_modal_shapes=df_phi_floors,
            max_active_modes=max_active_modes,
        )
        struct_data.validate_dfs()
        return struct_data

    def validate_dfs(self):
        for i, df in enumerate(self.df_modal_shapes):
            if len(df) != len(self.df_floors):
                raise ValueError(
                    f"Floor phi data (index {i}) has different number of floors than the floors csv. "
                    "Make sure that all phi and floor CSVs match the number of floors"
                )
        if len(self.df_modes) < len(self.df_modal_shapes):
            raise ValueError(
                "Less modes than phi data for it provided. Check if the floors used are correct."
            )

    def normalize_all_mode_shapes(self):
        """Normalize mode shapes for HFPI. Alters the `df_phi_floors`"""
        if self.is_normalized:
            return

        [normalize_mode_shapes(self.df_floors, df_phi) for df_phi in self.df_modal_shapes]
        self.is_normalized = True


class HFPIDimensionalData(BaseModel):
    """Analytical data required to analyze a given HFPI model"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    U_H: float
    height: float
    base: float
    xi: float = 0.02  # Critical damping ratio

    @property
    def dynamic_pressure(self):
        return 0.613 * (self.U_H**2)

    @property
    def CST(self):
        return min(self.base, self.height) / self.U_H

    @property
    def time_normalization_factor(self):
        return self.CST

    @property
    def force_normalization_factor(self):
        return self.base * self.height * self.dynamic_pressure

    @property
    def moments_normalization_factor(self):
        return self.base * self.base * self.height * self.dynamic_pressure


def read_hfpi_forces(hdf_path: pathlib.Path, scalar_key: str) -> pd.DataFrame:
    """Read forces for HFPI from path, with scalar key specified"""
    df_force = pd.read_hdf(hdf_path)
    req_keys = ["time_normalized"]
    if not _validate_keys_df(df_force, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI Forces HDF {hdf_path.as_posix()}. Found only keys {df_force.columns}"
        )
    return df_force


def fill_hfpi_forces(forces_df: pd.DataFrame, n_floors: int):
    """Fill missing floors with zeros"""
    floors = [int(k) for k in forces_df if not isinstance(k, str) or k.isnumeric()]
    for i in range(n_floors):
        if i in floors:
            continue
        forces_df[i] = 0


def scale_hfpi_forces(
    df_force: pd.DataFrame, key_name: str, *, force_factor: float, time_factor: float
):
    """Normalize HFPI forces by given factors"""

    df_force["time"] = df_force["time_normalized"] * time_factor
    df_force.drop("time_normalized", axis=1, inplace=True)
    col_mul = [k for k in df_force.columns if not isinstance(k, str) or not k.startswith("time")]
    df_force[col_mul] *= force_factor
    df_force.sort_values(by=["time"], inplace=True)


class HFPIForcesData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cf_x: pd.DataFrame
    cf_y: pd.DataFrame
    cm_z: pd.DataFrame
    is_scaled: bool = False

    def get_as_dct(self):
        def df2np(df: pd.DataFrame):
            return df.drop(columns=["time"]).to_numpy()

        return {"x": df2np(self.cf_x), "y": df2np(self.cf_y), "z": df2np(self.cm_z)}

    def fill_missing_floors(self, n_floors: int):
        for df in [self.cf_x, self.cf_y, self.cm_z]:
            fill_hfpi_forces(df, n_floors)

    @property
    def n_samples(self):
        return len(self.cf_x)

    @property
    def delta_t(self):
        k = "time_normalized"
        if "time" in self.cf_x.columns:
            k = "time"
        return self.cf_x[k][1] - self.cf_x[k][0]

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

    def get_scaled_forces(self, dim_data: HFPIDimensionalData) -> HFPIForcesData:
        """Generate HFPI scaled forces data"""
        time_factor = dim_data.time_normalization_factor

        if self.is_scaled:
            raise ValueError("Forces were already scaled, unable to scale again")

        cf_x = self.cf_x.copy()
        cf_y = self.cf_y.copy()
        cm_z = self.cm_z.copy()

        scale_hfpi_forces(
            cf_x,
            "FX",
            force_factor=dim_data.force_normalization_factor,
            time_factor=time_factor,
        )
        scale_hfpi_forces(
            cf_y,
            "FY",
            force_factor=dim_data.force_normalization_factor,
            time_factor=time_factor,
        )
        scale_hfpi_forces(
            cm_z,
            "MZ",
            force_factor=dim_data.moments_normalization_factor,
            time_factor=time_factor,
        )

        return HFPIForcesData(
            cf_x=cf_x,
            cf_y=cf_y,
            cm_z=cm_z,
            is_scaled=True,
        )


def get_moments_from_force(force: dict[str, np.ndarray], floor_heights: np.ndarray):
    moments = {}
    # Force in X causes -Y moment (right hand rule)
    moments["x"] = -force["y"] * floor_heights
    # Force in Y causes +Y moment (right hand rule)
    moments["y"] = force["x"] * floor_heights
    # Z is already a moment
    moments["z"] = force["z"].copy()
    return moments


def _get_stats_dct(dct: dict[str, np.ndarray], stats_type: Literal["min", "max", "mean"]) -> dict[str, np.ndarray] | dict[str, float]:
    if(stats_type == "max"):
        return {k: v.max(axis=0) for k, v in dct.items()}
    elif(stats_type == "min"):
        return {k: v.min(axis=0) for k, v in dct.items()}
    elif(stats_type == "mean"):
        return {k: v.mean(axis=0) for k, v in dct.items()}
    raise ValueError(f"Invalid stats type: {stats_type!r}, supports only 'min', 'max', 'mean'")


def _get_global_dct(dct: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    d = {k: v.sum(axis=1) for k, v in dct.items()}
    return d


class StaticResults(BaseModel):
    """Results generated from static analysis"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    floors_heights: np.ndarray

    # data as ["x", "y", "z"] = time series
    forces_static: dict[str, np.ndarray]
    moments_static: dict[str, np.ndarray]

    @property
    def global_forces_static(self):
        return _get_global_dct(self.forces_static)

    @property
    def global_moments_static(self):
        return _get_global_dct(self.moments_static)

    def get_stats_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.forces_static, stats_type)

    def get_stats_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.moments_static, stats_type)

    def get_stats_global_forces_static(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.global_forces_static, stats_type)

    def get_stats_global_moments_static(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.global_moments_static, stats_type)

def validate_forces_w_n_floors(forces: HFPIForcesData, n_floors: int):
    for name, df in [("Cfx",forces.cf_x), ("Cfy", forces.cf_y), ("Cmz",forces.cm_z)]:
        cols = df.columns
        for k in range(n_floors):
            if(int(k) not in cols and str(k) not in cols):
                raise KeyError(f"Coefficient {name} doesn't have all floors available. Nº of floors: {n_floors}; Columns found: {cols}")


def solve_static_forces(
    forces: HFPIForcesData, dim_data: HFPIDimensionalData, floors_heights: np.ndarray
):
    """Solve system for static forces"""
    forces.fill_missing_floors(len(floors_heights))
    validate_forces_w_n_floors(forces, len(floors_heights))
    normalized_forces = forces.get_scaled_forces(dim_data)

    force_static = normalized_forces.get_as_dct()
    moments_static = get_moments_from_force(force_static, floors_heights)
    return StaticResults(
        floors_heights=floors_heights, forces_static=force_static, moments_static=moments_static
    )


def compute_generalized_forces(
    forces: HFPIForcesData, structural_data: HFPIStructuralData
) -> pd.DataFrame:
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
    use_string = "0" in forces.cf_x.columns
    for n_mode in range(n_modes):
        df_phi = structural_data.df_modal_shapes[n_mode]
        f_tmp = np.zeros((n_floors, n_samples))
        for n_floor in range(n_floors):
            k_use = str(n_floor) if use_string else int(n_floor) 
            f_tmp[n_floor] = (
                cf_x[k_use] * df_phi["DX"].iloc[n_floor]
                + cf_y[k_use] * df_phi["DY"].iloc[n_floor]
                + cm_z[k_use] * df_phi["RZ"].iloc[n_floor]
            )
        # F_gen[n_mode] = f_tmp.sum(axis=1)
        F_gen[n_mode] = f_tmp.sum(axis=0)
    df_forces_gen = pd.DataFrame(F_gen)
    return df_forces_gen


def solve_euler_backwards(gen_force: np.ndarray, dt: float, wp: float, xi: float) -> np.ndarray:
    """Solve generalized displacement for a mode with Euler Backwards method

    gen_force: history series of generaliized force for one particular mode.
    dt: timestep
    wp: mode frequency (radians/sec)
    xi: dissipation (1%-2%)

    Returns displacement for time series for given mode
    """

    gf = gen_force
    n_samples = len(gen_force)
    gp = np.full((n_samples + 2), gf.mean() / (wp**2))
    # gp = np.full((n_samples + 2), 0)

    a = 2 - 2 * xi * wp * dt - (wp**2) * (dt**2)
    b = -1 + 2 * xi * wp * dt
    c = dt**2

    # displacement
    for t in range(2, n_samples + 2):
        gp[t] = a * gp[t - 1] + b * gp[t - 2] + c * gf[t - 2]
    return gp[2:]


def solve_runge_kunta(gen_force: np.ndarray, dt: float, wp: float, xi: float) -> np.ndarray:
    """Solve generalized displacement for a mode with Euler Backwards method

    gen_force: history series of generaliized force for one particular mode.
    dt: timestep
    wp: mode frequency (radians/sec)
    xi: dissipation (1%-2%)

    Returns displacement for time series for given mode
    """
    t_eval = np.arange(0, len(gen_force) * dt, dt)

    def system(t, y):
        i = min(int(t / dt), len(gen_force) - 1)  # Clamp to valid index
        F_t = gen_force[i]
        x, v = y
        dxdt = v
        dvdt = -2 * xi * wp * v - x + F_t
        return [dxdt, dvdt]

    x0 = gen_force.mean() / (wp**2)
    df = (gen_force[1:] - gen_force[:-1]).mean() / dt
    v0 = df / (2 * xi * wp) if xi * wp != 0 else 0.0

    sol = integrate.solve_ivp(
        system, (t_eval[0], t_eval[-1]), [x0, v0], t_eval=t_eval, method="RK45"
    )
    return sol.y[0]


def compute_mode_real_displacement(
    gen_mode_displacement: np.ndarray, df_mode_phi: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Solve real structure displacement for a given mode

    Args:
        gen_mode_displacement (np.ndarray): generalized displacement of building in given mode
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
        df_floor = df_mode_phi.iloc[n_floor]
        disp["x"][:, n_floor] = gen_mode_displacement * df_floor["DX"]
        disp["y"][:, n_floor] = gen_mode_displacement * df_floor["DY"]
        disp["z"][:, n_floor] = gen_mode_displacement * df_floor["RZ"]

    return disp


def compute_mode_static_equivalent_force(
    gen_mode_displacement: np.ndarray,
    df_mode_phi: pd.DataFrame,
    df_floors: pd.DataFrame,
    wp: float,
) -> dict[str, np.ndarray]:
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
    n_samples = len(gen_mode_displacement)

    static_eq = {}
    static_eq["x"] = np.zeros((n_samples, n_floors))
    static_eq["y"] = np.zeros((n_samples, n_floors))
    static_eq["z"] = np.zeros((n_samples, n_floors))

    for n_floor in range(n_floors):
        M = df_floors["M"][n_floor]
        R = df_floors["R"][n_floor]
        df_floor = df_mode_phi.iloc[n_floor]
        force_factor = (wp**2) * M
        moment_factor = (wp**2) * M * (R**2)
        static_eq["x"][:, n_floor] = gen_mode_displacement * force_factor * df_floor["DX"]
        static_eq["y"][:, n_floor] = gen_mode_displacement * force_factor * df_floor["DY"]
        static_eq["z"][:, n_floor] = gen_mode_displacement * moment_factor * df_floor["RZ"]

    return static_eq


def combine_modes(all_modes_dct: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Combine separate modes to get the total values"""

    sample = all_modes_dct[0]
    summed = {key: np.zeros_like(sample[key]) for key in ["x", "y", "z"]}

    for dct in all_modes_dct:
        for key in ["x", "y", "z"]:
            summed[key] += dct[key]

    return summed


class HFPIResults(BaseModel):
    """Results generated from HFPI analysis"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delta_t: float
    xi: float
    n_modes: int
    floors_heights: np.ndarray

    # data as ["x", "y", "z"] = time series
    displacement: dict[str, np.ndarray]
    forces_static_eq: dict[str, np.ndarray]
    moments_static_eq: dict[str, np.ndarray]

    static_results: StaticResults

    def save(self, filename: pathlib.Path):
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: pathlib.Path):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def global_forces_static_eq(self):
        return _get_global_dct(self.forces_static_eq)

    @property
    def global_moments_static_eq(self):
        return _get_global_dct(self.moments_static_eq)

    def get_acceleration(self) -> np.ndarray:
        acceleration = {}
        dt = self.delta_t
        for axis in ["x", "y"]:
            disp = self.displacement[axis]
            # Suppose last floor has the greatest acceleration
            disp = disp[:, -1]
            n = len(disp)
            acc = np.zeros(n)
            # Central difference for internal points
            acc[1:-1] = (disp[2:] - 2 * disp[1:-1] + disp[:-2]) / dt**2
            # Forward/backward difference for boundaries
            acc[0] = (disp[2] - 2 * disp[1] + disp[0]) / dt**2
            acc[-1] = (disp[-1] - 2 * disp[-2] + disp[-3]) / dt**2
            acceleration[axis] = acc

        scalar_acceleration = (acceleration["x"] ** 2 + acceleration["y"] ** 2) ** 0.5

        return scalar_acceleration

    def get_max_acceleration(self) -> float:
        return self.get_acceleration().max()

    def get_stats_forces_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.forces_static_eq, stats_type)

    def get_stats_moments_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.moments_static_eq, stats_type)

    def get_stats_global_forces_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.global_forces_static_eq, stats_type)

    def get_stats_global_moments_static_eq(self, stats_type: Literal["min", "max", "mean"]):
        return _get_stats_dct(self.global_moments_static_eq, stats_type)


def solve_hfpi(
    *,
    structural_data: HFPIStructuralData,
    dim_data: HFPIDimensionalData,
    forces: HFPIForcesData,
):
    """Solver HFPI (high frequency pressure integration) for given structure and forces conditions"""

    df_floors = structural_data.df_floors
    floors_heights = df_floors["Z"].to_numpy()

    structural_data.normalize_all_mode_shapes()
    validate_forces_w_n_floors(forces, structural_data.n_floors)
    normalized_forces = forces.get_scaled_forces(dim_data)
    normalized_forces.fill_missing_floors(structural_data.n_floors)

    generalized_forces = compute_generalized_forces(normalized_forces, structural_data)
    n_modes = structural_data.n_modes
    dt = normalized_forces.delta_t
    xi = dim_data.xi

    all_real_displacements = []
    all_static_eq_force = []

    for n_mode in range(n_modes):
        df_mode = structural_data.df_modes.iloc[n_mode]
        df_phi = structural_data.df_modal_shapes[n_mode]
        wp = df_mode["wp"]
        gen_displacement = solve_runge_kunta(generalized_forces[n_mode].to_numpy(), dt, wp, xi)
        real_displacement = compute_mode_real_displacement(gen_displacement, df_phi)
        all_real_displacements.append(real_displacement)

        static_eq_mode = compute_mode_static_equivalent_force(
            gen_displacement, df_phi, df_floors, wp
        )
        all_static_eq_force.append(static_eq_mode)

    displacement = combine_modes(all_real_displacements)

    force_static_eq = combine_modes(all_static_eq_force)
    moments_static_eq = get_moments_from_force(force_static_eq, floors_heights)

    return HFPIResults(
        delta_t=dt,
        xi=xi,
        n_modes=n_modes,
        floors_heights=floors_heights,
        displacement=displacement,
        forces_static_eq=force_static_eq,
        moments_static_eq=moments_static_eq,
        static_results=solve_static_forces(forces, dim_data, floors_heights),
    )
