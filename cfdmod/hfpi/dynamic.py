from __future__ import annotations

import pathlib
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import integrate

from cfdmod import utils
from cfdmod.hfpi import common, static


def read_hfpi_modes(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI modes from CSV. Expected columns:

    mode, period

    It adds a column frequency=1/period
    """

    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["mode", "period"]
    if not utils.validate_keys_df(df, req_keys):
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

    Z, M, I: height, center of rotation, mass, moment of inertia
    XR, YR required to update inertial moments from mass center
    """

    df = pd.read_csv(csv_path, index_col=None)
    # "XG", "YG", "XR", "YR", "I", "R"
    req_keys = ["Z", "M", "I", "XR", "YR"]
    if not utils.validate_keys_df(df, req_keys):
        raise KeyError(
            f"Not all required keys ({req_keys}) present in HFPI floors CSV {csv_path.as_posix()}. Found only keys {df.columns}"
        )
    # Radius of gyration
    df["R"] = (df["I"] / df["M"]) ** 0.5

    df = df[req_keys + ["R"]]

    df.sort_values(by="Z", inplace=True)
    return df


def read_hfpi_floor_phi(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read HFPI floor phi from CSV. Expected columns:

    DX, DY, RZ: displacement in X and Y direction and rotation in Z.
    """

    df = pd.read_csv(csv_path, index_col=None)
    req_keys = ["DX", "DY", "RZ"]
    if not utils.validate_keys_df(df, req_keys):
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
    M_pav = m * (dx**2 + dy**2 + (r * rz) ** 2)
    M_gen = sum(M_pav)
    if M_gen > 0:
        df_phi[["DX", "DY", "RZ"]] /= M_gen**0.5
    else:
        df_phi[["DX", "DY", "RZ"]] = 0


class HFPIStructuralData(BaseModel):
    """Structural data required to run HFPI"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # information about modes: how many and their natural frequency
    df_modes: pd.DataFrame
    # information about floors: height, center of rotation, mass, moment of inertia and radius
    df_floors: pd.DataFrame
    # list of modal shapes. Each shape has components of displacement X and Y and rotation Z.
    df_modal_shapes: list[pd.DataFrame]

    active_modes: list
    is_normalized: bool = False

    @property
    def n_modes(self):
        return len(self.active_modes)

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
        inactive_modes: list = [],
    ):
        df_modes = read_hfpi_modes(modes_csv)
        df_floors = read_hfpi_floors_data(floors_csv)
        df_phi_floors = []
        for p in phi_floors_csvs:
            df = read_hfpi_floor_phi(p)
            df_phi_floors.append(df)

        modes = list(df_modes["mode"])
        inactive_modes = [m - 1 for m in inactive_modes]  # normalize from 1 index to 0 index
        if max_active_modes < df_modes.shape[0]:
            modes_to_deactivate = [m for m in range(max_active_modes - 1, df_modes.shape[0])]
            inactive_modes = inactive_modes + modes_to_deactivate
        modes = [m - 1 for m in modes]  # normalize from 1 index to 0 index
        active_modes = [m for m in modes if m not in inactive_modes]

        struct_data = HFPIStructuralData(
            df_modes=df_modes,
            df_floors=df_floors,
            df_modal_shapes=df_phi_floors,
            active_modes=active_modes,
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


def compute_generalized_forces(
    forces: static.StaticForcesData, structural_data: HFPIStructuralData
) -> pd.DataFrame:
    """Compute generalized forces for a structure

    Generalized forces are the acting forces in the building projected to a given mode

    Args:
        forces (static.StaticForcesData): Forces to use as reference
        structural_data (HFPIStructuralData): Structural data for building

    Returns:
        pd.DataFrame: Dataframe with generalized forces by mode and by floor
    """

    F_gen: dict[int, np.ndarray] = {}
    n_floors = structural_data.n_floors
    n_samples = forces.n_samples

    cf_x, cf_y, cm_z = forces.cf_x, forces.cf_y, forces.cm_z
    use_string = "0" in forces.cf_x.columns
    for n_mode in structural_data.active_modes:
        df_phi = structural_data.df_modal_shapes[n_mode]
        f_tmp = np.zeros((n_floors, n_samples))
        for n_floor in range(n_floors):
            k_use = str(n_floor) if use_string else int(n_floor)
            CM_pos = np.array((structural_data.df_floors.iloc[int(k_use)][["XR", "YR"]]))
            cm_z_onCM = cm_z[k_use] - common.series_cross_product(CM_pos, cf_x[k_use], cf_y[k_use])
            f_tmp[n_floor, :] = (
                cf_x[k_use] * df_phi["DX"].iloc[n_floor]
                + cf_y[k_use] * df_phi["DY"].iloc[n_floor]
                + cm_z_onCM * df_phi["RZ"].iloc[n_floor]
            )
        F_gen[n_mode] = f_tmp.sum(axis=0)
    df_forces_gen = pd.DataFrame(F_gen)
    return df_forces_gen


def solve_runge_kunta(gen_force: np.ndarray, dt: float, wp: float, xi: float) -> np.ndarray:
    """Solve generalized displacement for a mode with Euler Backwards method

    gen_force: history series of generaliized force for one particular mode.
    dt: timestep
    wp: mode frequency (radians/sec)
    xi: dissipation (1%-2%)

    Returns displacement for time series for given mode
    """
    end_step = (len(gen_force) - 1) * dt
    t_eval = np.linspace(0, end_step, len(gen_force))

    from scipy.interpolate import interp1d

    f_func = interp1d(t_eval, gen_force, kind="linear", fill_value="extrapolate")

    # System definition
    def system(t, y):
        F_t = f_func(t)
        x, v = y
        # x1 = pos
        # x2 = vel
        # x1' = x2
        # x2' = F(t)−2xi.w_p x1 − w_p^2 x1
        dxdt = v
        dvdt = F_t - 2 * xi * wp * v - wp**2 * x
        return [dxdt, dvdt]

    # Initial condition estimation
    x0 = gen_force.mean() / (wp**2)
    df = (gen_force[1:] - gen_force[:-1]).mean() / dt
    v0 = df / (2 * xi * wp) if xi * wp != 0 else 0.0

    # Solve the ODE
    sol = integrate.solve_ivp(
        system, (t_eval[0], t_eval[-1]), [x0, v0], t_eval=t_eval, method="RK45"
    )
    return sol.y[0]


def compute_real_displacement(
    gen_mode_displacements: dict[int, np.ndarray], modal_shapes: dict[int, pd.DataFrame]
) -> np.ndarray:
    """Solve real structure displacement, summing contribution of all active modes

    Args:
        gen_mode_displacement (dict[int, np.ndarray]): generalized displacement of building for each mode
        modal_shapes (dict[int,pd.DataFrame]): Shape of each mode

    Returns:
        pd.DataFrame: Dataframe with real structure displacement in "x", "y" and "z"
    """
    all_real_displacements = []
    for n_mode in gen_mode_displacements.keys():
        df_phi = modal_shapes[n_mode]
        real_displacement = compute_mode_real_displacement(gen_mode_displacements[n_mode], df_phi)
        all_real_displacements.append(real_displacement)
    displacement = combine_modes(all_real_displacements)
    return displacement


def compute_mode_real_displacement(
    gen_mode_displacement: np.ndarray, df_mode_shape: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Solve real structure displacement for a given mode

    Args:
        gen_mode_displacement (np.ndarray): generalized displacement of building in given mode
        df_mode_shape (pd.DataFrame): Shape of current mode

    Returns:
        pd.DataFrame: Dataframe with real structure displacement in "x", "y" and "z"
    """

    n_floors = len(df_mode_shape)
    n_samples = len(gen_mode_displacement)

    disp = {}
    disp["x"] = np.zeros((n_samples, n_floors))
    disp["y"] = np.zeros((n_samples, n_floors))
    disp["z"] = np.zeros((n_samples, n_floors))

    for n_floor in range(n_floors):
        df_floor = df_mode_shape.iloc[n_floor]
        disp["x"][:, n_floor] = gen_mode_displacement * df_floor["DX"]
        disp["y"][:, n_floor] = gen_mode_displacement * df_floor["DY"]
        disp["z"][:, n_floor] = gen_mode_displacement * df_floor["RZ"]

    return disp


def compute_static_equivalent_forces(
    gen_mode_displacements: dict[int, np.ndarray],
    floors_mass: np.ndarray,
    floors_radius: np.ndarray,
    modal_shapes: dict[int, pd.DataFrame],
    wps: dict[int, float],
) -> dict[str, np.ndarray]:
    """Solve sum of static equivalent forces for all active modes

    Args:
        gen_mode_displacement (dict[int, np.ndarray]): generalized displacement of building for each mode
        floors_mass (np.ndarray): mass of each floor
        floors_radius (np.ndarray): radius of gyration of each floor
        modal_shapes (dict[int,pd.DataFrame]): Shape of each mode
        wp (dict[int,float]): frequency for each mode

    Returns:
        pd.DataFrame: Dataframe with static equivalent forces in "x", "y" and "z"
    """
    all_static_eq_force = []

    for n_mode in gen_mode_displacements.keys():
        df_phi = modal_shapes[n_mode]
        wp = wps[n_mode]

        static_eq_mode = compute_mode_static_equivalent_force(
            gen_mode_displacements[n_mode], df_phi, floors_mass, floors_radius, wp
        )
        all_static_eq_force.append(static_eq_mode)
    force_static_eq = combine_modes(all_static_eq_force)
    return force_static_eq


def compute_mode_static_equivalent_force(
    gen_mode_displacement: np.ndarray,
    df_mode_shape: pd.DataFrame,
    floors_mass: np.ndarray,
    floors_radius: np.ndarray,
    wp: float,
) -> dict[str, np.ndarray]:
    """Solve static equivalent force for a given mode

    Args:
        gen_mode_displacement (np.ndarray): geneeralized displacement of building in given mode
        df_mode_phi (pd.DataFrame): Shape of current mode
        floors_mass (np.ndarray): mass of each floor
        floors_radius (np.ndarray): radius of gyration of each floor
        wp (float): frequency for current mode

    Returns:
        pd.DataFrame: Dataframe with static equivalent forces in "x", "y" and "z"
    """

    n_floors = len(df_mode_shape)
    n_samples = len(gen_mode_displacement)

    static_eq = {}
    static_eq["x"] = np.zeros((n_samples, n_floors))
    static_eq["y"] = np.zeros((n_samples, n_floors))
    static_eq["z"] = np.zeros((n_samples, n_floors))

    for n_floor in range(n_floors):
        M = floors_mass[n_floor]
        R = floors_radius[n_floor]
        df_floor_mode_shape = df_mode_shape.iloc[n_floor]
        force_factor = (wp**2) * M
        moment_factor = (wp**2) * M * (R**2)
        static_eq["x"][:, n_floor] = (
            gen_mode_displacement * force_factor * df_floor_mode_shape["DX"]
        )
        static_eq["y"][:, n_floor] = (
            gen_mode_displacement * force_factor * df_floor_mode_shape["DY"]
        )
        static_eq["z"][:, n_floor] = (
            gen_mode_displacement * moment_factor * df_floor_mode_shape["RZ"]
        )
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

    floors_heights: np.ndarray
    floors_mass: np.ndarray
    floors_radius: np.ndarray
    modal_shapes: dict[int, pd.DataFrame]
    wps: dict[int, float]

    gen_displacements: dict[int, np.ndarray]

    # data as ["x", "y", "z"] = time series
    @cached_property
    def displacement(self) -> dict[str, np.ndarray]:
        return compute_real_displacement(self.gen_displacements, self.modal_shapes)

    @cached_property
    def forces_static_eq(self) -> dict[str, np.ndarray]:
        return compute_static_equivalent_forces(
            self.gen_displacements,
            self.floors_mass,
            self.floors_radius,
            self.modal_shapes,
            self.wps,
        )

    @cached_property
    def moments_static_eq(self) -> dict[str, np.ndarray]:
        return common.get_moments_from_force(self.forces_static_eq, self.floors_heights)

    def rotate_xy(self, angle_rot: float):
        common.rotate_values_xy(self.forces_static_eq, angle_rot)
        common.rotate_values_xy(self.moments_static_eq, angle_rot)

    # @property
    # def global_forces_static_eq(self):
    #     return common.get_global_dct(self.forces_static_eq)

    # @property
    # def global_moments_static_eq(self):
    #     return common.get_global_dct(self.moments_static_eq)

    def get_point_acceleration(
        self, cm_positions: pd.DataFrame, pos: tuple[float, float]
    ) -> np.ndarray:
        """Get acceleration considering position for radius (moment in Z)"""
        cms = cm_positions[["XR", "YR"]].to_numpy()
        n_floors = len(self.floors_heights)

        acceleration_ls = {"x": [], "y": []}
        for floor in range(n_floors):
            floor_acc = self.get_point_floor_acceleration(cms[floor], pos, floor)
            acceleration_ls["x"].append(floor_acc["x"])
            acceleration_ls["y"].append(floor_acc["y"])
        accelerations_full = {
            ax: np.column_stack(acceleration_ls[ax]) for ax in acceleration_ls.keys()
        }
        return accelerations_full

    def get_point_floor_acceleration(
        self, cm_position: tuple[float, float], pos: tuple[float, float] = (0, 0), floor: int = -1
    ) -> np.ndarray:
        """Get acceleration from given floor, considering radius for Z"""
        disp_full = {
            "x": self.displacement["x"][:, floor].copy(),
            "y": self.displacement["y"][:, floor].copy(),
        }
        rel_pos = np.array(pos) - np.array(cm_position)
        point_angle = np.arctan2(rel_pos[1], rel_pos[0])
        displ_angle = point_angle + self.displacement["z"][:, floor]
        r = (rel_pos[0] ** 2 + rel_pos[1] ** 2) ** 0.5

        disp_full["x"] += np.cos(displ_angle) * r
        disp_full["y"] += np.sin(displ_angle) * r
        dt = self.delta_t

        return common.second_derivative(disp_full, dt)

    def get_max_acceleration_per_floor(
        self,
        cm_positions: pd.DataFrame,
        pos: tuple[float, float] = (0, 0),
        peak_method: Literal["gumbel", "max", "peak-factor"] = "gumbel",
        peak_factor: float = 4,
    ) -> np.ndarray:
        accs = self.get_point_acceleration(cm_positions, pos)
        if peak_method == "max":
            acc_mag = (accs["x"] ** 2 + accs["y"] ** 2) ** 0.5
            return acc_mag.max(axis=0)
        if peak_method == "gumbel":
            acc_mag = (accs["x"] ** 2 + accs["y"] ** 2) ** 0.5
            acc_extreme = []
            for floor in range(acc_mag.shape[1]):
                acc_extreme.append(
                    common.gumbel_extreme_value(
                        hist_series=acc_mag[:, floor],
                        dt=self.delta_t,
                        peak_duration=0.000001,
                        event_duration=10 * 60,
                        extreme_type="max",
                        n_subdivisions=10,
                        non_exceedance_probability=0.78,
                    )
                )
            return np.array(acc_extreme)
        else:
            acc_peak = {ax: peak_factor * acc.std(axis=0) for ax, acc in accs.items()}
            acc_mag = (acc_peak["x"] ** 2 + acc_peak["y"] ** 2) ** 0.5
            return acc_mag

    def get_floor_max_acceleration(
        self,
        cm_position: tuple[float, float],
        pos: tuple[float, float] = (0, 0),
        floor: int = -1,
        peak_method: Literal["gumbel", "max", "peak-factor"] = "gumbel",
        peak_factor: float = 4,
    ) -> float:
        accs = self.get_point_floor_acceleration(cm_position, pos, floor)
        if peak_method == "gumbel":
            acc_mag = (accs["x"] ** 2 + accs["y"] ** 2) ** 0.5
            acc_extreme = common.gumbel_extreme_value(
                hist_series=acc_mag,
                dt=self.delta_t,
                peak_duration=0.000001,
                event_duration=10 * 60,
                extreme_type="max",
                n_subdivisions=10,
                non_exceedance_probability=0.78,
            )
            return acc_extreme
        elif peak_method == "max":
            acc_mag = (accs["x"] ** 2 + accs["y"] ** 2) ** 0.5
            return acc_mag.max()
        elif peak_method == "peak-factor":
            acc_peak = {ax: peak_factor * acc.std(axis=0) for ax, acc in accs.items()}
            return (acc_peak["x"] ** 2 + acc_peak["y"] ** 2) ** 0.5

    def get_stats_forces_static_eq(
        self,
        cm_positions: pd.DataFrame,
        stats_type: Literal["min", "max", "mean"],
        peak_method: Literal["gumbel", "extreme", "peak-factor"] = "gumbel",
        peak_factor: float = 4,
    ):
        forces, _ = common.move_loads_ref_from_CM_to_origin(
            self.forces_static_eq,
            self.moments_static_eq,
            cm_positions,
        )

        if peak_method == "extreme":
            return common.get_stats_dct(forces, stats_type)
        elif peak_method == "gumbel":
            return common.get_stats_dct_gumbell(forces, stats_type, self.delta_t)
        else:
            return common.get_stats_dct_peak_factor(forces, stats_type, peak_factor)

    def get_stats_monents_static_eq(
        self,
        cm_positions: pd.DataFrame,
        stats_type: Literal["min", "max", "mean"],
        peak_method: Literal["gumbel", "extreme", "peak-factor"] = "gumbel",
        peak_factor: float = 4,
    ):
        _, moments = common.move_loads_ref_from_CM_to_origin(
            self.forces_static_eq,
            self.moments_static_eq,
            cm_positions,
        )

        if peak_method == "extreme":
            return common.get_stats_dct(moments, stats_type)
        elif peak_method == "gumbel":
            return common.get_stats_dct_gumbell(moments, stats_type, self.delta_t)
        else:
            return common.get_stats_dct_peak_factor(moments, stats_type, peak_factor)

    def get_stats_global_forces_static_eq(
        self,
        cm_positions: pd.DataFrame,
        stats_type: Literal["min", "max", "mean"],
        peak_method: Literal["gumbel", "extreme", "peak-factor"] = "gumbel",
        peak_factor: float = 4,
    ):
        forces, _ = common.move_loads_ref_from_CM_to_origin(
            self.forces_static_eq,
            self.moments_static_eq,
            cm_positions,
        )
        global_forces = common.get_global_dct(forces)
        if peak_method == "extreme":
            return common.get_stats_dct(global_forces, stats_type)
        elif peak_method == "gumbel":
            return common.get_stats_dct_gumbell(global_forces, stats_type, self.delta_t)
        else:
            return common.get_stats_dct_peak_factor(global_forces, stats_type, peak_factor)

    def get_stats_global_moments_static_eq(
        self,
        cm_positions: pd.DataFrame,
        stats_type: Literal["min", "max", "mean"],
        peak_method: Literal["gumbel", "extreme", "peak-factor"] = "gumbel",
        peak_factor: float = 4,
    ):
        _, moments = common.move_loads_ref_from_CM_to_origin(
            self.forces_static_eq,
            self.moments_static_eq,
            cm_positions,
        )
        global_moments = common.get_global_dct(moments)
        if peak_method == "extreme":
            return common.get_stats_dct(global_moments, stats_type)
        elif peak_method == "gumbel":
            return common.get_stats_dct_gumbell(global_moments, stats_type, self.delta_t)
        else:
            return common.get_stats_dct_peak_factor(global_moments, stats_type, peak_factor)


def solve_hfpi(
    *,
    structural_data: HFPIStructuralData,
    dim_data: static.DimensionalData,
    forces: static.StaticForcesData,
    xi: float,
) -> HFPIResults:
    """Solver HFPI (high frequency pressure integration) for given structure and forces conditions"""

    df_floors = structural_data.df_floors
    floors_heights = df_floors["Z"].to_numpy()
    floors_mass = df_floors["M"].to_numpy()
    floors_radius = df_floors["R"].to_numpy()

    structural_data.normalize_all_mode_shapes()
    static.validate_forces_w_n_floors(forces, structural_data.n_floors)
    normalized_forces = forces.get_scaled_forces(dim_data)
    normalized_forces.fill_missing_floors(structural_data.n_floors)

    generalized_forces = compute_generalized_forces(normalized_forces, structural_data)
    dt = normalized_forces.delta_t

    all_gen_displacements = {}
    wps = {}
    modal_shapes = {}

    for n_mode in structural_data.active_modes:
        df_mode = structural_data.df_modes.iloc[n_mode]
        wp = float(df_mode["wp"])
        all_gen_displacements[n_mode] = solve_runge_kunta(
            generalized_forces[n_mode].to_numpy(), dt=dt, wp=wp, xi=xi
        )
        wps[n_mode] = wp
        modal_shapes[n_mode] = structural_data.df_modal_shapes[n_mode].copy()

    return HFPIResults(
        delta_t=dt,
        xi=xi,
        floors_heights=floors_heights,
        floors_mass=floors_mass,
        floors_radius=floors_radius,
        modal_shapes=modal_shapes,
        wps=wps,
        gen_displacements=all_gen_displacements,
    )
