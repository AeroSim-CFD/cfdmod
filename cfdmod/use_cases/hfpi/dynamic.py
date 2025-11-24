from __future__ import annotations

import pathlib
import pickle
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
import scipy.signal
import scipy.stats
from scipy import integrate

from cfdmod.use_cases.hfpi import common, static
from cfdmod import utils

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

def read_hfpi_floors_data(
    csv_path: pathlib.Path
) -> pd.DataFrame:
    """Read HFPI floors data from CSV. Expected columns:

    Z, M, I: height, center of rotation, mass, moment of inertia
    XR, YR required to update inertial moments from mass center
    """

    df = pd.read_csv(csv_path, index_col=None)
    # "XG", "YG", "XR", "YR", "I", "R"
    req_keys = ["Z", "M", "I","XR", "YR"]
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
        df_floors = read_hfpi_floors_data( floors_csv )
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
            CM_pos = np.array((structural_data.df_floors.iloc[int(k_use)][['XR','YR']]))
            cm_z_onCM = cm_z[k_use] - common.series_cross_product(CM_pos, cf_x[k_use], cf_y[k_use])
            f_tmp[n_floor, :] = (
                cf_x[k_use] * df_phi["DX"].iloc[n_floor]
                + cf_y[k_use] * df_phi["DY"].iloc[n_floor]
                + cm_z_onCM * df_phi["RZ"].iloc[n_floor]
            )
        # F_gen[n_mode] = f_tmp.sum(axis=1)
        F_gen[n_mode] = f_tmp.sum(axis=0)
    # print(F_gen)
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
    floors_heights: np.ndarray

    # data as ["x", "y", "z"] = time series
    displacement: dict[str, np.ndarray]
    forces_static_eq: dict[str, np.ndarray]
    moments_static_eq: dict[str, np.ndarray]

    def rotate_xy(self, angle_rot: float):
        common.rotate_values_xy(self.forces_static_eq, angle_rot)
        common.rotate_values_xy(self.moments_static_eq, angle_rot)

    @property
    def global_forces_static_eq(self):
        return common.get_global_dct(self.forces_static_eq)

    @property
    def global_moments_static_eq(self):
        return common.get_global_dct(self.moments_static_eq)

    def get_displacement_w_rotation(self, pos: tuple[float, float]) -> dict[str, np.ndarray]:
        r = (pos[0] ** 2 + pos[1] ** 2) ** 0.5
        theta = np.arctan2(pos[1], pos[0])
        disp_z = theta + self.displacement["z"]

        x = np.cos(disp_z) * r
        y = np.sin(disp_z) * r

        return {"x": x, "y": y}

    def get_acceleration(self, pos: tuple[float, float] = (0, 0)) -> np.ndarray:
        """Get acceleration considering position for radius (moment in Z)"""
        acceleration = {}
        dt = self.delta_t

        disp_full = {"x": self.displacement["x"].copy(), "y": self.displacement["y"].copy()}
        disp_rot = self.get_displacement_w_rotation(pos)
        disp_full["x"] += disp_rot["x"]
        disp_full["y"] += disp_rot["y"]

        for axis in ["x", "y"]:
            disp = disp_full[axis][:]
            acc = np.zeros_like(disp, dtype=np.float32)
            # Central difference for internal points
            acc[1:-1] = (disp[2:] - 2 * disp[1:-1] + disp[:-2]) / dt**2
            # Forward/backward difference for boundaries
            acc[0] = (disp[2] - 2 * disp[1] + disp[0]) / dt**2
            acc[-1] = (disp[-1] - 2 * disp[-2] + disp[-3]) / dt**2
            acceleration[axis] = acc

        scalar_acceleration = (acceleration["x"] ** 2 + acceleration["y"] ** 2) ** 0.5

        return scalar_acceleration

    def get_floor_acceleration(
        self, pos: tuple[float, float] = (0, 0), floor: int = -1
    ) -> np.ndarray:
        """Get acceleration from given floor, considering radius for Z"""
        return self.get_acceleration(pos=pos)[:, floor]

    def get_max_acceleration_per_floor(self, pos: tuple[float, float] = (0, 0), peak_method: Literal["extreme", "peak-factor"]="extreme", peak_factor: float=4) -> np.ndarray:
        if peak_method == "extreme":
            return self.get_acceleration(pos=pos).max(axis=0)
        else:
            acc = self.get_acceleration(pos=pos)
            acc_m = acc.mean(axis=0)
            return acc_m + (acc-acc_m).std(axis=0)*peak_factor

    def get_floor_max_acceleration(
        self, pos: tuple[float, float] = (0, 0), floor: int = -1, peak_method: Literal["extreme", "peak-factor"]="extreme", peak_factor: float=4
    ) -> float:
        if peak_method == "extreme":
            return self.get_floor_acceleration(pos=pos, floor=floor).max()
        else:
            acc = self.get_floor_acceleration(pos=pos, floor=floor)
            return acc.mean()+acc.std()*peak_factor

    def get_stats_forces_static_eq(self, stats_type: Literal["min", "max", "mean"], peak_method: Literal["extreme", "peak-factor"]="extreme", peak_factor: float=4):
        if peak_method=="extreme":
            return common.get_stats_dct(self.forces_static_eq, stats_type)
        else:
            return common.get_stats_dct_peak_factor(self.forces_static_eq, stats_type, peak_factor)

    def get_stats_moments_static_eq(self, stats_type: Literal["min", "max", "mean"], peak_method: Literal["extreme", "peak-factor"]="extreme", peak_factor: float=4):
        if peak_method=="extreme":
            return common.get_stats_dct(self.moments_static_eq, stats_type)
        else:
            return common.get_stats_dct_peak_factor(self.moments_static_eq, stats_type, peak_factor)


    def get_stats_global_forces_static_eq(self, stats_type: Literal["min", "max", "mean"], peak_method: Literal["extreme", "peak-factor"]="extreme", peak_factor: float=4):
        if peak_method=="extreme":
            return common.get_stats_dct(self.global_forces_static_eq, stats_type)
        else:
            return common.get_stats_dct_peak_factor(self.global_forces_static_eq, stats_type, peak_factor)

    def get_stats_global_moments_static_eq(self, stats_type: Literal["min", "max", "mean"], peak_method: Literal["extreme", "peak-factor"]="extreme", peak_factor: float=4):
        if peak_method=="extreme":
            return common.get_stats_dct(self.global_moments_static_eq, stats_type)
        else:
            return common.get_stats_dct_peak_factor(self.global_moments_static_eq, stats_type, peak_factor)
    
    # def get_stats_global_forces_static_eq_peak_factor(self, stats_type: Literal["min", "max"], peak_factor: float = 4):
    #     dict_vals = self.global_moments_static_eq
    #     result = {}
    #     for comp, values in dict_vals.items():
    #         mean = values.mean(axis=0)
    #         std = values.std(axis=0)
    #         signal = -1 if stats_type=='min' else 1
    #         result[comp] = mean + signal*std*peak_factor
    #     return result

def move_displacement_ref_from_CM_to_origin(
    displacement: dict[str, np.ndarray], 
    structural_data:  HFPIStructuralData,
) -> dict[str, np.ndarray]:
    """ Transforms displacements from the coordinate system of center of mass to original coordinate system.
        Currently not implemented. Just returns displacement without changes.
    """
    # n_floors = structural_data.n_floors
    # x, y, z = displacement['x'], displacement['y'], displacement['z']
    # for n_floor in range(n_floors):
    #     k_use = int(n_floor)
    #     print(n_floor, "old",y[-1,k_use])
    #     CM_pos = np.array((structural_data.df_floors.iloc[int(k_use)][['XR','YR']]))
    #     CM_R = (CM_pos[0]**2+CM_pos[1]**2)**0.5
    #     CM_ang = np.atan2(CM_pos[0], CM_pos[1])
    #     # new_ang = CM_ang + z[:,k_use]
    #     new_ang = z[:,k_use]
    #     displacement['x'][:,k_use] = CM_R*np.cos(new_ang) + x[:,k_use]
    #     displacement['y'][:,k_use] = y[:,k_use] + CM_R*np.sin(new_ang)
    #     print("new",displacement['y'][-1,k_use], z[-1,k_use], CM_ang)
    return displacement

def move_loads_ref_from_CM_to_origin(
    forces: dict[str, np.ndarray], 
    moments: dict[str, np.ndarray],
    structural_data:  HFPIStructuralData,
) -> tuple[dict[str, np.ndarray],dict[str, np.ndarray]]:
    """Transforms forces and moments of force from the coordinate system of center of mass to original coordinate system.

    Args:
        forces dict[str, np.ndarray]: dictionary with force. Keys are ['x','y''z'] and items are numpy arrays N timesteps by F floors.
        moments dict[str, np.ndarray]: dictionary with moments of force. Keys are ['x','y''z'] and items are numpy arrays N timesteps by F floors.
        structural_data (HFPIStructuralData): object with structural data

    Returns:
        tuple[dict[str, np.ndarray],dict[str, np.ndarray]]: Transformed dictionaries of force and moment of force in that order
    """
    n_floors = structural_data.n_floors
    fx, fy, mz = forces['x'], forces['y'], moments['z']
    for n_floor in range(n_floors):
        k_use = int(n_floor)
        CM_pos = np.array((structural_data.df_floors.iloc[int(k_use)][['XR','YR']]))
        moments['z'][:,k_use] = mz[:,k_use] + common.series_cross_product(CM_pos, fx[:,k_use], fy[:,k_use])
    forces["z"] = moments['z'].copy()
    return forces, moments

def solve_hfpi(
    *,
    structural_data: HFPIStructuralData,
    dim_data: static.DimensionalData,
    forces: static.StaticForcesData,
    xi: float,
    frequency_multiplier: float,
    return_values_on_origin: bool = True,
    apply_wavelet_filter: bool = False,
    filter_percentage: float =  99.5,
) -> HFPIResults:
    """Solver HFPI (high frequency pressure integration) for given structure and forces conditions"""

    df_floors = structural_data.df_floors
    floors_heights = df_floors["Z"].to_numpy()

    structural_data.normalize_all_mode_shapes()
    static.validate_forces_w_n_floors(forces, structural_data.n_floors)
    normalized_forces = forces.get_scaled_forces(dim_data)
    normalized_forces.fill_missing_floors(structural_data.n_floors)

    generalized_forces = compute_generalized_forces(normalized_forces, structural_data)
    dt = normalized_forces.delta_t
    if apply_wavelet_filter:
        generalized_forces = filter_with_wavelet(generalized_forces, structural_data, dt, filter_percentage)

    all_real_displacements = []
    all_static_eq_force = []

    for n_mode in structural_data.active_modes:
        df_mode = structural_data.df_modes.iloc[n_mode]
        df_phi = structural_data.df_modal_shapes[n_mode]
        wp = df_mode["wp"]*frequency_multiplier
        gen_displacement = solve_runge_kunta(
            generalized_forces[n_mode].to_numpy(), dt=dt, wp=wp, xi=xi
        )
        real_displacement = compute_mode_real_displacement(gen_displacement, df_phi)
        all_real_displacements.append(real_displacement)

        static_eq_mode = compute_mode_static_equivalent_force(
            gen_displacement, df_phi, df_floors, wp
        )
        all_static_eq_force.append(static_eq_mode)

    displacement = combine_modes(all_real_displacements)

    force_static_eq = combine_modes(all_static_eq_force)
    moments_static_eq = common.get_moments_from_force(force_static_eq, floors_heights)
    if return_values_on_origin:
        displacement = move_displacement_ref_from_CM_to_origin(displacement, structural_data)
        force_static_eq, moments_static_eq = move_loads_ref_from_CM_to_origin(force_static_eq, moments_static_eq, structural_data)

    return HFPIResults(
        delta_t=dt,
        xi=xi,
        floors_heights=floors_heights,
        displacement=displacement,
        forces_static_eq=force_static_eq,
        moments_static_eq=moments_static_eq,
    )




def filter_with_wavelet(
    generalized_forces: pd.DataFrame,
    structural_data: HFPIStructuralData,
    dt: float, #timestep
    filter_percentage: float = 99.5,
):
    """Apply filter using wavelet transform. Identify outliers based on rayleight regression of coefficientes and limits values to the percentile chosen"""
    
    window_size_1s = 1/dt
    df_modes = structural_data.df_modes
    reference_freq = df_modes['frequency'].iloc[0] #define window based on first mode
    g_std = int(3*(1/reference_freq)*window_size_1s)
    gauss_window = scipy.signal.windows.gaussian(8*g_std, std=g_std, sym=True)
    SFT = scipy.signal.ShortTimeFFT(gauss_window, hop=2, fs=1/dt, scale_to='psd')
    f_window = 4
    
    f_ids_range = set()
    for f_target in df_modes["frequency"]:
        f_idx = np.argmin(np.abs(SFT.f - f_target))
        f_ids_range.update(range(f_idx-f_window, f_idx+f_window))
    f_ids_range = list(f_ids_range)
    
    for mode_id in generalized_forces.columns:
        Fxx = SFT.stft(generalized_forces[mode_id].to_numpy())
        for f_id in f_ids_range:
            Z = Fxx[f_id, :]
            Ampl_freq = np.abs(Z) 
            sigma_hat = np.sqrt((Ampl_freq**2).mean() / 2.0)
            ray = scipy.stats.rayleigh(loc=0.0, scale=sigma_hat)
            A_lim = ray.ppf(filter_percentage/100)
            mask_ampltoohigh = (Ampl_freq>A_lim)
            Fxx[f_id, mask_ampltoohigh] = Z[mask_ampltoohigh]/Ampl_freq[mask_ampltoohigh]*A_lim
        generalized_forces[mode_id] = SFT.istft(Fxx, k1=generalized_forces.shape[0])
    return generalized_forces