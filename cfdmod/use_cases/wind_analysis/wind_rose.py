import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def weibull_exc(x, c, k):
    return np.exp(-((x / c) ** k))


def weibull_cdf(x, c, k):
    return 1 - np.exp(-((x / c) ** k))


def weibull_pdf(x, c, k):
    return (k / c) * (x / c) ** (k - 1) * np.exp(-((x / c) ** k))

def calc_number_of_wind_ocurrences(df_raw: pd.DataFrame, vel_division: list[float]) -> pd.DataFrame:
    """Calculate ocurrences of wind events in absolute terms.

    It returns the number of 

    Args:
        df (pd.DataFrame): wind data to consider for analysis. Must have keys: 
            ["Vavg", "direction"]
        vel_division (list[float]): list with velocities intervals to consider for the divisions.

    Returns:
        pd.DataFrame: DataFrame with number of velocity ocurrences for each direction in dataframe
    """

    ...

def combine_ocurrences_per_direction(df_ocurr: pd.DataFrame, weights: dict[float, float]) -> pd.DataFrame:
    ...

def calc_weights_for_direction(all_directions: list[float], direction_interest: tuple[float, float]) -> dict[float, float]:
    ...

def calc_wind_ocurrences_for_new_directions(df_ocurr: pd.DataFrame, directions_intervals: list[float]) -> pd.DataFrame:
    ...

def get_weibull_parameters(df_combined: pd.DataFrame) -> pd.DataFrame:
    ...

def plot_wind_rose(df_weibull: pd.DataFrame):
    ...
