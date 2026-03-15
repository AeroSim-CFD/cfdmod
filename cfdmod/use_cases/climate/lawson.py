import pandas as pd
import numpy as np

def fit_average_velocity(df_weibull: pd.DataFrame, simulation_velocities: dict[float, np.ndarray]) -> np.ndarray:
    """Process Lawson criterion for simulations and Weibull coefficients

    Args:
        df_weibull (pd.DataFrame): Weibull results, with all required wind angles (same as 
            `simulation_velocities` keys)
        simulation_velocities (dict[float, np.ndarray]): Array of wind velocities to analyze in
            set of directions, given by key values.

    Returns: 
        np.ndarray: Average velocity considering Weibull parameters and simualtion results
    """
    ...