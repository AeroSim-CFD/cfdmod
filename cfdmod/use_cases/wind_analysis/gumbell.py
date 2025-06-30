import pandas as pd

def fit_gumbell_velocity(df_raw: pd.DataFrame, wind_angles: list[float] | None = None, years_excedence: float = 50) -> float | dict[float, float]:
    """Fit Gumbell max values using raw data and the specifications for it

    Args:
        df_raw (pd.DataFrame): Dataframe with gust speeds to consider
        wind_angles (list[float] | None, optional): Wind angles to consider for Gumbell with 
            directionality or None for no directionality. Defaults to None.
        years_excedence (float, optional): Years of exceedence for Gumbell analysis. Defaults to 50.

    Returns:
        float | dict[float, float]: Maximun value for given excedence, by direction or global
    """

    ...
