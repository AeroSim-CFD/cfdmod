import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import weibull_min
import scipy


def directional_weibull_fit(data:pd.DataFrame, wind_direction_cuts: list[tuple[float,float]]) -> dict[tuple[float,float], tuple[float,float,list[float]]]:
    """Fit weibull for multiple wind directions
    """
    results = {}
    valid_selection = (np.isfinite(data['u_mean'])) & (data['u_mean']>0)
    for d_0, d_1 in wind_direction_cuts:
        if(d_0<d_1):
            dir_selection = (data['wind_direction']>=d_0) & (data['wind_direction']<d_1) & valid_selection
        else:
            dir_selection = (data['wind_direction']>=d_0) | (data['wind_direction']<d_1) & valid_selection
        
        if(dir_selection.sum() == 0):
            continue
        incidence_probability = dir_selection.sum() / valid_selection.sum()
        shape, scale = fit_weibull(data[dir_selection])
        results[(round(d_0, 2),round(d_1, 2))] = (incidence_probability, shape, scale)
    return results

def fit_weibull(data: pd.DataFrame) -> tuple[float, float, list[float]]:
    """Fit weibull max values using raw data and the specifications for it
    Default implementation by Vallis(2019) - BR-MIS

    Args:
        data (pd.DataFrame): Dataframe with gust speeds to consider
        wind_angles (list[float] | None, optional): Wind angles to consider for weibull with 
            directionality or None for no directionality. Defaults to None.
        years_excedence (float, optional): Years of exceedence for weibull analysis. Defaults to 50.

    Returns:
        float | dict[float, float]: Maximun value for given excedence, by direction or global
    """
    mask_valid_values = (np.isfinite(data['u_mean'])) & (data['u_mean']>0)
    shape, loc, scale = weibull_min.fit(data[mask_valid_values]['u_mean'].to_numpy(), floc=0)  

    return shape, scale


def get_weibull_quantile(shape, scale, percentile) -> float:
    return weibull_min.ppf(percentile, c=shape, scale=scale)

def get_weibull_probability_between_velocities(shape, scale, v_low, v_high) -> float:
    return weibull_min.cdf(v_high, c=shape, scale=scale) - weibull_min.cdf(v_low, c=shape, scale=scale)
    
def plot_weibull_pdf(data: list, shape: float, scale: float, bins: int|str='auto'):
    x = np.linspace(0, max(data), 100)
    weibull_pdf = weibull_min.pdf(x, shape, scale=scale, loc=0)

    plt.hist(data,bins=bins, density=True, alpha=0.5, label="Empirical data")
    plt.plot(x, weibull_pdf, 'r-', label='weibull fit')

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # Plot aesthetics
    plt.xlabel('Mean velocity [m/s]')
    plt.ylabel('Probability density')
    plt.grid(True)
    plt.legend()
    plt.show()
    
