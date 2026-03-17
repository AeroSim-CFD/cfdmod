import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import weibull_min
from scipy.special import gamma
from scipy.optimize import brentq
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
        # shape, scale = weibull_fit_moments(data[dir_selection]['u_mean'])
        results[(round(d_0, 2),round(d_1, 2))] = ((incidence_probability, shape, scale), dir_selection.sum())
    return results

def weibull_shape_from_mean_and_std(mean, std):
    def f(k):
        return (
            gamma(1 + 2.0/k) /
            gamma(1 + 1.0/k)**2
            - 1.0
            - (std/mean)**2
        )
    return brentq(f, 0.2, 20)

def weibull_scale_from_mean_and_shape(mean, shape):
    return mean / gamma(1.0 + 1.0 / shape)

def weibull_fit_moments(data):
    data = np.asarray(data)
    data = data[data > 0]  # recommended for wind data

    mean = data.mean()
    std = data.std(ddof=0)

    shape = weibull_shape_from_mean_and_std(mean,std)
    scale = weibull_scale_from_mean_and_shape(mean, shape)

    return shape, scale

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

    fig, ax = plt.subplots()
    ax.hist(data,bins=bins, density=True, alpha=0.5, label="Empirical data")
    ax.plot(x, weibull_pdf, 'r-', label='weibull fit')

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # Plot aesthetics
    ax.set_xlabel('Mean velocity [m/s]')
    ax.set_ylabel('Probability density')
    plt.grid(True)
    ax.legend()
    return fig, ax
    
