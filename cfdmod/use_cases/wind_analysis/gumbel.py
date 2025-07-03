import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import gumbel_r
import scipy

def directional_gumbel_fit(data:pd.DataFrame, wind_direction_cuts: np.ndarray, events_per_year: int=4) -> dict[tuple[float,float], tuple[float,float,list[float]]]:
    """Fit Gumbel for multiple wind directions
    """
    results = {}
    for i in range(len(wind_direction_cuts)):
        d_0, d_1 = wind_direction_cuts[i], wind_direction_cuts[(i+1)%len(wind_direction_cuts)]
        if(i < len(wind_direction_cuts)-1):
            dir_selection = (data['wind_direction']>=d_0) & (data['wind_direction']<d_1)
        else:
            dir_selection = (data['wind_direction']>=d_0) | (data['wind_direction']<d_1)
        
        if(dir_selection.sum() == 0):
            continue
        results[(int(d_0),int(d_1))] = fit_gumbel(data[dir_selection], events_per_year=events_per_year)
    return results

def fit_gumbel_BR_MIS(data: pd.DataFrame, events_per_year: int=4, reduced_variate_cut_point:float=-1) -> tuple[float, float, list[float]]:
    """Fit Gumbel max values using raw data and the specifications for it
    Fit method by Vallis(2019):
        Method of independent storms to select peaks
        Discards smaller peaks based on reduced variate threshold
        Fits by linear regression with classical Gumbel method (wrong, but conservative)
    """
    reduced_variates, selected_peaks = get_storm_peaks(data, events_per_year, reduced_variate_cut_point)
    reduced_variates = get_reduced_variate(selected_peaks, reescale_multiple=events_per_year)
    a, U, _, *_ = scipy.stats.linregress(reduced_variates, selected_peaks)
    # return U+np.log(events_per_year)*a, a, selected_peaks
    return U+np.log(events_per_year)*a, a, selected_peaks

def fit_gumbel_MLE_MIS(data: pd.DataFrame, events_per_year: int=4, reduced_variate_cut_point:float=-1) -> tuple[float, float, list[float]]:
    """Fit Gumbel max values using raw data and the specifications for it
    Fit method by maximum likelihood estimation:
        Method of independent storms to select peaks
        Discards smaller peaks based on reduced variate threshold
        Fits by maximum likelihood estimation (statistically sound, less conservative)
    """
    _, selected_peaks = get_storm_peaks(data, events_per_year, reduced_variate_cut_point)
    U, a = gumbel_r.fit(selected_peaks)
    return U+np.log(events_per_year)*a, a, selected_peaks

def fit_gumbel(data: pd.DataFrame, events_per_year: int=4) -> tuple[float, float, list[float]]:
    """Fit Gumbel max values using raw data and the specifications for it
    Default implementation by Vallis(2019) - BR-MIS

    Args:
        data (pd.DataFrame): Dataframe with gust speeds to consider
        wind_angles (list[float] | None, optional): Wind angles to consider for Gumbel with 
            directionality or None for no directionality. Defaults to None.
        years_excedence (float, optional): Years of exceedence for Gumbel analysis. Defaults to 50.

    Returns:
        float | dict[float, float]: Maximun value for given excedence, by direction or global
    """
    return fit_gumbel_BR_MIS(data=data, events_per_year=events_per_year, reduced_variate_cut_point=-1)

def get_storm_peaks(data: pd.DataFrame, events_per_year: int, reduced_variate_cut_point:float) -> list[float]:
    data = data.copy() #destructive procedure. Separating from original
    peak_values = []
    num_years = len(pd.to_datetime(data['datetime']).dt.year.unique())
    num_of_peaks = num_years*events_per_year
    
    for _ in range(num_of_peaks):
        peak_value = data['u_gust'].max()
        data = remove_storm_from_series(data, peak_value)
        peak_values.append(peak_value)
    peak_values = sorted(peak_values)
    reduced_variates = get_reduced_variate(peaks=peak_values, reescale_multiple=events_per_year)
    id_first_valid = np.searchsorted(reduced_variates, reduced_variate_cut_point, side='right')
    return reduced_variates[id_first_valid:], peak_values[id_first_valid:]
    
def get_reduced_variate(peaks: list[float], reescale_multiple: int) -> np.ndarray:
    sorted_peaks = sorted(peaks)
    n = len(sorted_peaks)
    excedent_probability_estimator = (np.arange(1,n+1)/(n+1))**reescale_multiple
    return -np.log(-np.log(excedent_probability_estimator))
        
def remove_storm_from_series(data: pd.DataFrame, peak_value: float, correlation_hours: float = 4*24):
    peak_row = data[data['u_gust'] == peak_value]
    peak_date = pd.to_datetime(peak_row['datetime'].iloc[0])
    start_event = peak_date - pd.Timedelta(hours=correlation_hours)
    end_event = peak_date + pd.Timedelta(hours=correlation_hours)
    mask_event = pd.to_datetime(data['datetime']).between(start_event, end_event)
    return data[~ mask_event]

def type_I_return_level(T, U, a):
    return U - a * np.log(-np.log(1 - 1/T))
    
def plot_gumbel_regression(list_of_maxima: list, U: float, a: float, events_per_year: int=4):
    n = len(list_of_maxima)
    i = np.arange(1, n + 1)
    Fi = ((i) / (n+1))**events_per_year
    Yi = -np.log(-np.log(Fi))

    plt.scatter(Yi, list_of_maxima, label='Empirical data')
    plt.plot(Yi, Yi*a + U, 'r-', label='Gumbel fit')

    # Plot aesthetics
    plt.xlabel(r'Reduced Variate $\left(\frac{V-U}{a} \right)$')
    plt.ylabel('Peak gust speeds [m/s]')
    plt.grid(True)
    plt.legend()
    plt.show()
    


def plot_gumbel_pdf(list_of_maxima: list, U: float, a: float, events_per_year: int=4):
    x = np.linspace(min(list_of_maxima) - 2, max(list_of_maxima) + 2, 100)
    gumbel_pdf = gumbel_r.pdf(x, loc=U-np.log(events_per_year)*a, scale=a)

    plt.hist(list_of_maxima,bins='auto', density=True, alpha=0.5, label="Empirical data")
    plt.plot(x, gumbel_pdf, 'r-', label='Gumbel fit')

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # Plot aesthetics
    plt.xlabel('Velocidade de rajada de 3s [m/s]')
    plt.ylabel('Probability density')
    plt.grid(True)
    plt.legend()
    plt.show()
    
