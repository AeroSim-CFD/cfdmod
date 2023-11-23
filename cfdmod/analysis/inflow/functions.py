import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter


def spectral_density(
    velocity_signal: np.ndarray,
    timestamps: np.ndarray,
    reference_velocity: float,
    characteristic_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a FFT over a velocity signal

    Args:
        velocity_signal (np.ndarray): Array of instantaneous velocity signal
        timestamps (np.ndarray): Array of timestamps of the signal
        reference_velocity (float): Value for reference velocity. For normalization
        characteristic_length (float): Value for Characteristic length. For normalization

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple with spectral density values array and normalized frequency values array
    """

    def filter_avg_data(data: np.ndarray) -> np.ndarray:
        filtered_data = gaussian_filter(data, sigma=3)  # Sigma smooths the curve
        return filtered_data

    delta_t = timestamps[1] - timestamps[0]
    signal_frequency = 1 / delta_t

    (xf, yf) = scipy.signal.periodogram(velocity_signal, signal_frequency, scaling="density")
    st = np.std(velocity_signal)
    yf = xf * yf / st**2
    xf = xf * characteristic_length / reference_velocity  # Stroulhall number N = f * L / U

    # Get the filter coefficients so we can check its frequency response.
    yf = filter_avg_data(yf)
    return xf[2:], yf[2:]
