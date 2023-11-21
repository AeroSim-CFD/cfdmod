****************
Spectral Density
****************

Spectral density analysis is a detailed examination of the **frequency distribution of turbulent fluctuations** in the incoming wind.
By decomposing the turbulent signal into its constituent frequencies, spectral density provides valuable information about the **energy distribution across different scales**.

Spectral characteristics of the inflow helps in validating CFD simulations to replicate the diverse range of turbulent eddies present in the atmospheric boundary layer, contributing to a more realistic depiction of turbulence structures and their impact on the flow.

The relationship between spectral density and LES mesh size can be expressed through a power-law scaling, recognizing that the **mesh size influences the resolved range of turbulent eddies**.
The correlation can be formulated as:

.. math::
    S(f) \propto \frac{1}{(\Delta x)^\alpha}

The scaling exponent depends on the specific characteristics of the turbulent flow being simulated and the numerical methods employed.
It reflects the rate at which **spectral density diminishes with decreasing mesh size**.
Smaller mesh sizes allow for the **resolution of higher-frequency turbulent structures**, impacting the spectral content of the simulation.

The correlation suggests that as the LES mesh size decreases, enabling finer resolution of turbulent eddies, the spectral density increases, capturing a broader range of frequencies.
Conversely, coarser meshes may result in a loss of information about high-frequency turbulent structures, resulting in **cut-off frequency range**, which scales will not be captured.