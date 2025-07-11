import matplotlib.pyplot as plt
import numpy as np

from cfdmod.use_cases.s1 import profile
from cfdmod.use_cases.plot_config import plot_style


def plot_numerical_and_analytical_vel_profile(
    *,
    z: np.ndarray,
    H: float,
    u_num: np.ndarray,
    Iu_num: np.ndarray,
    u_num_ref: float,
    cat_eu: profile.EUCat,
    cat_nbr: profile.NBRCat,
    u_ref: float = 1,
    Fr: float = 0.65,
):
    arr_u_eu = profile.get_EU_cat_u_profile(z=z, H=H, cat=cat_eu, u_ref=u_ref, Fr=Fr)
    arr_u_nbr = profile.get_NBR_cat_u_profile(z=z, H=H, cat=cat_nbr, u_ref=u_ref, Fr=Fr)
    arr_u_num = u_num / u_num_ref
    arr_Iu_num = Iu_num
    arr_Iu_eu = profile.get_EU_cat_Iu_profile(z=z, cat=cat_eu)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=True)

    ax[0].plot(arr_u_num, z / H, **plot_style["AeroSim"]["line"])
    ax[0].plot(arr_u_eu, z / H, **plot_style["EU"]["line"])
    ax[0].plot(arr_u_nbr, z / H, **plot_style["ABNT"]["line"])
    ax[0].set_xlabel(r"$ \overline{u}_x / u_H$")
    ax[0].set_ylabel("$z/H$")
    # ax[0].legend(["AeroSim", "Experimental"], loc="lower right")
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xlim(0, 1.4)
    ax[0].set_title("Velocidade longitudinal média", fontsize=16)
    ax[0].legend(loc="best", frameon=False, fontsize=10)
    # ax[0].grid()

    ax[1].plot(arr_Iu_num, z / H, **plot_style["AeroSim"]["line"])
    ax[1].plot(arr_Iu_eu, z / H, **plot_style["EU"]["line"])
    ax[1].set_xlabel(r"$ ( \overline{u\prime u\prime} )^{1/2} / \overline{u}_x$")
    ax[1].set_ylim(0, 2)
    ax[1].set_xlim(0, 0.5)
    ax[1].set_title("Intensidade turbulenta longitudinal", fontsize=16)
    # ax[1].grid()

    # fig.legend(loc="right")
    # plt.tight_layout(rect=[0, 0, 0.8, 1])
    # plt.show(fig)
    return fig, ax
