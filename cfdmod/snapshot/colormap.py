import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba

# class ColormapFactory:
#     """Factory class for building colormaps.
#     Matplotlib colormap objects are supported in PyVista and many other libraries.
#     """

#     def __init__(self, scalar_range: tuple[float, float], n_divs: int):
#         self.scalar_range = scalar_range
#         self.n_divs = n_divs

#     @classmethod
#     def hex_to_rgb(cls, hex_code: str) -> tuple[float, float, float]:
#         return tuple(int(hex_code.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))

#     def _build_colormap(self, hex_color_list: list[str]) -> LinearSegmentedColormap:
#         N = len(hex_color_list)
#         scalar_range = np.array(self.scalar_range)

#         if np.all(scalar_range <= 0):
#             hex_colors = hex_color_list[: int(N / 2) - 1]
#         elif np.all(scalar_range >= 0):
#             hex_colors = hex_color_list[int(N / 2) :]
#         elif np.abs(scalar_range.min()) > np.abs(scalar_range.max()):
#             delta = (scalar_range.max() - scalar_range.min()) / N
#             n_min = round(-scalar_range.min() / delta - 0.5)
#             n_max = N - 1 - n_min
#             hex_colors = hex_color_list[: int(N / 2) + n_max]
#         elif np.abs(scalar_range.min()) < np.abs(scalar_range.max()):
#             delta = (scalar_range.max() - scalar_range.min()) / N
#             n_max = round(scalar_range.max() / delta - 0.5)
#             n_min = N - 1 - n_max
#             hex_colors = hex_color_list[int(N / 2) + 1 - n_min :]
#         else:
#             hex_colors = hex_color_list

#         rgb_colors = [self.hex_to_rgb(color) for color in hex_colors]

#         custom_cmap = LinearSegmentedColormap.from_list(
#             "custom_colormap", rgb_colors, N=self.n_divs
#         )

#         return custom_cmap

#     def build_default_colormap(self) -> LinearSegmentedColormap:
#         full_colors = [
#             "#062544",
#             "#03619F",
#             "#00A3D6",
#             "#01A781",
#             "#2ABC29",
#             "#CDECAF",
#             "#FFF585",
#             "#FFF22A",
#             "#FEA500",
#             "#F60100",
#             "#A70012",
#             "#510007",
#         ]
#         return self._build_colormap(full_colors)


#######################


class CustomColormapFactory:
    """Factory class for building custom colormaps.
    Matplotlib colormap objects are supported in PyVista and many other libraries.
    """

    def __init__(self, value_edges: list[float], colors: list[str]):
        self.value_edges = value_edges
        self.colors = colors
        self.scalar_range = (value_edges[0], value_edges[-1])
        self.n_divs = 1024  # lots of divisions to allow the ilusion of non-uniform color distribution by repeating the same color for a range of values

    def get_lookuptable(self) -> pv.LookupTable:
        map_mask = np.linspace(self.scalar_range[0], self.scalar_range[1], self.n_divs)
        c_map = np.ones((self.n_divs, 4))
        for v_low, v_high, color in zip(self.value_edges[:-1], self.value_edges[1:], self.colors):
            c_map[(v_low <= map_mask) & (map_mask < v_high)] = to_rgba(color)
        c_map[(map_mask == v_high)] = to_rgba(color)
        lut = pv.LookupTable()
        lut.n_values = self.n_divs
        lut.scalar_range = self.scalar_range
        lut.annotations = {v: str(v) for v in self.value_edges}
        lut.SetNanColor(1.0, 1.0, 1.0, 1.0)
        lut.apply_cmap(ListedColormap(c_map))
        return lut

    def get_scalar_bar_args_modifier(self) -> dict:
        return {"n_labels": 0}
