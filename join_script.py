import pathlib

import pandas as pd

data_path = pathlib.Path(
    "../insight/Docker/local/volume/Prologis Cajamar 4 - 000/G300/cases/000-p_inf_1/data"
)

body_hist_series = pd.read_hdf(data_path / "bodies.G300_hs.data.h5")
negative_body_hist_series = pd.read_hdf(data_path / "bodies.G300_neg_hs.data.h5")

last_index = body_hist_series.point_idx.max()
negative_body_hist_series.point_idx += last_index + 1

new_df = pd.concat([body_hist_series, negative_body_hist_series]).sort_values(
    by=["time_step", "point_idx"]
)
new_df.to_hdf(data_path / "bodies.G300.merged.data.h5", key="df", mode="w")
