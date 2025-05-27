import pathlib

import pandas as pd


def h5_keys_to_string(filename: pathlib.Path):
    output_filename = filename.with_name(filename.stem + ".2string.h5")

    with pd.HDFStore(filename, mode="r") as store:
        with pd.HDFStore(output_filename, mode="w") as out_store:
            for key in store.keys():
                df = store.get(key)

                # Convert digit column names to strings
                df.columns = [
                    (
                        str(col)
                        if isinstance(col, int) or (isinstance(col, str) and col.isdigit())
                        else col
                    )
                    for col in df.columns
                ]

                out_store.put(key, df)


h5_keys_to_string(
    pathlib.Path("./fixtures/tests/pressure/data/new.bodies.galpao.data.resampled.h5")
)
h5_keys_to_string(
    pathlib.Path("./fixtures/tests/pressure/data/new.points.static_pressure.data.resampled.h5")
)
