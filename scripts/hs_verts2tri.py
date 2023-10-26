import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from nassu.lnas import LagrangianFormat, LagrangianGeometry


def get_vertices_ratios(
    geometry: LagrangianGeometry,
) -> tuple[list[set[int]], list[list[tuple[int, float]]]]:
    # Set of triangles that each vertice participates to
    vertices_triangles_set: list[set[int]] = [set() for _ in geometry.vertices]
    # Ratio for each triangle to spread as (t_idx, ratio)
    vertices_triangles_ratio: list[list[tuple[int, float]]] = [[] for _ in geometry.vertices]
    for t_idx, t in enumerate(geometry.triangles):
        for v_idx in t:
            vertices_triangles_set[v_idx].add(t_idx)
    # Get the ratio of vertice area from each triangle
    for v_idx, t_idxs in enumerate(vertices_triangles_set):
        total_area = sum(geometry.areas[t_idx] for t_idx in t_idxs)
        for t_idx in t_idxs:
            t_area = geometry.areas[t_idx]
            ratio = t_area / total_area
            vertices_triangles_ratio[v_idx].append((t_idx, ratio))

    return vertices_triangles_set, vertices_triangles_ratio


def convert_folder(folder: pathlib.Path):
    filename = folder / "hist_series.triangles.hdf"
    filename_read = folder / "hist_series.pickle"
    geometry = LagrangianFormat.from_file(folder / "mesh.lnas").geometry
    if filename.exists():
        print(f"Already converted to file {filename}")
        return
    print()
    print(f"Converting file {filename_read}...")

    t0 = time.time()
    time_key = "time_stamp"
    interp_key = "p"
    idx_key = "point_index"

    df = pd.read_pickle(filename_read)
    # Set of triangles that each vertice participates to
    vertices_triangles_set, vertices_triangles_ratio = get_vertices_ratios(geometry)

    n_verts = len(geometry.vertices)
    n_steps = df[time_key].nunique()
    steps_arr = np.sort(df[time_key].unique())
    hs_triangles = np.zeros((n_steps, len(geometry.triangles)), dtype=np.float32)
    dfs_join = []

    t_convert = time.time()
    for step_idx, step in enumerate(steps_arr):
        df_use = df.iloc[n_verts * step_idx : n_verts * (step_idx + 1)]
        for idx, row in df_use.iterrows():
            # print(row)
            v_idx = int(row[idx_key])
            value = row[interp_key]
            v_ratio = vertices_triangles_ratio[v_idx]
            for t_idx, ratio in v_ratio:
                # FIXME Do we divide here by three?
                value_add = value * ratio
                hs_triangles[step_idx, t_idx] += value_add
        df_step = pd.DataFrame(
            {
                interp_key: hs_triangles[step_idx],
                idx_key: np.arange(len(geometry.triangles), dtype=np.int32),
            }
        )
        df_step[time_key] = step
        dfs_join.append(df_step)
    print(f"Conversion time {time.time() - t_convert:.2f}")

    t_save = time.time()
    df_save = pd.concat(dfs_join)
    df_save.to_hdf(filename, "hs")

    print(f"Save time {time.time()-t_save:.2f}")

    t_total = time.time() - t0

    print(f"Total time {t_total:.2f}. Converted file", filename)


def recursively_convert_folders(folder: pathlib.Path):
    if folder.is_dir():
        if (folder / "hist_series.pickle").exists():
            convert_folder(folder)

    for f in folder.iterdir():
        if f.is_dir():
            recursively_convert_folders(f)


def main(*args):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        required=True,
        help="Base folder to iter for convertion (converts hs from vertices to triangles)",
        type=str,
    )
    parsed_args = ap.parse_args(*args)

    path = pathlib.Path(parsed_args.p)
    recursively_convert_folders(path)


if __name__ == "__main__":
    main(sys.argv[1:])
