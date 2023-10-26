import argparse
import pathlib
import sys

import numpy as np
from nassu.lnas import LagrangianFormat, LagrangianGeometry


def convert_folder(folder: pathlib.Path):
    if (folder / "mesh.lnas").exists():
        print(f"Alread converted to file {(folder / 'mesh.lnas')}")
        return

    points = np.fromfile(folder / "mesh_points.npy", dtype=np.float32)
    points = np.reshape(points, (points.shape[0] // 3, 3))
    triangles = np.fromfile(folder / "mesh_triangles.npy", dtype=np.int32)
    triangles = np.reshape(triangles, (triangles.shape[0] // 3, 3))

    lnas_geometry = LagrangianGeometry(vertices=points, triangles=triangles)
    lnas = LagrangianFormat(
        version="v0.4.3", name=folder.name, normalization=None, geometry=lnas_geometry, surfaces={}
    )
    lnas.to_file(folder / "mesh.lnas")
    print("Converted file", folder / "mesh.lnas")

    # lnas_from = LagrangianFormat.from_file(folder / "mesh.lnas")
    # np.testing.assert_equal(lnas_from.geometry.vertices, points)
    # np.testing.assert_equal(lnas_from.geometry.triangles, triangles)


def recursively_convert_folders(folder: pathlib.Path):
    if folder.is_dir():
        if (folder / "mesh_points.npy").exists():
            convert_folder(folder)

    for f in folder.iterdir():
        if f.is_dir():
            recursively_convert_folders(f)


def main(*args):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        required=True,
        help="Base folder to iter for convertion (converts triangles and vertices to lnas)",
        type=str,
    )
    parsed_args = ap.parse_args(*args)

    path = pathlib.Path(parsed_args.p)
    recursively_convert_folders(path)


if __name__ == "__main__":
    main(sys.argv[1:])
