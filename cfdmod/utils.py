import pathlib
from typing import Any

import numpy as np
import pandas as pd
from ruamel.yaml import YAML


def create_folders_for_file(filename: pathlib.Path):
    """Creates folders to save given file

    Args:
        filename (pathlib.Path): Filename to setup folder
    """

    filename.parent.mkdir(parents=True, exist_ok=True)


def create_folder_path(path: pathlib.Path):
    """Creates folders path

    Args:
        path (pathlib.Path): Path to create
    """

    path.mkdir(parents=True, exist_ok=True)


def read_yaml(filename: pathlib.Path) -> Any:
    """Read YAML from file

    Args:
        filename (str): File to read from

    Raises:
        Exception: Unable to read YAML from file

    Returns:
        Any: YAML content as python objects (dict, list, etc.)
    """
    if not filename.exists():
        raise Exception(f"Unable to read yaml. Filename {filename} does not exists")

    # Read YAML from file
    with open(filename, "r", encoding="utf-8") as f:
        try:
            yaml = YAML(typ="safe")
            return yaml.load(f)
        except Exception as e:
            raise Exception(f"Unable to load YAML from {filename}. Exception {e}") from e


def convert_dataframe_into_matrix(
    source_dataframe: pd.DataFrame,
    row_data_label: str = "time_step",
    column_data_label: str = "point_idx",
    value_data_label: str = "rho",
) -> pd.DataFrame:
    """Converts a dataframe into a matrix form representation

    Args:
        source_dataframe (pd.DataFrame): Source dataframe
        row_data_label (str): Label for the row values (index). Defaults to "time_step".
        column_data_label (str): Label for the column values. Defaults to "point_idx".
        value_data_label (str): Label for the values. Defaults to "rho".

    Returns:
        pd.DataFrame: Matrix form of the dataframe
    """
    expected_columns = [row_data_label, column_data_label, value_data_label]
    df_columns = source_dataframe.columns
    if not all([col in df_columns for col in expected_columns]):
        raise KeyError(
            f"Dataframe columns {df_columns} must contain target columns {expected_columns}"
        )
    matrix = pd.pivot(
        data=source_dataframe,
        index=row_data_label,
        columns=column_data_label,
        values=value_data_label,
    )
    matrix.reset_index(inplace=True)
    matrix.index.name = f"{row_data_label}_idx"

    return matrix


def convert_matrix_into_dataframe(
    matrix_df: pd.DataFrame,
    row_data_label: str = "time_step",
    column_data_label: str = "point_idx",
    value_data_label: str = "rho",
    column_order: tuple[str, str, str] = (),
    sort_order: list[str] = [],
    column_dtype: np.dtype = np.int32,
) -> pd.DataFrame:
    """Converts a matrix into a dataframe form representation

    Args:
        matrix_df (pd.DataFrame): Matrix dataframe to convert
        row_data_label (str): Label for the row values (index). Defaults to "time_step".
        column_data_label (str): Label for the column values. Defaults to "point_idx".
        value_data_label (str): Label for the values. Defaults to "rho".
        column_order (tuple[str, str, str], optional): _description_. Defaults to ().
        sort_order (list[str], optional): Order for sorting columns. Defaults to [].
        column_dtype (np.dtype, optional): Type of the values used to index the columns. Defaults to np.int32.

    Returns:
        pd.DataFrame: Dataframe form of the matrix
    """
    default_column_order = [column_data_label, value_data_label, row_data_label]
    default_sort_order = [row_data_label, column_data_label]
    if len(column_order) != 3 and len(column_order) != 0:
        raise Exception("Column order must have 3 or 0 elements")
    dataframe = pd.melt(
        matrix_df, id_vars=row_data_label, var_name=column_data_label, value_name=value_data_label
    )
    if len(sort_order) != 0:
        dataframe.sort_values(by=sort_order, inplace=True)
    else:
        dataframe.sort_values(by=default_sort_order, inplace=True)
    dataframe.reset_index(inplace=True)
    dataframe[column_data_label] = dataframe[column_data_label].astype(column_dtype)
    dataframe = dataframe[
        default_column_order if len(column_order) == 0 else [c for c in column_order]
    ]

    return dataframe
