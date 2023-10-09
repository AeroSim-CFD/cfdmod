import pandas as pd

from cfdmod.use_cases.pressure.statistics import Statistics


def calculate_statistics(
    region_data: pd.DataFrame, statistics_to_apply: list[Statistics]
) -> pd.DataFrame:
    """Calculates statistics for pressure coefficient of a body data

    Args:
        region_data (pd.DataFrame): Dataframe of the region data shape coefficients
        statistics_to_apply (Statistics): List of statistical functions to apply

    Returns:
        pd.DataFrame: Statistics for shape coefficient
    """
    group_by_point = region_data.groupby("region_idx")

    statistics_data = pd.DataFrame({"region_idx": region_data["region_idx"].unique()})

    if "avg" in statistics_to_apply:
        statistics_data["Ce_avg"] = group_by_point["Ce"].mean()
    if "min" in statistics_to_apply:
        statistics_data["Ce_min"] = group_by_point["Ce"].min()
    if "max" in statistics_to_apply:
        statistics_data["Ce_max"] = group_by_point["Ce"].max()
    if "std" in statistics_to_apply:
        statistics_data["Ce_rms"] = group_by_point["Ce"].std()

    # Calculate skewness and kurtosis using apply
    if "skewness" in statistics_to_apply:
        skewness = group_by_point["Ce"].apply(lambda x: x.skew()).reset_index(name="skewness")
        statistics_data["Ce_skewness"] = skewness["skewness"]
    if "kurtosis" in statistics_to_apply:
        kurtosis = group_by_point["Ce"].apply(lambda x: x.kurt()).reset_index(name="kurtosis")
        statistics_data["Ce_kurtosis"] = kurtosis["kurtosis"]

    return statistics_data
