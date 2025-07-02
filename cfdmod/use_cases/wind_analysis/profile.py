from typing import Callable, Literal, TypeVar
import pandas as pd
import pathlib
from pydantic import BaseModel, ConfigDict, Field

def _validate_keys_df(df: pd.DataFrame, keys: list[str]):
    if any(k not in df.columns for k in keys):
        return False
    return True

class ProfileCalculator_NBR(BaseModel):
    """Data for wind analysis and calculation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pandas with keys: wind_direction, I, II, III, IV, V, Kd
    # Kd is optional and defaults to read, it defaults to one
    directional_data: pd.DataFrame
    V0: float
    U_H_overwrite: float | None = None

    @classmethod
    def build(cls, data_csv: pathlib.Path, V0: float, U_H_overwrite: float | None = None):
        df = pd.read_csv(data_csv, index_col=None)
        df = df.fillna(0)
        req_keys = ["wind_direction", "I", "II", "III", "IV", "V"]
        if not _validate_keys_df(df, req_keys):
            raise KeyError(
                "Not all required keys are in wind CSV. "
                f"Required ones are: {req_keys}, found {list(df.columns)}"
            )
        if "Kd" not in df.columns:
            df["Kd"] = 1
        df = df[req_keys + ["Kd"]]
        df.sort_values(by=["wind_direction"], inplace=True)
        return ProfileCalculator_NBR(directional_data=df, V0=V0, U_H_overwrite=U_H_overwrite)

    def S2(self, height: float, direction: float, time_filter_seconds: int|float):
        # parameters from NBR 6123, mean speed of 10min
        if time_filter_seconds not in [3, 600, 3600]:
            raise Exception('Currently S2 is implemented only for 3s, 10min and 1h')
        if time_filter_seconds ==3:
            Fr = 1
            p = {"I": 0.06, "II": 0.085, "III": 0.1, "IV": 0.12, "V": 0.15}
            b = {"I": 1.1, "II": 1.00, "III": 0.94, "IV": 0.86, "V": 0.74}
        if time_filter_seconds == 600:
            Fr = 0.69
            p = {"I": 0.095, "II": 0.15, "III": 0.185, "IV": 0.23, "V": 0.31}
            b = {"I": 1.23, "II": 1.00, "III": 0.86, "IV": 0.71, "V": 0.50}
        if time_filter_seconds == 3600:
            Fr = 0.65
            p = {"I": 0.1, "II": 0.16, "III": 0.2, "IV": 0.25, "V": 0.35}
            b = {"I": 1.25, "II": 1.00, "III": 0.85, "IV": 0.68, "V": 0.44}
            

        df = self.directional_data
        row = df.loc[(df["wind_direction"] - direction).abs().idxmin()].squeeze()
        sum_p = sum(row[k] * p[k] for k in p.keys())
        sum_b = sum(row[k] * b[k] for k in b.keys())
        return Fr * sum_b * (height / 10) ** sum_p

    def S3(self, recurrence_period: float):
        return 0.54 * (0.994 / recurrence_period) ** -0.157

    def get_U_H(self, height: float, direction: float, recurrence_period: float, time_filter_seconds: float=600) -> float:
        if self.U_H_overwrite is not None:
            return self.U_H_overwrite

        df = self.directional_data
        row = df.loc[(df["wind_direction"] - direction).abs().idxmin()].squeeze()
        V0 = self.V0
        kd = row["Kd"]
        S2 = self.S2(height, direction, time_filter_seconds)
        S3 = self.S3(recurrence_period)
        return V0 * kd * S2 * S3