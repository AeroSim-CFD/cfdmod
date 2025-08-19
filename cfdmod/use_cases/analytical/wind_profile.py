from typing import Callable, Literal, TypeVar
import pandas as pd
import pathlib
from pydantic import BaseModel, ConfigDict, Field
from cfdmod.use_cases.climate.wind_profile import WindProfile
from cfdmod import utils


class WindProfile_NBR(WindProfile):
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
        if not utils.validate_keys_df(df, req_keys):
            raise KeyError(
                "Not all required keys are in wind CSV. "
                f"Required ones are: {req_keys}, found {list(df.columns)}"
            )
        if "Kd" not in df.columns:
            df["Kd"] = 1
        df = df[req_keys + ["Kd"]]
        df.sort_values(by=["wind_direction"], inplace=True)
        return WindProfile_NBR(directional_data=df, V0=V0, U_H_overwrite=U_H_overwrite)

    def get_opencountry_profile(self):
        directional_data_cat2 = self.directional_data.copy()
        for cat in ['I','III','IV','V']:
            directional_data_cat2[cat] = 0
        directional_data_cat2["II"] = 1
        return WindProfile(U_H_overwrite=self.U_H_overwrite, directional_data=directional_data_cat2, V0=self.V0)

    def p(self, direction: float, time_filter_seconds: int|float):
        validate_time_filter(time_filter_seconds)
        p_database = {
            3: {"I": 0.06, "II": 0.085, "III": 0.1, "IV": 0.12, "V": 0.15},
            5: {"I": 0.065, "II": 0.09, "III": 0.105, "IV": 0.125, "V": 0.16},
            10: {"I": 0.07, "II": 0.1, "III": 0.115, "IV": 0.135, "V": 0.175},
            600: {"I": 0.095, "II": 0.15, "III": 0.185, "IV": 0.23, "V": 0.31},
            3600: {"I": 0.1, "II": 0.16, "III": 0.2, "IV": 0.25, "V": 0.35}
        }
        p = p_database[time_filter_seconds]
        df = self.directional_data
        row = df.loc[(df["wind_direction"] - direction).abs().idxmin()].squeeze()
        return sum(row[k] * p[k] for k in p.keys())
    
    def b(self, direction: float, time_filter_seconds: int|float):
        validate_time_filter(time_filter_seconds)
        b_database = {
            3: {"I": 1.1, "II": 1.00, "III": 0.94, "IV": 0.86, "V": 0.74},
            5: {"I": 1.11, "II": 1.00, "III": 0.94, "IV": 0.85, "V": 0.73},
            10: {"I": 1.12, "II": 1.00, "III": 0.93, "IV": 0.84, "V": 0.71},
            600: {"I": 1.23, "II": 1.00, "III": 0.86, "IV": 0.71, "V": 0.50},
            3600: {"I": 1.25, "II": 1.00, "III": 0.85, "IV": 0.68, "V": 0.44},
        }
        b = b_database[time_filter_seconds]
        df = self.directional_data
        row = df.loc[(df["wind_direction"] - direction).abs().idxmin()].squeeze()
        return sum(row[k] * b[k] for k in b.keys())

    def F_r(self, time_filter_seconds: int|float):
        validate_time_filter(time_filter_seconds)
        Fr = {
            3: 1,
            5: 0.98,
            10: 0.95,
            600: 0.69,
            2600: 0.65,
        }
        return Fr[time_filter_seconds]

    def S2(self, height: float, direction: float, time_filter_seconds: int|float):
        # parameters from NBR 6123, mean speed of 10min
        p = self.p(direction, time_filter_seconds)
        b = self.b(direction, time_filter_seconds)
        Fr = self.F_r(time_filter_seconds)
        return Fr * b * (height / 10) ** p

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

class WindProfile_EU(WindProfile):
    """Data for wind analysis and calculation for EU standard EN1991"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pandas with keys: wind_direction, I, II, III, IV, V, Kd
    # Kd is optional and defaults to read, it defaults to one
    directional_data: pd.DataFrame
    Vb: float
    U_H_overwrite: float | None = None

    @classmethod
    def build(cls, data_csv: pathlib.Path, Vb: float, U_H_overwrite: float | None = None):
        df = pd.read_csv(data_csv, index_col=None)
        req_keys = ["wind_direction", "z0"]
        if not utils.validate_keys_df(df, req_keys):
            raise KeyError(
                "Not all required keys are in wind CSV. "
                f"Required ones are: {req_keys}, found {list(df.columns)}"
            )
        if "Kd" not in df.columns:
            df["Kd"] = 1
        df = df[req_keys + ["Kd"]]
        df.sort_values(by=["wind_direction"], inplace=True)
        return WindProfile_EU(directional_data=df, Vb=Vb, U_H_overwrite=U_H_overwrite)

    def get_opencountry_profile(self):
        directional_data_cat2 = self.directional_data.copy()
        directional_data_cat2["z0"] = 0.05
        return WindProfile(U_H_overwrite=self.U_H_overwrite, directional_data=directional_data_cat2, Vb=self.Vb)

    def kr(self, direction: float):
        direction = float(direction)
        df = self.directional_data
        row = df.loc[df["wind_direction"] == direction].squeeze()
        return 0.19*(row['z0']/0.05)**0.07

    def c_prob(self, rec_period: float=50) -> float:
        K=0.2
        n=0.5
        p = 1/rec_period
        return ((1-K*np.log(-np.log(1-p)))/(1-K*np.log(-np.log(0.98))))**n
    
    def c_r(self, height: float, direction: float) -> float:
        direction = float(direction)
        df = self.directional_data
        row = df.loc[df["wind_direction"] == direction].squeeze()
        return self.kr(direction)*np.log(height/row['z0'])

    def get_U_H(
        self, height: float, direction: float, recurrence_period: float=50, use_kd: bool=True
    ) -> float:
        if self.U_H_overwrite is not None:
            return self.U_H_overwrite

        direction = float(direction)
        df = self.directional_data
        row = df.loc[df["wind_direction"] == direction].squeeze()
        Vb = self.Vb
        Kd = row["Kd"] if use_kd else 1
        c_season = 1 # for future implementations, ...maybe 
        c_r = self.c_r(height, direction)
        c_prob = self.c_prob(recurrence_period)
        return (Vb*Kd*c_prob*c_season) * c_r


def validate_time_filter(time_filter_seconds: int|float):
    if time_filter_seconds not in [3, 5, 10, 600, 3600]:
        raise Exception('S2 is implemented only for 3s, 5s, 10s, 10min and 1h')