import numpy as np
import pandas as pd

InflowProfileType = "Velocity" | "Pressure"


class InflowProfile:
    profile_type: InflowProfileType
    profile_data: pd.DataFrame
    profile_points: pd.DataFrame
