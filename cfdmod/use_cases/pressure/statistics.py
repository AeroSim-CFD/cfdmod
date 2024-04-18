__all__ = ["Statistics"]

from typing import Literal

Statistics = Literal[
    "max", "min", "std", "mean", "mean_eq", "skewness", "kurtosis", "xtr_min", "xtr_max"
]
