from dataclasses import dataclass


@dataclass
class NormalizationParameters:
    reference_velocity: float
    characteristic_length: float
