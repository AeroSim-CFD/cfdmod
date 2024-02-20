from __future__ import annotations

import pathlib

from pydantic import BaseModel, Field, field_validator, model_validator

from cfdmod.utils import read_yaml


class SnapshotConfig(BaseModel):
    @classmethod
    def from_file(cls, filename: pathlib.Path) -> SnapshotConfig:
        yaml_vals = read_yaml(filename)
        cfg = cls(**yaml_vals)

        return cfg
