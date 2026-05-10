"""Field ops -- overwrite or create same-shape fields on a data source.

Field ops never change the shape of the elements axis or the time axis;
they only rewrite a field's values. Numpy-style algebra (the four
broadcasting rules from issue #131) lives in
:mod:`cfdmod.core.algebra`; the temporal-filter family lives here.
"""

from __future__ import annotations

__all__ = [
    "MovingAverageParams",
    "moving_average",
]

from cfdmod.core.ops.field.moving_average import MovingAverageParams, moving_average
