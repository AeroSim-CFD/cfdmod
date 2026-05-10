"""Time ops -- mutate the time axis only.

Three primitive ops, mirroring the proposal in issue #131:

- :func:`window_selection` -- restrict to a ``[start, end]`` interval.
- :func:`translate` -- set the time-at-index-0 to a new value.
- :func:`rescale` -- multiply the time axis by a scalar (e.g. to
  convert solver time to convective time).

Each op takes a :class:`DataSource` and a Pydantic params object and
returns a new :class:`DataSource`. Field arrays are sliced (window) or
left untouched (translate, rescale); the heavy data layer is the
:class:`FieldStore`, never re-materialised.
"""

from __future__ import annotations

__all__ = [
    "WindowSelectionParams",
    "TranslateParams",
    "RescaleTimeParams",
    "window_selection",
    "translate",
    "rescale",
]

from cfdmod.core.ops.time.rescale import RescaleTimeParams, rescale
from cfdmod.core.ops.time.translate import TranslateParams, translate
from cfdmod.core.ops.time.window import WindowSelectionParams, window_selection
