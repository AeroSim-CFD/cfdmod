from .vertices import SectionVertices
from .shed import Shed
from .section import AltimetrySection
from .probe import AltimetryProbe

__doc__ = """
Altimetry module

This modules provides classes and functions for processing a surface
and plotting the resulting sections. These sections are defined by a plane,
that is defined by its normal and origin.
"""

__all__ = [
    "SectionVertices",
    "Shed",
    "AltimetrySection",
    "AltimetryProbe",
]
