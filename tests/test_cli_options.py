"""Option parsing on the global CLI, exercised without running a pipeline."""

from __future__ import annotations

import pytest

from cfdmod.__main__ import _parse_size

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4G", 4 << 30),
        ("512M", 512 << 20),
        ("2K", 2048),
        ("1.5G", int(1.5 * (1 << 30))),
        ("2000000", 2_000_000),
        ("4GB", 4 << 30),
        (" 8g ", 8 << 30),
        (None, None),
    ],
)
def test_parse_size_accepts_human_units(text, expected):
    assert _parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "G", "abc", "-4G", "0"])
def test_parse_size_rejects_nonsense(text):
    with pytest.raises(ValueError):
        _parse_size(text)
