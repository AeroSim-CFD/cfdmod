"""Small parsing helpers for structural-export text tables.

The Brazilian structural tools export numbers with a comma decimal
separator (``"7,036E+00"``) in Latin-1 text files whose comment lines
start with ``//``. These helpers isolate that quirk so the format
parsers stay readable.
"""

from __future__ import annotations

__all__ = ["to_float", "iter_data_rows"]

import pathlib
from typing import Iterator


def to_float(token: str) -> float:
    """Parse a possibly comma-decimal numeric token (``"7,036E+00"`` -> float)."""
    return float(token.strip().replace(",", "."))


def iter_data_rows(
    path: str | pathlib.Path, *, encoding: str = "latin-1", comment: str = "//"
) -> Iterator[list[str]]:
    """Yield whitespace/TAB-split token lists for each non-comment, non-empty line.

    Comment lines (``//`` by default) and blank lines are skipped. Tokens
    are split on any run of whitespace, so both TAB- and space-separated
    tables parse.
    """
    with pathlib.Path(path).open("r", encoding=encoding) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(comment):
                continue
            yield line.split()
