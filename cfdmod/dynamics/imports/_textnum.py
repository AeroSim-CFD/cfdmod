"""Small parsing helpers for structural-export text tables.

The Brazilian structural tools export numbers with a comma decimal
separator (``"7,036E+00"``) in Latin-1 text files whose comment lines
start with ``//``. These helpers isolate that quirk so the format
parsers stay readable.
"""

from __future__ import annotations

__all__ = ["to_float", "iter_data_rows", "norm_text"]

import pathlib
import unicodedata
from typing import Iterator


def to_float(token: str) -> float:
    """Parse a possibly comma-decimal numeric token (``"7,036E+00"`` -> float)."""
    return float(token.strip().replace(",", "."))


def norm_text(text) -> str:
    """Accent- and case-insensitive normalized text, for header matching."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()


def iter_data_rows(
    path: str | pathlib.Path,
    *,
    encoding: str = "latin-1",
    comment: str = "//",
    sep: str | None = None,
) -> Iterator[list[str]]:
    """Yield split token lists for each non-comment, non-empty line.

    Comment lines (``//`` by default) and blank lines are skipped. With
    ``sep=None`` tokens split on any run of whitespace (TQS PORTELS, where
    fields are single tokens); pass ``sep="\\t"`` for tables whose fields may
    contain spaces (e.g. PORTICO floor names like ``"T. CAIXA D'AGUA"``), so
    only tabs delimit.
    """
    with pathlib.Path(path).open("r", encoding=encoding) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(comment):
                continue
            if sep is None:
                yield line.split()
            else:
                yield [tok.strip() for tok in line.split(sep)]
