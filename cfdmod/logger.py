"""The library's logger, and the opt-in console configuration for the CLI.

A library gets a named logger and a ``NullHandler``; the application decides
level, destination and format. This module used to do the opposite -- at import
it set ``DEBUG`` on the ``cfdmod`` logger and attached a colour
``StreamHandler`` to stdout -- which took over logging for anything that
imported cfdmod. A service got unfiltered, ANSI-wrapped DEBUG in its job log
and no way to redirect it without reaching in and mutating the logger object.

So: importing cfdmod now configures nothing. The colour formatter is still
here, behind :func:`configure_logging`, which the CLI calls -- the terminal
experience is unchanged, it is just a choice now. Colour follows ``isatty``
rather than being emitted unconditionally, so a redirected run gets plain text
instead of escape codes in the file.
"""

from __future__ import annotations

__all__ = ["logger", "configure_logging", "CustomFormatter", "LOGGER_NAME"]

import logging
import sys

LOGGER_NAME = "cfdmod"

_FMT = "[%(asctime)s] [%(levelname)s] - %(name)s - %(message)s"

# Marks the handler configure_logging installed, so a second call replaces it
# instead of stacking another one and printing everything twice.
_CONSOLE_FLAG = "_cfdmod_console"


class CustomFormatter(logging.Formatter):
    """Level-coloured console formatter.

    Colour is a constructor argument rather than always-on: writing escape
    codes into a redirected log file helps nobody.
    """

    fmt_str = _FMT

    def __init__(self, *, color: bool = True) -> None:
        super().__init__(_FMT)
        self.color = color
        self._by_level: dict[int, logging.Formatter] = {}
        if not color:
            return
        from colorama import Fore

        for level, shade in (
            (logging.DEBUG, Fore.LIGHTMAGENTA_EX),
            (logging.INFO, Fore.WHITE),
            (logging.WARNING, Fore.YELLOW),
            (logging.ERROR, Fore.LIGHTRED_EX),
            (logging.CRITICAL, Fore.RED),
        ):
            self._by_level[level] = logging.Formatter(f"{shade}{_FMT}{Fore.RESET}")

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._by_level.get(record.levelno)
        if formatter is None:
            return super().format(record)
        return formatter.format(record)


# The library's logger. No level and no real handler: whatever imports cfdmod
# decides those. NullHandler exists only to suppress the "no handlers" warning.
logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream=None,
    color: bool | None = None,
) -> logging.Handler:
    """Attach a console handler to the ``cfdmod`` logger. **Applications only.**

    Library code must never call this. The CLI does, which is why the terminal
    output is unchanged; anything embedding cfdmod configures logging its own
    way and is not overridden.

    Idempotent: a second call replaces the handler the first one installed.

    Args:
        level: Level to set on the ``cfdmod`` logger.
        stream: Destination, default ``sys.stderr``. Logs are diagnostics, and
            keeping them off stdout is what lets a command's real output be
            piped. (The old import-time handler wrote to stdout, mixing the
            two.)
        color: Force colour on or off. ``None`` auto-detects ``isatty``.

    Returns:
        The installed handler, so a caller can tune or remove it.
    """
    destination = sys.stderr if stream is None else stream
    if color is None:
        color = bool(getattr(destination, "isatty", lambda: False)())

    for existing in list(logger.handlers):
        if getattr(existing, _CONSOLE_FLAG, False):
            logger.removeHandler(existing)

    handler = logging.StreamHandler(destination)
    handler.setFormatter(CustomFormatter(color=color))
    setattr(handler, _CONSOLE_FLAG, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler
