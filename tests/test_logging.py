"""Importing cfdmod must not configure logging; the CLI opts in.

`cfdmod/logger.py` used to set DEBUG on the `cfdmod` logger and attach a colour
StreamHandler to stdout at import. A library doing that takes over logging for
whatever imports it: a service got unfiltered ANSI-wrapped DEBUG in its job log
with no way to redirect it, and a notebook got the same chatter.
"""

from __future__ import annotations

import io
import logging
import subprocess
import sys
import textwrap

import pytest

from cfdmod.logger import LOGGER_NAME, CustomFormatter, configure_logging, logger

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_logger_state():
    """Leave the ``cfdmod`` logger exactly as found."""
    handlers = list(logger.handlers)
    level = logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_importing_cfdmod_installs_only_a_null_handler():
    """Run in a fresh interpreter: a test that already imported cfdmod would
    not notice a handler added at import time."""
    code = textwrap.dedent(
        """
        import logging, sys
        import cfdmod
        import cfdmod.logger  # the module that used to configure at import
        lg = logging.getLogger("cfdmod")
        real = [h for h in lg.handlers if not isinstance(h, logging.NullHandler)]
        print(f"handlers={len(real)} level={lg.level} root={len(logging.root.handlers)}")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout.strip()
    # level 0 == NOTSET: the library states no opinion.
    assert out == "handlers=0 level=0 root=0"


def test_configure_logging_attaches_a_handler_and_sets_the_level():
    stream = io.StringIO()
    configure_logging(logging.DEBUG, stream=stream, color=False)

    logger.debug("hello")
    assert "hello" in stream.getvalue()
    assert logger.level == logging.DEBUG


def test_configure_logging_is_idempotent():
    """A second call must replace, not stack -- otherwise output doubles."""
    stream = io.StringIO()
    configure_logging(logging.INFO, stream=stream, color=False)
    configure_logging(logging.INFO, stream=stream, color=False)

    logger.info("once")
    assert stream.getvalue().count("once") == 1


def test_configure_logging_defaults_to_stderr():
    """Diagnostics off stdout, so a command's real output stays pipeable."""
    handler = configure_logging(logging.INFO)
    assert handler.stream is sys.stderr


def test_colour_follows_isatty_by_default():
    """A redirected run must not collect escape codes."""
    not_a_tty = io.StringIO()
    configure_logging(logging.INFO, stream=not_a_tty)
    logger.info("plain")
    assert "\x1b[" not in not_a_tty.getvalue()


def test_colour_can_be_forced_on():
    stream = io.StringIO()
    configure_logging(logging.INFO, stream=stream, color=True)
    logger.info("bright")
    assert "\x1b[" in stream.getvalue()


def test_formatter_without_colour_emits_no_escape_codes():
    record = logging.LogRecord(LOGGER_NAME, logging.WARNING, __file__, 1, "msg", None, None)
    assert "\x1b[" not in CustomFormatter(color=False).format(record)


def test_an_unconfigured_library_logger_emits_nothing():
    """The state an embedding application starts from."""
    logger.handlers[:] = [logging.NullHandler()]
    logger.setLevel(logging.NOTSET)

    stream = io.StringIO()
    captured = logging.StreamHandler(stream)
    logging.getLogger().addHandler(captured)
    try:
        logger.debug("should not appear anywhere")
    finally:
        logging.getLogger().removeHandler(captured)

    assert stream.getvalue() == ""


def test_logging_logger_satisfies_the_logger_protocol():
    """`Logger` is structural, so no adapter is needed to inject the real one."""
    from cfdmod.core.protocols import Logger

    assert isinstance(logger, Logger)
