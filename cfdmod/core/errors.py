"""Typed exception hierarchy for the v3 core (issue #147).

A service consumer that drives templates through :func:`run_template`
otherwise has to string-match bare ``KeyError`` / ``ValueError`` to map
failures onto its own error codes. The classes here give it precise,
catchable types while staying fully backward compatible: each subclasses
the builtin it replaced, so existing ``except (KeyError, ValueError)``
handlers keep working unchanged.

Hierarchy::

    CfdmodError
    +- TemplateError            (also a ValueError)  -- bad template shape
    |   +- TemplateReferenceError (also a KeyError)  -- dangling / unknown ref
    +- OpError                  (also a RuntimeError) -- op raised while running
    +- StorageKeyError          (also a KeyError)    -- missing storage key
"""

from __future__ import annotations

__all__ = [
    "CfdmodError",
    "TemplateError",
    "TemplateReferenceError",
    "OpError",
    "StorageKeyError",
]


class CfdmodError(Exception):
    """Base class for every error cfdmod raises deliberately."""

    def __str__(self) -> str:
        # KeyError-derived subclasses would otherwise wrap the message in
        # repr() quotes; defining __str__ here (ahead of KeyError in the MRO)
        # keeps every cfdmod error message plain.
        if len(self.args) == 1 and isinstance(self.args[0], str):
            return self.args[0]
        return super().__str__()


class TemplateError(CfdmodError, ValueError):
    """A pipeline template is malformed.

    Covers contract violations (bad op wiring), a ``rhs`` on a unary op,
    duplicate step ids, and an input whose loaded kind does not match its
    declaration. Subclasses ``ValueError`` for backward compatibility.
    """


class TemplateReferenceError(TemplateError, KeyError):
    """A template references something that does not exist.

    A dangling ``source`` / ``rhs`` / output reference, or an unknown op
    kind. Subclasses ``KeyError`` (these were raised as ``KeyError``
    before the hierarchy existed) and :class:`TemplateError`.
    """


class OpError(CfdmodError, RuntimeError):
    """An op raised while :func:`run_template` executed a step.

    Carries the failing step id and op kind so a consumer can report
    which step blew up without parsing the message.
    """

    def __init__(
        self, message: str, *, step_id: str | None = None, op_kind: str | None = None
    ) -> None:
        super().__init__(message)
        self.step_id = step_id
        self.op_kind = op_kind


class StorageKeyError(CfdmodError, KeyError):
    """A storage backend has no data source under the requested key.

    Subclasses ``KeyError`` for backward compatibility.
    """
