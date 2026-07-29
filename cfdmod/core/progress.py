"""Progress reporting and cooperative cancellation for a template run.

A post-processing run on a real case takes minutes. Without these, a caller --
the CLI, and above all the interface -- can only block until it returns: no bar,
no way to kill a job whose user walked away, no way to see which step is slow.

Two separate seams, deliberately:

- :class:`RunEvent` + an ``on_progress`` callback, for reporting.
- a ``cancel`` predicate, for stopping.

They are not merged into "the progress callback raises to cancel", which reads
neatly and behaves badly: every exception in a caller's logging code would then
look like a deliberate abort.

**What cancellation promises.** cfdmod cannot interrupt numpy mid-call and does
not pretend to. The predicate is polled at boundaries -- between steps, between
time windows, and before each output is written -- so a cancelled run stops at
the next boundary and, because the check precedes every write, does not leave a
half-written output set behind. A single very long op will run to completion
before the run notices.
"""

from __future__ import annotations

__all__ = ["RunCancelled", "RunEvent", "RunPhase"]

from dataclasses import dataclass
from typing import Literal

from cfdmod.core.errors import CfdmodError

RunPhase = Literal["load", "step", "write"]
"""Which part of the run an event belongs to.

``load`` and ``write`` are included because on a big case they are a real share
of the wall clock -- a bar that sits at 0% while inputs load and then jumps to
100% is worse than no bar.
"""


@dataclass(frozen=True)
class RunEvent:
    """One unit of progress within a template run.

    A frozen object rather than a tuple so fields can be added later without
    breaking a caller's unpacking.

    Attributes:
        phase: ``load``, ``step`` or ``write``.
        name: The input name, step id, or output name.
        index: 0-based position within the phase.
        total: How many units this phase has.
        op_kind: The op being executed. ``None`` outside the ``step`` phase.
        window: 0-based time window, when the run is chunked. ``None`` when it
            is not.
        n_windows: How many time windows the run has, or ``None`` when
            unchunked.
    """

    phase: RunPhase
    name: str
    index: int
    total: int
    op_kind: str | None = None
    window: int | None = None
    n_windows: int | None = None

    @property
    def fraction(self) -> float:
        """Progress through this phase, in ``[0, 1]``.

        Deliberately per-phase rather than a whole-run estimate: the runner
        cannot know how the cost splits between loading, computing and writing
        without timing it, and a confidently wrong overall percentage is worse
        than an honest per-phase one.
        """
        if self.total <= 0:
            return 1.0
        return min(1.0, (self.index + 1) / self.total)

    def describe(self) -> str:
        """One line for a log or a terminal."""
        where = f"{self.index + 1}/{self.total}"
        if self.n_windows:
            where += f" window {(self.window or 0) + 1}/{self.n_windows}"
        kind = f" ({self.op_kind})" if self.op_kind else ""
        return f"[{self.phase}] {where} {self.name}{kind}"


class RunCancelled(CfdmodError):
    """Raised when the caller's ``cancel`` predicate returned True.

    Carries where the run stopped so a consumer can report it without parsing
    the message. It is a :class:`~cfdmod.core.errors.CfdmodError`, not a bare
    ``RuntimeError``, so "the user cancelled" is distinguishable from "the run
    failed" -- they mean very different things to whatever is watching the job.
    """

    def __init__(self, phase: RunPhase, name: str) -> None:
        super().__init__(f"run cancelled before {phase} {name!r}")
        self.phase = phase
        self.name = name
