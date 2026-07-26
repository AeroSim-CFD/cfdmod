"""Guard: every repo path named in .claude/ agent and skill files must exist.

Agent definitions drift silently - a module moves, a script is deleted, and the
agent file keeps naming the old path. Nothing errors; the next agent just
follows a dead reference. This makes that a test failure instead.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / ".claude" / "check_agent_paths.py"


@pytest.mark.unit
def test_agent_files_reference_existing_paths():
    result = subprocess.run(
        [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr
