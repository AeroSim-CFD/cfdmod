#!/usr/bin/env python3
"""Fail when an agent or skill file points at a repo path that does not exist.

Agent definitions go stale silently. A directory is renamed, a script is
deleted, a module moves - and the `.claude/` file keeps confidently naming the
old path. Nothing errors: the next agent just follows a dead reference and
hand-rolls something instead of reusing what is already there. That failure mode
is the reason this exists.

It scans every `.claude/agents/*.md` and `.claude/skills/**/*.md` for
backtick-quoted repository paths and checks that each one still resolves.

Run directly:

    python3 .claude/check_agent_paths.py

Two optional side files live next to this script:

`agent_paths_allow.txt`
    Paths the checker must not flag, one per line, `#` for comments. Legitimate
    entries: build output that is generated rather than committed, and a path a
    file names precisely in order to say it does NOT exist. Keep it short - every
    entry is a path this check can no longer protect for you.

`agent_paths_bases.txt`
    Extra directories to resolve against, one per line, for repos where agents
    write paths relative to a nested project dir rather than the repo root.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CLAUDE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLAUDE_DIR.parent
ALLOWLIST = CLAUDE_DIR / "agent_paths_allow.txt"
BASES = CLAUDE_DIR / "agent_paths_bases.txt"

# Only look inside backticks. Prose mentions of a name are not path claims.
TOKEN = re.compile(r"`([^`\n]+)`")

# Templated or elided, so not a literal claim about the tree.
PLACEHOLDER = set("<>*${}|")


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.rstrip("/"))
    return out


def search_bases() -> list[Path]:
    bases = [REPO_ROOT]
    for extra in _read_lines(BASES):
        candidate = REPO_ROOT / extra
        if candidate.is_dir():
            bases.append(candidate)
    return bases


def is_path_claim(token: str, roots: set[str]) -> bool:
    """True when the token is a literal claim that a repo path exists."""
    if any(c in token for c in PLACEHOLDER):
        return False
    if "://" in token or token.startswith(("http", "www.")):
        return False
    if "..." in token or token.endswith((".", ",")):
        return False
    if " " in token:  # a command line, not a path
        return False
    return token.split("/", 1)[0] in roots


def sources() -> list[Path]:
    found = sorted((CLAUDE_DIR / "agents").glob("*.md"))
    skills = CLAUDE_DIR / "skills"
    if skills.exists():
        found += sorted(skills.rglob("*.md"))
    return found


def scan() -> list[tuple[str, str]]:
    bases = search_bases()
    # A token is a candidate if its first segment names a real entry under any
    # search base, so agents may write repo-root- or project-dir-relative paths.
    roots: set[str] = set()
    for base in bases:
        roots |= {p.name for p in base.iterdir()}

    allow = set(_read_lines(ALLOWLIST))
    missing: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for src in sources():
        rel = str(src.relative_to(REPO_ROOT))
        for raw in TOKEN.findall(src.read_text()):
            token = raw.strip().rstrip("/")
            if not is_path_claim(token, roots) or token in allow:
                continue
            if any((base / token).exists() for base in bases):
                continue
            key = (rel, token)
            if key in seen:
                continue
            seen.add(key)
            missing.append(key)
    return missing


def main() -> int:
    missing = scan()
    if not missing:
        print(f"ok: every path referenced in {len(sources())} agent/skill files resolves")
        return 0
    print("Agent/skill files reference paths that do not exist:\n", file=sys.stderr)
    width = max(len(src) for src, _ in missing)
    for src, token in missing:
        print(f"  {src:<{width}}  ->  {token}", file=sys.stderr)
    print(
        f"\n{len(missing)} broken reference(s). Fix the path, or - if it is generated at "
        "runtime, or named precisely to say it does NOT exist - add it to "
        f"{ALLOWLIST.name} with a comment saying which.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
