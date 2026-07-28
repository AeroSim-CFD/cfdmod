---
name: fix-issue
description: Take a cfdmod GitHub issue from "read it" to "merged and closed" - branch off main in a worktree, post the plan, implement with tests, verify the gates, review your own diff, open the PR, squash-merge it, and confirm it landed. Use when asked to "fix issue N", "take #N", "resolve and PR this", or to merge a PR of your own that is ready.
---

# Fix a cfdmod issue end to end

Covers the whole loop for `AeroSim-CFD/cfdmod`: read -> plan -> worktree -> implement ->
verify -> **review your own diff** -> PR -> **squash-merge** -> confirm it landed -> clean up.

**This skill carries an explicit merge authorization from Waine.** The project convention is
never to commit to `main` directly and never to leave work sitting on a local branch. Inside
this workflow you are additionally authorized to merge your own issue-fix PR and to flip your
own draft to ready. The guardrails that stay hard are at the bottom.

> **A merge here does not reach a user.** Unlike the aerosim web app, `main` on cfdmod deploys
> nothing. `aerosim-cfdmod` reaches users only through a deliberate, separately-authorized
> release (the `cfdmod-release` skill: version bump, tag, GitHub release, PyPI publish), and
> **releases are always Waine's call**. That is what makes merging your own fix low-risk here.
>
> What a merge *does* reach: `main` is what consulting cases and the GPU-server checkout track,
> and it is the base every other branch rebases onto. So a red `main` costs everyone else time
> even though it costs no user anything.

## 1. Read the issue

`gh issue view` works normally in this repo.

```bash
gh issue view <N> --json number,title,state,labels,body --jq '{number,title,state,labels:[.labels[].name],body}'
gh issue view <N> --comments
```

Follow the issue's own references. When the issue is about code a previous PR introduced, read
that PR - `gh pr view <M>` and `gh pr diff <M>` - because the design intent and the previous
attempt usually live there, not in the issue.

Read `CLAUDE.md` for the conventions before writing code rather than re-deriving them: the
`core/` -> `ops` -> `recipes` layering, the per-module `cli.py` / `run.py` / `parameters.py` /
`functions.py` split, Pydantic v2 configs with `from_file`, lnas over trimesh for meshes, and
the ASCII-only rule.

## 2. Post the plan

`CLAUDE.md` requires the implementation plan to live on the issue, not in the chat:

```bash
gh issue comment <N> --body "..."
```

Post it and keep going. Wait for approval only when the issue turns out to need a design
decision it does not itself settle (a new public API shape, a format change, a physics
convention) - then say so and stop. A direct "fix it, PR it" from Waine outranks the wait
either way.

## 3. Branch off `main` in a worktree

`main` is the only base. Never `git checkout -b` in Waine's primary checkout - it is meant to
stay clean.

```bash
git fetch origin main
git worktree add ../cfdmod-<slug> -b <type>/<slug> origin/main
```

`<type>` is `fix/`, `feat/`, `refactor/`, `test/`, `docs/` or `chore/`, matching the
conventional-commit prefix the PR will use.

Branch from `origin/main`, not from whatever the primary checkout is on.

## 4. Set the worktree up

A fresh worktree has no `.venv`. `uv` builds one on first use, so there is no install step to
remember - but the **extras** are not optional in practice:

```bash
cd ../cfdmod-<slug>
uv sync --all-extras
```

Plain `uv sync` installs only the dev group, and four test modules import `trimesh` or
`vtkmodules` at module scope, so collection dies before a single test runs. The Dagger CI
syncs all extras, which is why this only ever bites locally. If an extra will not install on
this machine, say which suites you therefore did not run - do not report a clean run.

## 5. Implement

- **A new computation is a core change**: an op under `cfdmod/core/ops/`, then a recipe that
  composes it. A new figure, deliverable or notebook stage is post-processing
  (`cfdmod/building/`, `cfdmod/report.py`, `cfdmod/mesh_field.py`, `examples/`). When a
  notebook needs a primitive that does not exist, that primitive is a core change.
- **Every bug gets a regression test**, and a structural bug gets a structural fix. Tests
  mirror the source layout under `tests/`; fixtures go in `fixtures/tests/`.
- **Errors go through the typed hierarchy** in `cfdmod/core/errors.py`
  (`TemplateError`, `TemplateReferenceError`, `OpError`, `StorageKeyError`), not bare
  `Exception` / `KeyError`. A service consumer catches types, not message strings.
- **ASCII only**, everywhere: source, tests, YAML, notebooks, docs. `->` not an arrow, `^2` not
  a superscript, `u_mean` not a Greek letter. LaTeX is fine inside equations.
- **HARD RULE: no issue or PR references in anything published.** Nothing under `docs/`, no
  `README.md`, no docstring, no notebook prose may contain `#<n>`, a `gh issue` link, or
  "(issue #131)". Issue numbers belong in commit messages, PR bodies and issue comments only.
  Note that existing code violates this in places - do not copy it, and do not go fix it as a
  drive-by either.
- When your change orphans code (a helper whose only caller you replaced), delete it and its
  test in the same commit rather than leaving a second path behind.

## 6. Verify what you touched

Run the gates the change can actually affect, and read the output before claiming anything.

```bash
cd ../cfdmod-<slug>
uv run ruff check cfdmod tests && uv run ruff format --check cfdmod tests
uv run pytest tests/<the mirrors of what you touched> -q
```

Scope the pytest run to the mirrors of the modules you changed plus the ones that exercise the
paths around them (`grep` for the symbol you changed under `tests/`). The full suite is
minutes; a targeted run is seconds and is what makes a review loop viable. Before the merge,
though, run the whole default selection once:

```bash
uv run pytest -q            # -m 'not perf' is the configured default
```

The authoritative gate is Dagger, which runs lint + the pytest suite + the Sphinx docs build in
one container:

```bash
dagger call --source=. check
```

Run it when Docker is available - it is the only thing that covers the **docs build**, so a
change touching `docs/` or any docstring Sphinx renders is not verified without it. When Docker
is not available, say that the docs build was not run.

Two more, only when relevant:

- `uv run pytest -m perf` - the synthetic big-data benchmarks, opt-in. Run them when the change
  claims a performance win, and put the before/after numbers in the PR body.
- `uv run pytest -m property` - the hypothesis suite over the data-source layer, when you
  touched `cfdmod/core/data_source.py` or the ops contract.

**Prove that a red is pre-existing instead of asserting it.** Spin a throwaway worktree at the
base commit and run the same test there:

```bash
git worktree add --detach ../cfdmod-base-check origin/main
```

Remove it when done. If a red only reproduces on your branch, it is yours.

## 7. Review your own diff before anyone sees it

Green tests mean your change does what you told it to. They do not mean it is correct: your
tests were written from the same assumptions as your code, so they share its blind spots. Read
the whole diff back as an adversary before opening the PR.

**Measure the claims, do not read them.** A finding is only real once you have run it. A
throwaway script that prints a number, deleted right after, is the cheapest instrument you
have - and for a numerics library it is usually the *only* honest one:

```python
import time
t0 = time.perf_counter(); run_it(); print("s:", time.perf_counter() - t0)
```
```python
import tracemalloc
tracemalloc.start(); run_it(); print("peak MB:", tracemalloc.get_traced_memory()[1] / 1e6)
```

Run it at two sizes (e.g. 500 and 5000 timesteps). A flat cost is fine; a slope where you did
not expect one is a finding.

**Go looking for these specifically** - they are what slips past a suite that passes:

- **A number that is quietly wrong.** The tests fix a normalisation, a reference area, a lever
  arm, a time scale - and they were written from the same wrong assumption as the code. Check
  the physics against the source (NBR 6123, the recipe docstring) rather than against your own
  test. A wrong coefficient ships as a plausible number in an engineer's deliverable and is
  discovered months later, if ever.
- **Per-timestep or per-triangle work in Python.** A loop over the element axis, or an h5 file
  reopened once per timestep, is invisible on a 200-triangle fixture and dominant on a real
  case. Ask what the loop count is on a 500k-triangle, 10k-timestep case.
- **Memory that scales with the full time axis.** cfdmod's whole point is data that does not
  fit comfortably in RAM. An op that materialises `(n_elements, n_timesteps)` when it could
  stream is a defect even when it passes.
- **A frozen value object mutated in place.** `DataSource` is frozen and its updates re-run the
  validators (`_copy_validated`). Code that reaches around that - mutating a numpy array a
  store handed out, or `model_copy` without re-validation - breaks the invariant the type
  advertises.
- **Optional dependencies imported at module scope.** `trimesh`, `vtkmodules`, `pyvista`,
  `tables` and `fast_simplification` are extras. An import at the top of a module that a base
  install reaches breaks that install. Import inside the function, and guard the test with
  `pytest.importorskip`.
- **Degenerate inputs a validator lets through.** What do `0`, empty, `None` and a
  single-timestep series actually do downstream? A `max(1, x)` clamp can turn "off" into "every
  step".
- **Whatever you copied from.** If your new code is a sibling of existing code, a defect you
  find in yours is almost certainly in the original too.

**Fix what you find, in this branch, with a regression test each.** Do not open a follow-up
issue for a bug you are already looking at. That includes a defect in the *existing* code your
change is a sibling of - it gets its own commit.

**Loop it, but bounded.** Fixing a defect changes the diff, and the new code deserves the same
pass. Stop at whichever comes first: a round that turns up nothing meeting the bar below, or
**three rounds** - past that you are polishing, not finding.

**The bar, and it is the important half of this step.** A finding is worth a round only if it
changes what the engineer, the solver, or the data actually gets:

> a wrong number - silent data loss - memory or time that scales when it should not - a crash
> on input the library will really see - an install or import that breaks.

Everything else is noise, and shipping noise is worse than shipping nothing: it buries the real
findings and burns a review the human has to read. **Not findings:** naming, comment wording,
"could be more elegant", a pattern you would have chosen differently, hypothetical future
requirements, or coverage for a branch that cannot occur. Do not restructure working code in
round 3 to justify the round - a round that finds nothing is a *good* outcome, and saying so in
one line is the correct output.

**Then write the review down** as a PR comment (`gh pr comment <N>`), once the PR exists: what
you found and verified, what you checked and found sound, and what you deliberately did not
check. That comment is what the human reads instead of the diff, so it has to be honest about
its own gaps.

## 8. Commit, push, PR

- Conventional commit subject with the scope, matching the repo's history:
  `fix(building): ...`, `feat(core): ...`, `test(building): ...`, `chore(agents): ...`.
  Put the issue number in the body, not the subject.
- Commit in logical chunks and push each - do not leave commits sitting only locally.
- **No release-notes entry.** `docs/source/release_notes.md` is written at release time by the
  `cfdmod-release` skill, describing behaviour for users. Do not add a section for your PR, and
  never put the issue or PR number in that file.
- Open as a **draft**, with the base set at creation time and a closing keyword in the body:

```bash
gh pr create --draft --base main --head <branch> --title "fix(scope): ..." \
  --body "... closes #<N> ..."
```

The PR body says what changed, why, **which suites you ran and which you did not**, and the
before/after numbers for any performance claim.

## 9. Merge it

You are authorized to merge your own issue-fix PR once the gates you ran are green.

```bash
gh pr view <PR> --json baseRefName,isDraft,mergeable,mergeStateStatus,statusCheckRollup
```

- `baseRefName` must be `main`. `gh pr edit --base` is unreliable; if it is wrong, retarget via
  `gh api -X PATCH repos/AeroSim-CFD/cfdmod/pulls/<PR> -f base=main`, or close and reopen.
- `mergeable: MERGEABLE`. On `CONFLICTING` or `BEHIND`, merge `origin/main` into the branch,
  resolve, re-run the affected gates, push, re-check. `UNKNOWN` just means GitHub has not
  computed it yet - re-run the query.

```bash
gh pr ready <PR>
gh pr merge <PR> --squash --delete-branch
```

**Squash, always.** Every commit on `main` is one PR - `fix(building): ... (#212)`,
`release: v3.7.0 (#216)`. The squash subject becomes that commit, so make the PR title the
commit message you want.

## 10. Confirm it landed

`merged=true` is not proof. Confirm against the branch itself:

```bash
git fetch origin
git log origin/main --oneline -3
gh issue view <N> --json state --jq .state
```

GitHub closes keyworded issues on merge; if the keyword was missing, close it by hand with
`gh issue close <N> -c "closed via #<PR>"`.

## 11. Clean up

```bash
git worktree remove ../cfdmod-<slug>
git worktree list                      # confirm the primary checkout is still clean
```

`--delete-branch` on the merge already removed the remote branch.

Report back: what changed, what you ran and what it said, the PR link, and that the issue is
closed. If a release would be needed for anyone to actually get the fix, say so - and stop
there, because that is Waine's call.

## What is still off limits

Relaxed means relaxed about merging your own finished work, not about these:

- **Never commit or push directly to `main`.** Merging your PR into it is authorized; pushing
  commits onto it is not.
- **Never propose or perform a release.** No version bump in `pyproject.toml`, no tag, no
  GitHub release, no `uv publish`. `uv publish` is irreversible. Releases are Waine's call and
  he raises them; the `cfdmod-release` skill is what runs them when he does.
- **Never merge a PR that is not the one you just built for this issue**, and never merge
  someone else's PR.
- **Never weaken a gate to get a merge through** - skipping a test, loosening an assertion on a
  number you did not verify, or adding an entry to the agent-path allowlist that
  `.claude/check_agent_paths.py` reads, to silence it rather than fixing the path.
- **Never add `pymeshlab` as a dependency.** It is GPL and would force GPL on every downstream
  user; `pyproject.toml` documents this. `fast-simplification` (MIT) is the approved decimator.
- Stop and ask when a conflict resolution is non-obvious, when a gate is red for a reason you
  introduced and cannot fix, or when the fix needs a design decision the issue does not settle.
