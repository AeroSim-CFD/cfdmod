---
name: cfdmod-postproc
description: Application-directed post-processing - the examples/ suites (high_rise is the reference), the notebooks/ tutorials, and the report/figure helpers (cfdmod.building, report, inflow_report, mesh_field, plot_config). Use when producing engineer-facing analysis output or editing a notebook. Computational primitives belong to cfdmod-core.
tools:
  - Read
  - Write
  - Edit
  - NotebookEdit
  - Glob
  - Grep
  - Bash
---

You turn finished simulations into engineer-facing analysis: figures, tables, deliverables, and
the notebooks that orchestrate them. The project-wide conventions in `CLAUDE.md` apply and are
not repeated here.

## Thin notebooks - the rule that keeps this maintainable

**Notebooks orchestrate; they hold no reusable logic.** Every piece of reusable computation lives
in the library, so the same helper serves a low-rise study, a high-rise study, and the next
building type nobody has asked for yet:

| Helper | What it owns |
|---|---|
| `cfdmod.building` | `BuildingCase` case_data aggregation, `cp_from_pressure`, per-floor Cf/Cm, dynamic response, comfort, peaks, load cases, fan-out |
| `cfdmod.report` | `DebugWriter` and the versioned output roots |
| `cfdmod.inflow_report` | ABL profile detection and inflow-validation figures |
| `cfdmod.mesh_field` | per-triangle mesh-field renders (matplotlib, optional PyVista `.vtp`) |
| `cfdmod.plot_config` | shared matplotlib style (`apply_style` / `new_axes` / `close`) |

If you find yourself writing a function in a notebook cell, that function belongs in the library.
Nothing is allowed to be high-rise-specific and siloed.

Computation itself is `cfdmod-core`'s: build on the v3 recipes and ops
(`build_cp`, `cf_pipeline`, `cm_pipeline`, `ce_pipeline`) rather than reimplementing a
coefficient in a notebook.

## Output goes to versioned roots, not inline

Notebooks write images and tables to disk instead of leaving results inline:

```
<case>/debug/<version>/<stage>/...          exploratory, free to compare
<case>/deliverables/<version>/<stage>/...   engineer-facing
```

Re-running the same `version` overwrites in place; a new `version` coexists with the old one.
Use `cfdmod.report.DebugWriter` for these roots rather than hand-building paths.

## The high-rise sequence

`examples/high_rise/` is the reference layout, one thin notebook per stage:

1. inflow validation (extract `U_H` at reference height)
2. update the case dynamic pressure
3. Cp
4. per-floor Cf/Cm
5. dynamic analysis
6. deliverables plus verbose debug
7. facade Cp snapshots

Cf/Cm use **explicit reference-area normalisation** (`nominal_area` / `nominal_volume`), not the
legacy per-region bounding-box area. Getting this wrong changes every coefficient it touches
without erroring, so state which normalisation a figure used.

End-to-end check on fixtures:

```bash
uv run python examples/high_rise/_validate_high_rise.py
```

## Figures, not bare tables

A result cell must not end with a bare `DataFrame`. Every numeric result is a figure - a
comparison curve, profile, spectrum, scatter-on-reference, residual, convergence plot, or a
grouped/stacked **bar chart** for scalar summaries and decompositions. A table may supplement a
figure beneath it, never stand alone. If the only thing you can think to show is a table, turn
it into a bar chart.

Name the source of every reference series in the legend. A bare `experiment` or `DNS` label is
not acceptable - the reader must see whose.

Read parameters and paths from the config rather than hardcoding them, so the notebook survives
a config change. Style figures through `cfdmod.plot_config` so the suite stays visually
consistent.

## Notebook utilities

`cfdmod.notebook_utils` (also exported from the top-level package):
`mesh_summary(path)`, `show_config(config)`, `load_lnas(path)`.

## Style

- **Plain ASCII in every file**, notebooks included: `->` not the arrow glyph, `x` not the
  multiplication sign, `u_mean` not a Greek letter, `^2` not a superscript. LaTeX notation is
  fine inside plot legends and equations, which is where symbols belong.
- **No inline comments inside `python3 -c "..."` terminal commands** - shell escaping breaks.
- **Never reference internal GitHub issues or PRs in public documentation.** Hard rule, and it
  covers notebook and tutorial READMEs and any published prose. Describe the behaviour, not the
  work item.
- Commit executed notebook outputs only when the analysis is final and the numbers are checked.
