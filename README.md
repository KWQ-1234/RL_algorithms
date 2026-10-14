# RL Algorithms for the Mathematically Inclined

> Which changes to LLM post-training actually matter — and how to tell. A companion to [RL Infrastructure for the Mathematically Inclined](https://github.com/chenggong-zhang/RL_infra).

**Status:** draft. Sections are written in the order given in [`plan/2026-10-14-rl-algorithms-blog-outline.md`](plan/2026-10-14-rl-algorithms-blog-outline.md); unwritten sections render as "Draft pending" stubs.

## Layout

| Path | Contents |
|---|---|
| `_sections/*.html` | One fragment per top-level section, assembled in filename order |
| `templates/index.template.html` | Page shell: styles, left nav, right TOC rail |
| `scripts/build_index.py` | Assembles `index.html` from the template and the fragments |
| `diagrams/` | SVG figures referenced by the sections |
| `plan/` | Chapter outline and writing order |
| `research/evidence-report.md` | Evidence base: per-axis verdicts, A–D grades, contradictions |
| `research/notes/` | The eight research notes behind the report, with reading-depth tags |

## Build

```bash
python3 scripts/build_index.py
```

Open `index.html` in a browser. There is no other build step.

## Evidence conventions

- Verdicts are given on four dimensions: ceiling, efficiency, stability, side effects.
- Grades: **A** multiple independent controlled ablations · **B** one controlled ablation or a clear mathematical mechanism · **C** originating paper only · **D** contradictory evidence.
- A number read only from an abstract or a search snippet is either left out of the text or labelled as such.
