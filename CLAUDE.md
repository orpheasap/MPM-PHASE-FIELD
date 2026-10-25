# MPM_PHASE_FIELD

## Layout

The repo holds two self-contained sub-projects, each with its own backbone (`src/`), drivers,
tests, `CLAUDE.md` and `README.md`:

- **`serial_py/`** — the serial numpy MPM project, and the reference every GPU result is checked
  against. Rules in `serial_py/CLAUDE.md`; read it before touching anything under `serial_py/`.
- **`WARP/`** — the GPU (NVIDIA Warp) staging ground. Rules in `WARP/CLAUDE.md`; read it before
  touching anything under `WARP/`. See `docs/adr/0001-warp-is-a-gpu-staging-ground.md`.

Neither imports from the other: `serial_py/` never imports from `WARP/`, and `WARP/` never imports
from `serial_py/src/` (a helper it needs is copied into `WARP/src/`). Both sub-projects use the
package name `src`, so run each test suite from inside its own directory, never one `pytest` over
the whole repo.

Shared across both, at the repo root: `CONTEXT.md` (the glossary), `docs/` (ADRs and agent docs),
`.scratch/` (issues), `README.md` and `screenshots/`.

## Project rules (both sub-projects)

### Backbones are additive

A sub-project's `src/` is its backbone library: reusable routines only. Once frozen, changes to it
must be **purely additive**, and every existing driver must keep running unchanged — including
drivers written before a new feature existed. The reason is reproducibility: old simulation scripts
must stay runnable indefinitely, while new capabilities accumulate in `src/` as options rather than
replacements. The details (what is frozen, and since when) are in each sub-project's `CLAUDE.md`.

### Drivers hard-code their choices

The scripts *outside* `src/` are the applications, and they pick which backbone routines to call by
hard-coding them. Do not add runtime flags, basis selectors, or config switches so a driver can
choose at run time — switching is done by editing the driver.

### Drivers run from their own directory

Drivers are run with their sub-project directory as the working directory
(`cd serial_py; python pre_noched_plate_PF.py`). Output and checkpoint paths such as
`vtk_output_Yvonnet/checkpoint.npz` are relative to the working directory, so the `resume*.py`
drivers only find their checkpoints when run from there.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` at the root plus `docs/adr/`. See `docs/agents/domain.md`.
