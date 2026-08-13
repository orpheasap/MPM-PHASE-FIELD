# MPM_PHASE_FIELD

## Project rules

### `src/` is an additive backbone

`src/` is the backbone library: reusable routines only. Changes to it must be **purely additive**.
Every existing driver at the repo root (`pre_noched_plate_PF.py`, `pre_noched_plate_el.py`,
`pre_noched_plate_PF_aniso_layer.py`, `Yvonnet_plate.py`, `resume*.py`, `_aniso_main_test.py`, …)
must keep running unchanged — including drivers written before a new feature existed.

The reason is reproducibility: old simulation scripts must stay runnable indefinitely, while new
capabilities accumulate in `src/` as options rather than replacements.

When adding a capability, create a new module in `src/` (or new functions in an existing one) and
leave existing signatures and behaviour untouched — `git diff HEAD -- src/` should show no changes
to pre-existing functions. Then edit only the specific driver(s) named in the request.

### Drivers hard-code their choices

The scripts *outside* `src/` are the applications, and they pick which backbone routines to call by
hard-coding them. Do not add runtime flags, basis selectors, or config switches so a driver can
choose at run time — switching is done by editing the driver.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` at the root plus `docs/adr/`. See `docs/agents/domain.md`.
