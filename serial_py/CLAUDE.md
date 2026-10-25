# serial_py — serial numpy sub-project

`serial_py/` is the project's serial (numpy) MPM: the phase-field and anisotropic phase-field
plates, the Yvonnet plate, and their resume drivers. It is the reference every Warp result in
`../WARP/` is checked against — see `../docs/adr/0001-warp-is-a-gpu-staging-ground.md`.

## Project rules

### Self-contained

- Drivers live at `serial_py/` and are run with `serial_py/` as the working directory.
- Code here imports only from `serial_py/src/` (and third-party packages). Never import from
  `../WARP/`.

### `src/` is an additive backbone

`src/` is the backbone library: reusable routines only. Changes to it must be **purely additive**.
Every existing driver (`pre_noched_plate_PF.py`, `pre_noched_plate_el.py`,
`pre_noched_plate_PF_aniso_layer.py`, `Yvonnet_plate.py`, `Yvonnet_plate_gimp.py`, `resume*.py`,
`_aniso_main_test.py`, …) must keep running unchanged — including drivers written before a new
feature existed.

The reason is reproducibility: old simulation scripts must stay runnable indefinitely, while new
capabilities accumulate in `src/` as options rather than replacements.

When adding a capability, create a new module in `src/` (or new functions in an existing one) and
leave existing signatures and behaviour untouched — `git diff HEAD -- serial_py/src/` should show
no changes to pre-existing functions. Then edit only the specific driver(s) named in the request.

### Drivers hard-code their choices

The scripts *outside* `src/` are the applications, and they pick which backbone routines to call by
hard-coding them. Do not add runtime flags, basis selectors, or config switches so a driver can
choose at run time — switching is done by editing the driver.

## Docs

- `README.md` — what is here and how to run it.
- `documentation/` — LaTeX write-ups of the explicit USL MPM phase-field algorithms.
- Glossary: the root `../CONTEXT.md` (single context — use its terms).
