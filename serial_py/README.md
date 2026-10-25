# serial_py — serial numpy MPM

The serial (numpy) implementation of phase-field fracture with the material point method. It is
the reference that every GPU result in `../WARP/` is checked against.

## Drivers

| Driver | What it runs | Output |
|---|---|---|
| `pre_noched_plate_el.py` | Pre-notched plate, linear elastic (no damage). | `vtk_output/` |
| `pre_noched_plate_PF.py` | Pre-notched plate, single-field phase-field fracture. | `vtk_output/` |
| `pre_noched_plate_PF_aniso_layer.py` | Pre-notched plate, layered anisotropic phase field (bulk + interfacial damage). | `vtk_output_aniso/` |
| `Yvonnet_plate.py` | Smaller pre-notched plate, anisotropic two-field phase field with Yvonnet-style toughness parameters. | `vtk_output_Yvonnet/` |
| `Yvonnet_plate_gimp.py` | uGIMP counterpart of `Yvonnet_plate.py`, on a padded background grid. | `vtk_output_Yvonnet_gimp/` |
| `resume.py`, `resume_aniso.py`, `resume_Yvonnet.py`, `resume_Yvonnet_gimp.py` | Continue a run from the `checkpoint.npz` in the matching output directory. | same |
| `plot_energy_budget.py` | Plot the energy budget from a checkpoint (default `vtk_output_Yvonnet/`). | — |
| `_aniso_main_test.py` | Older scratch driver for the anisotropic model. | `vtk_output_aniso/` |
| `test_main.py` | Older scratch script: builds a mesh and particles and plots them. | — |

Run from this directory:

```bash
cd serial_py
python pre_noched_plate_PF.py
python -m pytest tests
```

Output and checkpoint paths are relative to the working directory, so run the `resume*.py`
drivers from here too.

## Layout

- `src/` — the backbone: mesh, particles, basis, quadrature, materials, phase-field model, solver
  utilities, VTK export, plots.
- `tests/` — unit tests for the backbone.
- `documentation/` — LaTeX write-ups of the explicit USL MPM phase-field algorithms.

Rules for working here: `CLAUDE.md`. Glossary: `../CONTEXT.md`.
