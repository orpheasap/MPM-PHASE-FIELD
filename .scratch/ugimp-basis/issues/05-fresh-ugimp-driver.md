# 05 — Fresh uGIMP driver

**What to build:** A researcher can run a complete uGIMP simulation of the Yvonnet-style
anisotropic two-field phase-field plate and get a VTK series and a checkpoint out of it, to
open alongside the existing hat-basis results.

A new driver mirrors the existing fresh-run Yvonnet driver in structure, section ordering and
printed output, so the two diff cleanly, but hard-codes its own choices:

- **uGIMP basis and nine-cell connectivity in all three loops** — particle-to-grid (mass,
  momentum, internal force, applied traction, pseudo-mass and both damage projections), damage
  residual assembly (grid gradients of both damage fields), and grid-to-particle (velocity,
  position, velocity gradient and map-back of both damage fields). The same node set must be
  used in all three, or the lumped pseudo-mass will not match the residuals it normalises.
- **One particle per cell**, seeded at cell centres, so the particle domain half-width equals
  the cell size in each direction. Held fixed at its seeded value for the whole run — this is
  uGIMP, not cpGIMP; the particle domain is never convected with the deformation gradient.
- **Padded grid** at the pad width pinned in 04. Particles are seeded over the plate extent
  and translated by the plate origin offset *before* the initial state is captured, so
  reference positions and reference volumes are recorded in grid coordinates. Seeding, notch
  removal and Neumann boundary detection use plate cell counts; grid assembly and the
  neighbour lookup use padded counts.
- **Four-row notch** as pinned in 04.
- **Literal Python loops** in the style of the existing driver, with an early skip on zero
  weight immediately after evaluating the shape function. No vectorisation — a driver that
  does not visually diff against the hat-basis one is much harder to trust.
- Step count and output interval matching the existing driver, so the first comparison is
  like-for-like on everything except basis and notch.
- Its **own output directory and checkpoint**, so that re-running the hat-basis driver — which
  clears its own output directory on startup — cannot destroy these results.
- The existing default parameters copied and extended with the particles-per-cell count, both
  particle domain half-widths and the pad width, then written into the checkpoint, so ticket
  06 can rebuild an identical basis and grid without redefining anything.

No runtime flags, basis selectors or configuration switches anywhere — switching basis is done
by choosing which driver to run.

**Blocked by:** 02, 03, 04.

**Status:** ready-for-agent

- [x] The driver runs to completion from a clean checkout and reports a nonzero maximum bulk
      damage and interfacial damage
- [x] A VTK series and a checkpoint are written to an output directory distinct from the
      hat-basis driver's
- [x] The checkpoint records the particles-per-cell count, both particle domain half-widths and
      the pad width alongside everything the existing checkpoint records
- [x] The uGIMP basis and nine-cell connectivity are used in all three loops, with the same node
      set in each
- [x] The particle domain half-widths are set once at seeding and never updated
- [x] Plate-local quantities (seeding, notch, Neumann rows) use plate cell counts; grid
      assembly uses padded counts
- [x] The time step printed by the new driver equals the one printed by the existing Yvonnet
      driver for the same physical parameters
- [x] All four existing drivers still run unchanged, and the backbone diff shows additions only
- [x] No runtime flag, basis selector or configuration switch is introduced

## Comments

- `Yvonnet_plate_gimp.py` at the repo root, mirroring `Yvonnet_plate.py` section for section;
  `diff Yvonnet_plate.py Yvonnet_plate_gimp.py` is 155 changed lines, all of them basis, grid,
  notch, output directory or comment.
- Particle domain half-widths are `hx/2` and `hy/2` -- the *width* equals the cell size, which is
  what makes the middle branch of the kernel empty and matches `tests/test_gimp_integration.py`.
  The issue body's "half-width equals the cell size" would give a two-cell support instead; see
  the same note on ticket 02.
- Pad width 2 and the four-row notch match `PAD_CELLS` and `NOTCH_ROWS` pinned in 04. Layout comes
  out as the test predicts: 9800 particles on a 104x104 grid, 200 removed for the notch.
- Ran to completion: 100 steps, max bulk d = 0.0013, max interfacial alpha = 0.0007, two VTK
  frames plus `checkpoint.npz` under `vtk_output_Yvonnet_gimp/`.
- `dtime = 3.9917e-09 s`, identical to the hat-basis driver's, as `build_padded_model` guarantees.
- `src/` is untouched by this commit. The four existing drivers were verified by compiling them
  and by the empty backbone diff; they were not executed, since each runs a full simulation.
- `.gitignore` gained `vtk_output_Yvonnet_gimp/`, matching the three existing output directories.
