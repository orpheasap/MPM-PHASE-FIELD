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

- [ ] The driver runs to completion from a clean checkout and reports a nonzero maximum bulk
      damage and interfacial damage
- [ ] A VTK series and a checkpoint are written to an output directory distinct from the
      hat-basis driver's
- [ ] The checkpoint records the particles-per-cell count, both particle domain half-widths and
      the pad width alongside everything the existing checkpoint records
- [ ] The uGIMP basis and nine-cell connectivity are used in all three loops, with the same node
      set in each
- [ ] The particle domain half-widths are set once at seeding and never updated
- [ ] Plate-local quantities (seeding, notch, Neumann rows) use plate cell counts; grid
      assembly uses padded counts
- [ ] The time step printed by the new driver equals the one printed by the existing Yvonnet
      driver for the same physical parameters
- [ ] All four existing drivers still run unchanged, and the backbone diff shows additions only
- [ ] No runtime flag, basis selector or configuration switch is introduced
