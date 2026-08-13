# Spec: uGIMP basis for the Yvonnet two-field phase-field MPM plate

Status: ready-for-agent

## Problem Statement

The MPM drivers in this repo all interpolate between particles and the background grid
with linear hat functions, whose support is a single background cell. A particle that
convects across a cell boundary sees its nodal weights change with a discontinuous
gradient, which injects spurious noise into the internal force, the strain rate, and —
in the phase-field drivers — into the damage residuals that ride on the same shape
functions. In a fracture simulation this noise is indistinguishable from real damage
nucleation, so the researcher cannot tell whether a crack path is physics or an
artefact of the basis.

The researcher wants to run the Yvonnet-style anisotropic two-field phase-field plate
with a uGIMP (undeformed Generalized Interpolation Material Point) basis instead, whose
particle domains overlap neighbouring cells and whose shape functions are C1 across cell
boundaries, and compare the result against the existing hat-function run. Every existing
driver must keep running exactly as before.

## Solution

Add uGIMP as a *parallel* capability rather than a replacement, following the repo's
additive-backbone rule:

- Two new routines are appended to the existing backbone modules — a uGIMP shape
  function and a nine-cell grid connectivity builder — leaving every existing routine
  untouched, so all four existing drivers are unaffected.
- One new routine builds a *padded* background grid: the grid is enlarged by a ring of
  empty cells and the plate is seeded inset from the grid edge, so that uGIMP particle
  domains near the loaded surfaces never overhang the end of the grid.
- Two new drivers, `Yvonnet_plate_gimp` and `resume_Yvonnet_gimp`, mirror the existing
  `Yvonnet_plate` / `resume_Yvonnet` pair line-for-line but hard-code the uGIMP basis,
  the padded grid, and a widened pre-notch. They write to their own output directory
  and their own checkpoint, with no interoperability with the hat-basis checkpoint.

The researcher runs the new driver, resumes it in batches exactly as today, and opens
the resulting VTK series alongside the hat-basis series.

## User Stories

1. As a researcher, I want a uGIMP shape function available in the backbone, so that I
   can build drivers that do not suffer cell-crossing noise.
2. As a researcher, I want the uGIMP shape function to take the same argument order and
   return the same `(N, dNdx)` pair as the existing hat routine, so that a driver using
   it diffs cleanly against the equivalent hat-basis driver.
3. As a researcher, I want the one-dimensional GIMP branch logic inlined for both
   directions inside a single routine, so that it reads the same way as the existing hat
   routine rather than introducing a separate one-dimensional helper.
4. As a researcher, I want the shape function to return zero weight and zero gradient
   outside the particle's support, so that callers can skip non-contributing nodes with
   a simple test.
5. As a researcher, I want a nine-cell connectivity map built once from the background
   grid, so that each particle can be scattered to every node its wider support reaches.
6. As a researcher, I want that connectivity map to hold fewer cells at the grid border,
   so that the routine never indexes past the end of the grid.
7. As a researcher, I want the connectivity map built once before the time loop, so that
   no work is repeated on a grid that never moves.
8. As a researcher, I want particle domain size held fixed at its seeded value, so that
   the driver implements uGIMP and not cpGIMP.
9. As a researcher, I want one particle per background cell as today, so that the new
   driver's particle count and cost stay comparable to the existing one.
10. As a researcher, I want the particle domain size derived from the cell size and the
    particle-per-cell count, so that changing the seeding density automatically changes
    the basis support.
11. As a researcher, I want the background grid enlarged by a ring of empty cells, so
    that particle domains near the plate surface always land on nodes that exist.
12. As a researcher, I want the plate seeded inset from the grid edge by that same ring,
    so that the padding sits outside the material on all four sides.
13. As a researcher, I want the padding to leave the material model and every stability
    limit unchanged, so that the padded run and the unpadded run take identical time
    steps and the comparison stays honest.
14. As a researcher, I want the padded-grid construction shared between the fresh-run
    driver and the resume driver, so that a resumed run can never reconstruct a
    different grid than the one its particles were computed on.
15. As a researcher, I want the padded-grid routine to report both the plate cell counts
    and the grid cell counts, so that a driver can seed particles and apply boundary
    conditions in plate coordinates while assembling on the grid.
16. As a researcher, I want the pre-notch widened so that particle supports on opposite
    crack faces cannot overlap, so that the wider uGIMP support does not silently heal
    the crack once particles convect.
17. As a researcher, I want the widened notch centred exactly on the plate midline, so
    that a symmetric-tension problem stays symmetric and any asymmetry in the resulting
    crack path is attributable to the physics rather than to the notch.
18. As a researcher, I want the uGIMP basis used in the particle-to-grid pass, so that
    mass, momentum, internal force and the applied traction all scatter consistently.
19. As a researcher, I want the uGIMP basis used when assembling the bulk and
    interfacial damage residuals, so that the residuals are consistent with the
    pseudo-mass that divides them.
20. As a researcher, I want the uGIMP basis used in the grid-to-particle pass, so that
    velocity, position, velocity gradient and both damage fields map back on the same
    basis they were built from.
21. As a researcher, I want the same node set used in all three passes, so that the
    lumped pseudo-mass matches the residuals it normalises.
22. As a researcher, I want zero-weight nodes skipped inside the loops, so that the
    wider stencil costs arithmetic only where it contributes.
23. As a researcher, I want the driver written as literal loops in the style of the
    existing driver rather than vectorised, so that I can verify it line-by-line against
    the hat-basis version.
24. As a researcher, I want the new driver's step count and output interval to match the
    existing driver's, so that the first comparison is like-for-like.
25. As a researcher, I want the new driver to write to its own output directory, so that
    re-running the hat-basis driver cannot delete my uGIMP results.
26. As a researcher, I want the new driver to write its own checkpoint into that
    directory, so that the resume workflow I already use works unchanged.
27. As a researcher, I want the particle-per-cell count, both particle domain sizes and
    the pad width recorded in the saved parameters, so that the resume driver rebuilds
    an identical basis and grid without redefining anything.
28. As a researcher, I want the resume driver to refuse to start when its own checkpoint
    is absent, so that I get a clear error instead of a silently wrong run.
29. As a researcher, I want the resume driver to refuse the hat-basis checkpoint, so
    that no run can change basis midway through and become impossible to explain later.
30. As a researcher, I want the resume driver to keep the existing batch-and-prompt
    workflow, so that I can keep extending a run in increments without relearning
    anything.
31. As a researcher, I want the resume driver to keep appending to the same VTK series
    and history arrays, so that a long run reads as one continuous simulation.
32. As a researcher, I want the existing four drivers to run byte-identically after this
    change, so that every simulation script I have already published stays reproducible.
33. As a researcher, I want no runtime flag or basis selector added anywhere, so that the
    repo's convention of hard-coding choices in drivers is preserved.
34. As an AFK agent, I want a test that fails loudly rather than printing results for a
    human to read, so that I can verify my own work.
35. As an AFK agent, I want a single test that composes the padded grid, the connectivity
    map and the shape function over the driver's real particle layout, so that a
    regression in the interaction between them is caught even when each part is
    individually correct.
36. As a researcher, I want that test to assert partition of unity over every particle,
    so that any leakage of mass, momentum or traction off the end of the grid is caught.
37. As a researcher, I want that test to assert that shape function gradients sum to
    zero over every particle, so that a rigid translation produces no spurious internal
    force.
38. As a researcher, I want that test to assert that no node is reachable from particles
    on both sides of the notch, so that a notch too narrow for the basis is caught before
    it silently heals a crack.
39. As a researcher, I want narrow tests of each branch of the shape function, so that a
    typo in one branch is localised rather than showing up only as a failed integration
    assertion.
40. As a researcher, I want a test asserting the shape function reproduces the hat
    function at cell centres for the seeded particle domain size, so that the known
    equivalence of the two bases in the undeformed configuration is locked in as a
    regression.
41. As a researcher, I want a test asserting the connectivity map returns the right cell
    counts at a corner, an edge and the interior, so that the zero-based port of the
    neighbour lookup is verified at every boundary case.
42. As a researcher, I want a project glossary, so that terms like particle domain,
    support, pseudo-mass and the two damage fields have one agreed meaning across the
    codebase.

## Implementation Decisions

### Basis

- **uGIMP, fixed particle domain.** The particle domain half-width is set once at
  seeding from the cell size and the particles-per-cell count and never convected with
  the deformation gradient. cpGIMP is explicitly not implemented.
- **One particle per cell**, seeded at cell centres, as in the existing driver. The
  particle domain therefore equals the cell size in each direction. A consequence worth
  recording: with a particle domain equal to the cell size, the middle branch of the
  GIMP kernel is empty and the support half-width is one and a half cells; at a particle
  sitting exactly at a cell centre the resulting weights are *identical* to the hat
  function, and the two bases diverge only once particles convect off centre. This is
  correct, not a defect, and is locked in by a test.

### Backbone routines (all additive; no existing routine is modified)

Appended to the `src.solver_utilities` module, alongside the existing hat routine and
particle–element map:

- `get_gimp2d_shape(x, deltax, deltay, lpx, lpy) -> (N, dNdx)` — the four-branch uGIMP
  kernel, with the one-dimensional logic inlined for both directions in a single
  routine, deliberately mirroring the existing hat routine's signature and return shape
  (`x` first, `dNdx` a length-2 array, `0.0` and a zero vector returned outside support).
- `build_gimp_element_map(mesh) -> list[np.ndarray]` — one entry per background cell,
  each the sorted unique node set of that cell and its up-to-eight neighbours; ragged at
  the grid border. A module-private neighbour lookup backs it, rewritten in zero-based
  form rather than transliterated from the one-based reference implementation.

Appended to the `src.yvonnet_model` module, alongside the existing default parameters
and model builder, both of which are left untouched:

- `build_padded_model(params) -> dict` — reads a pad width in cells from the parameter
  dict, calls the existing model builder with the plate extent enlarged by that pad on
  all four sides, and returns the existing model dict plus the plate origin offset and
  the plate cell counts.

### Padded grid

- The pad is **2 cells**. One cell is the minimum that keeps a surface particle's
  support on the grid in the undeformed configuration; the second is margin for outward
  convection under the applied traction.
- The existing model builder uses the plate extent *only* to construct the grid; every
  derived quantity — the wave speed, the elastodynamic limit, both damage viscosities
  and both damage stability limits — depends on cell size, toughness and length scale
  and not on extent. Padding therefore leaves the time step bit-identical.
- **Footgun to respect:** the cell counts inside the returned model dict are the
  *padded* counts. Particle seeding, notch removal and Neumann boundary detection must
  use the separately returned plate cell counts and the plate origin offset; grid
  assembly and the neighbour lookup must use the padded counts.
- Particles are seeded over the plate extent and then translated by the plate origin
  offset before the initial state is captured, so that reference positions and reference
  volumes are recorded in grid coordinates.
- The existing particle–element map is reused unchanged: it floors position by cell size
  and clamps, which is correct on the padded grid for offset particles.

### Notch

- **Four plate-local cell rows** are removed, straddling the plate midline, giving a void
  four cells tall centred exactly on the crack line, spanning the existing notch length
  in the x direction. The nearest surviving particles sit two and a half cells either
  side of the crack line; their supports end one cell short of each other, leaving two
  cells of clear separation.
- This deliberately differs from the hat-basis driver's two-row notch. The consequence is
  recorded under Further Notes.

### Drivers

- `Yvonnet_plate_gimp` and `resume_Yvonnet_gimp`, mirroring `Yvonnet_plate` and
  `resume_Yvonnet` in structure, section ordering and printed output, so the pairs diff
  cleanly.
- The fresh driver copies the existing default parameters and adds the particles-per-cell
  count, both particle domain half-widths, and the pad width, then calls the padded model
  builder. The extended dict is written into the checkpoint exactly as today; the
  existing model builder ignores the extra keys, so nothing else needs to change.
- The uGIMP basis and the nine-cell connectivity replace the hat basis and the four-node
  cell connectivity in **all three** loops — particle-to-grid (mass, momentum, internal
  force, applied traction, pseudo-mass, and both damage projections), damage residual
  assembly (grid gradients of both damage fields), and grid-to-particle (velocity,
  position, velocity gradient, and map-back of both damage fields).
- Loops stay literal Python, with an early skip on zero weight immediately after
  evaluating the shape function. No vectorisation.
- Step count and output interval match the existing driver.
- Output directory and checkpoint are separate from the hat-basis run's. The resume
  driver reads only the uGIMP checkpoint and hard-errors when it is absent; it never
  accepts the hat-basis checkpoint.
- The batch-and-prompt resume workflow, the checkpoint save helper, and the VTK series
  and history array appending are carried over unchanged in behaviour.
- No runtime flags, basis selectors or config switches are added anywhere.

### Documentation

- A project glossary is created at the repo root, covering: background grid, particle
  mesh, particle domain, support, connectivity, pseudo-mass, bulk damage, interfacial
  damage, pad ring, and notch. Glossary only — no implementation detail, no spec content.
- No architecture decision record. Adding uGIMP as a parallel driver is neither hard to
  reverse nor surprising given the additive-backbone and hard-coded-driver conventions
  already documented in the repo instructions.

## Testing Decisions

**What makes a good test here.** Tests assert external behaviour — the properties a
correct basis must have — never internal structure. They must fail on their own rather
than printing numbers for a human to interpret.

**Prior art.** The repo's only existing test module sits in the tests directory,
inserts the repo root on the import path before importing from the backbone, groups
checks into plain functions, and runs them from a main guard. That layout, import idiom
and runner are kept. The prior art is, however, **print-only**: it computes closeness
checks and prints the boolean result without asserting, so it can never fail. The new
tests deviate on exactly this point and use real assertions. The existing test module is
left untouched.

**Primary seam — one integration test.** The two new drivers are scripts, not importable
modules; importing either runs a full simulation and writes output, so they are not a
usable seam. The highest usable seam is a single test that reconstructs the driver's real
setup — padded model with the driver's pad width, plate particles seeded at cell centres
and offset, four-row notch removed — and then asserts the two properties the whole design
turns on:

1. **Partition of unity.** For every particle, the weights over its connectivity entry
   sum to one, and the gradients sum to the zero vector. This passes only if the pad is
   wide enough *and* the connectivity is correct *and* the kernel is correct, so it
   catches leakage at the loaded surfaces that per-function unit tests would miss.
2. **Notch separation.** No grid node receives nonzero weight from particles on both
   sides of the notch.

These are the two runtime diagnostics that were considered for the driver and
deliberately dropped from it; they belong in a test, where they fail loudly and cost
nothing at simulation time.

**Secondary — narrow kernel checks.** Localising checks so a typo is pinpointed rather
than surfacing only as a failed integration assertion:

- Each branch of the uGIMP kernel returns the expected weight and gradient, including
  exact zero outside support.
- The kernel reproduces the hat function at cell centres when the particle domain equals
  the cell size — the known equivalence in the undeformed configuration, locked in as a
  regression.
- The neighbour lookup returns four cells at a grid corner, six along an edge and nine in
  the interior, verifying the zero-based port at every boundary case.

**Not covered by automated tests.** The two drivers themselves. They are verified by
running them and inspecting the VTK output against the hat-basis run.

## Out of Scope

- cpGIMP, or any convection of the particle domain with the deformation gradient.
- More than one particle per cell. The particle domain size is derived from the
  particles-per-cell count so this can be changed later by editing the driver, but this
  work does not change it or validate it.
- Vectorising any MPM loop, in the new drivers or the existing ones.
- Any modification to an existing backbone routine, or to any of the four existing
  drivers. The diff over the backbone must show additions only.
- A uGIMP variant of any driver other than the Yvonnet pair.
- A hat-basis driver with a four-row notch, which is what a clean basis-only comparison
  would require — see Further Notes.
- Runtime diagnostics for boundary leakage or notch bridging inside the drivers.
- Any runtime flag, basis selector or configuration switch.
- An architecture decision record.

## Further Notes

**The comparison against the hat-basis run is not like-for-like.** The existing driver
has a two-row notch and the new one has a four-row notch, so a difference in crack
initiation time or path between the two runs is attributable to either the basis or the
notch, and the two cannot be separated. This was an accepted trade: the narrower notch is
bridgeable by a uGIMP support once particles convect, which would let the crack faces
transmit momentum and diffuse damage across a notch the hat basis structurally cannot
bridge. If a clean basis-only comparison is wanted later, the cheaper route is a four-row
variant of the hat-basis driver, not a two-row uGIMP one.

**Boundary leakage returns silently if particles drift far enough.** With a two-cell pad
and no runtime diagnostic, a surface particle that convects more than one cell beyond the
plate edge will again deposit weight onto nodes that do not exist, and nothing will say
so. At the driver's default step count this cannot happen; across many resume batches it
eventually could. The integration test asserts the property in the undeformed
configuration only. Raising the pad width is a one-line driver edit.

**uGIMP and the hat basis are identical at step zero** for this particle layout, as noted
under Implementation Decisions. A first comparison that shows no difference in the
earliest frames is expected behaviour, not a failed integration.
