# 03 — Padded background grid

**What to build:** A driver author can request a background grid that is larger than the plate,
with a ring of empty cells all around it, so that a particle domain near a plate surface always
lands on nodes that exist rather than depositing weight off the end of the grid.

A routine is appended to the existing Yvonnet model module, beside the default parameters and
the model builder, both of which are left untouched. It reads a pad width in cells from the
parameter dict, calls the existing model builder with the plate extent enlarged by that pad on
all four sides, and returns the existing model dict plus the plate origin offset and the plate
cell counts.

The key property to preserve: the existing model builder uses the plate extent *only* to
construct the grid. Every derived quantity — wave speed, elastodynamic limit, both damage
viscosities, both damage stability limits, the resulting time step — depends on cell size,
toughness and length scale, never on extent. Padding must therefore leave all of them
bit-identical, so that a padded run and an unpadded run march in lockstep and any comparison
between them stays honest.

Footgun to make impossible to miss: the cell counts inside the returned model dict are the
**padded** counts. Callers seeding particles, removing the notch or detecting boundary
particles must use the separately returned plate cell counts and plate origin offset. Callers
assembling on the grid must use the padded counts. Make this explicit in the routine's
documentation.

**Blocked by:** None — can start immediately. Independent of 02.

**Status:** ready-for-agent

- [ ] The padded builder returns everything the existing builder returns, plus the plate origin
      offset and the plate cell counts
- [ ] Every derived solver constant is identical to the unpadded builder's for the same
      physical parameters, verified by test
- [ ] Grid cell counts equal plate cell counts plus twice the pad width in each direction,
      verified by test
- [ ] The plate origin offset equals the pad width times the cell size in each direction,
      verified by test
- [ ] A pad width of zero reproduces the unpadded grid exactly, verified by test
- [ ] The routine's documentation states plainly which returned counts are the plate's and
      which are the grid's
- [ ] Neither the default parameters nor the existing model builder is modified
- [ ] Both existing Yvonnet drivers still run unchanged
