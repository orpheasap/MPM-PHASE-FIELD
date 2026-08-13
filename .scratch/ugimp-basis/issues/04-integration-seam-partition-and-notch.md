# 04 — Integration seam: partition of unity and notch separation

**What to build:** A researcher can trust that the uGIMP basis, the nine-cell connectivity and
the padded grid *compose* correctly over the actual plate layout — not merely that each works
in isolation. Each of the three can be individually correct while the composition still leaks
weight off the grid at the loaded surfaces, or lets a particle support reach across the notch.

One test reconstructs the driver's real setup — padded model at the chosen pad width, plate
particles seeded one per cell at cell centres and translated by the plate origin offset, the
four-row notch removed — and asserts the two properties the whole design turns on:

1. **Partition of unity.** For every particle, the uGIMP weights over its connectivity entry
   sum to one and the gradients sum to the zero vector. This passes only if the pad is wide
   enough *and* the connectivity is correct *and* the kernel is correct. It is what catches
   silent leakage of mass, momentum and applied traction at the top and bottom surfaces, where
   the load is applied.
2. **Notch separation.** No grid node receives nonzero weight from particles on both sides of
   the notch. This is what catches a notch too narrow for the wider uGIMP support, which would
   let the crack faces transmit momentum and diffuse damage across a gap the hat basis
   structurally cannot bridge.

These are the two runtime diagnostics that were deliberately kept out of the drivers. They
belong here, where they fail loudly and cost nothing at simulation time.

This ticket is where the pad width and the notch geometry are pinned. The notch is four
plate-local cell rows straddling the plate midline, giving a void four cells tall centred
exactly on the crack line and spanning the notch length in x. The nearest surviving particles
then sit two and a half cells either side of the crack line, and their supports end one cell
short of each other — two cells of clear separation.

**Blocked by:** 02, 03.

**Status:** ready-for-agent

- [ ] The test builds the padded model, seeds plate particles at cell centres, offsets them,
      and removes the four-row notch — matching what the driver in 05 will do
- [ ] Weights sum to one within tolerance for every particle, including those in the top and
      bottom rows where the traction is applied
- [ ] Gradients sum to the zero vector within tolerance for every particle
- [ ] No node has nonzero weight from a particle above the notch and a particle below it
- [ ] The test fails, not errors, if the pad width is reduced to zero — demonstrating it
      actually detects the leakage it claims to
- [ ] The test fails if the notch is narrowed to two rows — demonstrating it actually detects
      bridging
- [ ] Real assertions; fails on its own without a human reading output
- [ ] No existing backbone routine or driver is modified
