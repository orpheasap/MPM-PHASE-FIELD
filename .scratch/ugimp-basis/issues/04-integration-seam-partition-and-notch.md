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

- [x] The test builds the padded model, seeds plate particles at cell centres, offsets them,
      and removes the four-row notch — matching what the driver in 05 will do
- [x] Weights sum to one within tolerance for every particle, including those in the top and
      bottom rows where the traction is applied
- [x] Gradients sum to the zero vector within tolerance for every particle
- [x] No node has nonzero weight from a particle above the notch and a particle below it
- [x] The test fails, not errors, if the pad width is reduced to zero — demonstrating it
      actually detects the leakage it claims to
- [x] The test fails if the notch is narrowed to two rows — demonstrating it actually detects
      bridging
- [x] Real assertions; fails on its own without a human reading output
- [x] No existing backbone routine or driver is modified

## Comments

- `tests/test_gimp_integration.py`. Pad width 2 and a four-row notch are pinned there as
  `PAD_CELLS` and `NOTCH_ROWS`; the driver in 05 must match.
- Layout: 9800 particles on a 104x104 grid, 2400 particles on each face of the notch.
- **Both failure demonstrations need drift.** At exact cell centres uGIMP collapses onto the
  hat function, whose support is one cell, so the seeded layout sums to one *even with no pad
  at all* — the zero-pad check is therefore run after convecting the plate by 0.45 cells,
  where it leaks 0.19. Likewise the two-row notch shares only a single node at zero drift
  (and that only through floating-point round-off, since the supports touch at exactly zero
  weight); under 0.45 cells of drift it shares 53. The drift margin is what the pad width and
  the notch width actually buy, so that is what the tests assert.
- Partition of unity is asserted at the seeded positions and at +-0.45 cells; residuals are
  ~1e-14 in every case.
- No backbone routine or driver was modified — this ticket added a test file only.
