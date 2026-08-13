# 02 — uGIMP kernel and nine-cell connectivity

**What to build:** Given a background grid and a particle position, a driver author can obtain
the set of nodes that particle's uGIMP support reaches, and the correct interpolation weight
and gradient at each of them. Today only the linear hat basis is available, whose support is a
single cell; this adds the wider uGIMP support beside it.

Two routines are appended to the existing solver-utilities backbone module, beside the hat
shape function and the particle–element map:

- A **uGIMP shape function** taking a particle-to-node offset, the two cell sizes and the two
  particle domain half-widths, returning a weight and a length-2 gradient. The
  one-dimensional four-branch GIMP logic is inlined for both directions inside this single
  routine — no separate one-dimensional helper — mirroring how the existing hat routine is
  written. Argument order and return shape deliberately mirror the hat routine so that a
  driver using one diffs cleanly against a driver using the other. Outside the support it
  returns a zero weight and a zero gradient, so callers can skip non-contributing nodes with
  a simple test.
- A **nine-cell connectivity builder** taking a background grid and returning one entry per
  cell: the sorted unique node set of that cell together with its up-to-eight neighbours.
  Entries are ragged — fewer cells contribute at the grid border — and the routine never
  indexes past the end of the grid. It is built once and does not change, since the grid is
  Eulerian. A module-private neighbour lookup backs it; write it in zero-based form as a
  rewrite, not a transliteration of the one-based reference implementation.

Note the expected degeneracy: when the particle domain half-width equals the cell size, the
middle branch of the kernel is empty and the support half-width is one and a half cells. This
is correct and is locked in by a test below.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The uGIMP shape function returns the expected weight and gradient in each of the four
      branches, verified by test
- [ ] The shape function returns exactly zero weight and a zero gradient outside the support,
      verified by test
- [ ] The shape function reproduces the linear hat function at a cell centre when the particle
      domain equals the cell size, verified by test against the existing hat routine
- [ ] The neighbour lookup returns 4 cells at a grid corner, 6 along an edge and 9 in the
      interior, verified by test
- [ ] The connectivity builder returns one entry per cell, each with no repeated nodes and no
      out-of-range node index, verified by test at the border and in the interior
- [ ] Weights sum to one and gradients sum to the zero vector for a particle in the grid
      interior, verified by test
- [ ] Tests fail loudly on their own — real assertions, not printed comparisons. Follow the
      existing test module's directory, import idiom and main guard, but not its print-only
      style
- [ ] No existing backbone routine is modified; the backbone diff shows additions only
- [ ] All four existing drivers still import and run unchanged
