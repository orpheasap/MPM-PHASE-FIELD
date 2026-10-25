# Parallel conventions for the Warp backbone

How Warp code in `WARP/src/` is written. These are implementation conventions, not domain terms —
domain vocabulary lives in the root `CONTEXT.md`.

## State lives on the device

- Particles are seeded on the host with the numpy backbone (particle mesh, Gauss points) and
  uploaded to `wp.array`s **once**, before the time loop.
- The whole time loop runs on the device. The host does not read particle or grid data inside a
  step.
- The host copies data back (`.numpy()`) **only on output steps**: VTK snapshots, tracked-particle
  samples, energy samples — all on the same `vtk_interval`. Explicit steps between outputs make no
  host round-trips.
- At the end of the run, one final copy feeds the post-processing (metrics, plots).

## Precision

- **float32** throughout: `wp.float32`, `wp.vec2`, `wp.mat22`, `wp.mat33`. Do not mix in
  `float64`; a kernel that needs it is a decision to raise, not to slip in.
- Kernels ported from GeoWarp are retyped from `float64`/`vec2d`/`mat33d`. GeoWarp's `+1e-8`
  cell-index nudges are below float32 resolution and become no-ops; that is accepted.

## Reproducibility

Runs are reproducible **to round-off, not bitwise**. P2G scatters with `wp.atomic_add`, whose order
varies between launches, so float32 results differ in the last bits from run to run. Checks and
tests compare with tolerances, never with equality.

## Kernel shape

- **One thread per particle** for P2G, G2P and constitutive updates; **one thread per node** for
  grid reset and grid updates (momentum update, rigid wall).
- P2G loops over the particle's 3×3 cpGIMP stencil from its lower-left support corner (GeoWarp
  style) and writes nodal quantities with `wp.atomic_add`. G2P loops over the same stencil and only
  reads the grid, so it needs no atomics for grid data; per-step scalar diagnostics (e.g. φ_max,
  viscous dissipation) may still be reduced into a per-step device array with `wp.atomic_max` /
  `wp.atomic_add`.
- Nodes are indexed flat: `ix + iy * (n_grid_x + 1)`.
- The background grid has **square cells** (one `dx`) — the cpGIMP kernel assumes it.
- Shape functions are `@wp.func`s shared by P2G and G2P, so both see the same weights.

## Module layout

- `warp_gimp.py` — cpGIMP weights/gradients and the convected, clamped particle-domain update
  (ported from GeoWarp).
- `warp_hencky.py` — Hencky elastic stress and the elastic material parameters.
- `warp_usl.py` — the USL step kernels and the `run_warp_solver` time loop.
- `warp_phasefield.py` — AT2 `@wp.func`s (g, g′, w, w′) and the bulk-damage kernels
  `grid_phase_project` and `damage_residual`. Each sits behind a device-wide barrier (∇φ needs
  every φ_I, the φ update needs every F_I), so each is its own kernel.
- `warp_small_strain.py` — plane-strain elasticity matrix and `stress_update_pf` (small-strain
  hypoelastic ε, σ₀, degraded σ, history field).
- `warp_usl_pf.py` — the USL step kernels with bulk damage and nominal traction, and the
  `run_warp_pf_solver` time loop.
- `warp_usl_aniso.py` — the two-field (bulk damage d + anisotropic interfacial damage α) USL step
  of the Yvonnet plate, SI units, plane stress: two-field reset, `p2g_aniso`,
  `grid_two_field_project`, `damage_residual_two_field` (one gather of ∇d, ∇α, d̃, α̃; scatters
  F^d and F^α; writes both crack surface energies), `grid_update_aniso`, `g2p_aniso`, the
  total-strain `stress_update_aniso` (σ₀ = C_a(α) : ε recomputed each step), and the
  `run_warp_aniso_solver` time loop.
- `phasefield.py` — host-side phase-field constants, copied from `serial_py/src/phasefield.py`.

New capabilities arrive as new `warp_*.py` modules or new functions (after the freeze, see
`../CLAUDE.md`).
