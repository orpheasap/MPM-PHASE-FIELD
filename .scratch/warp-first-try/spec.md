# Spec: first Warp driver — elastic copper Taylor bar with cpGIMP

Status: ready-for-agent

## Problem Statement

Every MPM driver in this project runs in Python loops over particles on the CPU, which limits the
resolution the anisotropic phase-field plate can reach. The researcher wants to move to GPU-parallel
MPM with NVIDIA Warp, and wants to learn the path on a simple problem first: the copper Taylor bar
already set up in `WARP/CopperTaylorImpact.py`, but run as pure elastic, with GeoWarp's cpGIMP
basis, entirely on the GPU. The result has to be trustworthy enough that the same backbone can later
carry the phase-field model.

## Solution

A new Warp backbone in `WARP/src/warp_*.py` and a new driver `WARP/CopperTaylorImpact_warp.py`,
alongside — not replacing — the numpy backbone and `CopperTaylorImpact.py`, which keep running
unchanged. The driver seeds the bar exactly as the numpy driver does, uploads it to the device
once, runs an explicit USL loop on the GPU, writes a sparse VTK series in the numpy driver's format,
and ends with three physics checks that say whether the run is right.

Rules for the sub-project are in `WARP/CLAUDE.md`, kernel conventions in `WARP/docs/parallel.md`,
the reasoning in `docs/adr/0001-warp-is-a-gpu-staging-ground.md`. Use the glossary terms of the
root `CONTEXT.md`.

## Decisions

**Physics**
- Explicit **USL** with a FLIP/PIC blend `alpha` (0.99, as the numpy driver):
  reset grid → P2G (mass, momentum, internal force) → grid momentum update + rigid wall →
  G2P (velocity blend, `x += dt Σ w v_I`, `L = Σ v_I ⊗ ∇w`, `F ← (I + dt L) F`) →
  convected particle domain update → stress update. No MUSL second mapping. g = 0.
- **Hencky elasticity**, plane strain, `F` stored as a 3×3 matrix with `F33 = 1`. Stress from
  `b = F Fᵀ`: eigen-decompose, `e = ½ ln λ²`, Kirchhoff `τ = λ tr(e) I + 2μ e` in the principal
  frame, Cauchy `σ = τ / J`. Follows GeoWarp's `assemble_residual_2d_hencky`
  (`geowarp_clone/mpm/mpm_utils.py` ~L510–560) with the plastic return removed and `b` taken
  directly from `F` rather than stored. No Johnson-Cook, no EOS, no damage.
- Copper: ρ = 8940, E = 115e9, ν = 0.31. V0 = 190 m/s downward.

**Basis and geometry**
- **cpGIMP** ported from GeoWarp `mpm/gimp.py`: `get_GIMP_shape_function_and_gradient_2d` and
  `update_GIMP_lp_2d`, retyped to float32. **Convected particle domain**: `lp = lp0 ⊙ (column
  norms of F)`, with `lp0` a **per-direction 2-vector** (GeoWarp's is a scalar).
- `lp0 = (hx/2, hy/2)` from the particle mesh element (15 × 50 elements over 7.6 × 25.4 mm,
  one Gauss point each → 750 particles).
- **Cantilever-style layout**: square 1 mm cells; the **rigid wall** is the node row at
  `y = dx`; the row below is an empty pad ring; the bar's bottom sits at `y = dx + 0.1 mm`.
  The particle domain is **clamped** on all four sides at `[dx, L − dx]` exactly as GeoWarp does.
  Grid ≈ 25 × 31 cells.
- **Rigid wall**: frictionless and one-sided — on the wall row, `(m v)_y ← max((m v)_y, 0)` after the
  momentum update. Not a fixed (Dirichlet) boundary.

**Execution**
- float32 throughout. State lives on the device; host copies only on output steps.
- Atomics in P2G; results reproducible to round-off, not bitwise.
- `dt = 0.5 · dx / c_p`, `c_p = sqrt((K + 4G/3)/ρ)` ≈ 4220 m/s → dt ≈ 1.2e-7 s. Run ≈ 40 µs
  (≈ 3 contact times, ≈ 340 steps). `vtk_interval = 10`.

**Output**
- VTK as the numpy driver does it: pyvista `.vtp` per output step + `write_pvd` → `simulation.pvd`,
  in `WARP/vtk_output_warp/`. **Four fields only**: `velocity`, `von_mises` (full 3×3 Cauchy
  stress), `pressure` (−tr σ / 3, including σzz), `J`.
- Tracked particles (top-left, bottom-left) sampled on output steps → `particle_history_warp.csv`.
- Energy history (kinetic, elastic, total) on output steps → `energy_history_warp.csv`.

**Checks** (printed at the end of the driver, pass/fail with the measured value)
1. **Energy**: |E_total(end) − E_total(0)| ≤ 5 % of initial kinetic energy. Elastic energy is
   `Σ V0_p [ μ e:e + ½ λ (tr e)² ]`.
2. **Contact time** ≈ `2L / c_bar`, with the plane-strain bar speed
   `c_bar = sqrt(E / (ρ (1 − ν²)))` ≈ 3770 m/s → ≈ 13.5 µs; pass within 15 %. Contact is measured
   from the wall's reaction (the momentum removed by the wall clamp), accumulated **on the device**
   into a per-step array and read once at the end — no per-step host sync.
3. **Rebound**: mean particle `v_y` after contact ≥ 0.8 · V0 (upward).

Tolerances are first guesses; if a check fails, report the measured values rather than loosening it.

## Out of scope

MUSL, plasticity, Johnson-Cook, Mie-Grüneisen, damage, phase field, float64, a numpy reference run
of the elastic model, performance tuning, scaled-up runs.
