# Spec: Warp pre-notched plate with phase-field fracture (cpGIMP)

Status: ready-for-agent

## Problem Statement

The phase-field plate `pre_noched_plate_PF.py` runs as Python loops over 63–64 k particles on the
CPU, which makes the default run 4 steps long and a full crack propagation out of reach. The first Warp
driver (`WARP/CopperTaylorImpact_warp.py`) proved the GPU path on an elastic bar. The researcher
now wants the isotropic AT2 phase-field plate on the GPU, built from the same parallel pattern
(GeoWarp style) and the cpGIMP basis, as the stepping stone to the anisotropic plate and, later,
plasticity.

## Solution

A new driver `WARP/pre_noched_plate_PF_warp.py` and new Warp backbone modules, alongside the
existing ones, which are not modified. The driver seeds the plate as the numpy driver does (with
the changes listed under geometry), uploads it once, runs an explicit USL step with an explicit
bulk-damage update entirely on the device, writes a sparse VTK series plus two CSV histories, and
ends with two checks.

Rules: `WARP/CLAUDE.md` (the Warp backbone is **frozen**, so add new modules and never edit
`warp_gimp.py`, `warp_hencky.py` or `warp_usl.py`), kernel conventions in `WARP/docs/parallel.md`,
reasoning in `docs/adr/0001-warp-is-a-gpu-staging-ground.md`. Use the glossary terms of the root
`CONTEXT.md`. In particular, the field is **bulk damage** (φ), and H is the **history field**.

## Decisions

**Physics (port of `pre_noched_plate_PF.py`, Algorithm 1)**
- Concrete, plane strain: E = 32e3 MPa, ν = 0.2, ρ = 2.45e-9 t/mm³ (mm–N–s–tonne units).
- AT2: ℓ = 0.25 mm, Gc = 3e-3 N/mm, η = 1e-9 MPa·s, κ = 1e-7; g = (1−φ)², w = φ², c_w = 0.5,
  crack coefficient c = Gc / (2 c_w ℓ). Host constants come from a copy of the root
  `src/phasefield.py` placed at `WARP/src/phasefield.py`.
- **Small-strain hypoelastic**, as the reference: dε = sym(L)·dt (Voigt, engineering shear),
  ε += dε, σ₀ += C:dε, σ = (g(φ) + κ)·σ₀, V = det F · V₀. History field
  H = max(H, ½ σ₀:ε) with **no tension/compression split**. Not Hencky.
  - Plasticity comes later. Keep the constitutive step in its own kernel so that a plastic
    version can replace it in a new driver. Note: once plasticity exists, ψ₀ must come from the
    *elastic* strain. Do not build that now.
- USL, **pure FLIP (α = 1.0)**, as the reference. g = 0, no Dirichlet BCs.
- **Load**: constant σ = 1 MPa from t = 0, +y on the top material cell row, −y on the bottom
  row. GeoWarp-style nominal traction: a per-particle `wp.vec2` array `t_p` = (0, ±σ) on loaded
  particles and (0, 0) elsewhere, deposited in P2G as f_I += w_I · t_p · h (fixed segment length,
  1 particle per cell). See GeoWarp `mpm_utils.py` ~L971–994.

**Step: kernel sequence (one thread per particle / per node, as in `parallel.md`)**
1. `reset_grid_pf` (nodes): zero m, mv, f_int+f_ext, m^φ, φ-numerator, F_I.
2. `p2g_pf` (particles, 3×3 cpGIMP stencil, `atomic_add`): mass, momentum, internal force from
   the **degraded** stress, traction force, m^φ += w·V₀, φ-num += w·V₀·φ_p.
3. `grid_phase_project` (nodes): φ_I = φ-num / m^φ where m^φ > tol.
4. `damage_residual` (particles): gather ∇φ_p = Σ ∇w_I φ_I over the 9 nodes, then `atomic_add`
   F_I += [g′(φ_p)·w_I·H_p + c·(½ w′(φ_p)·w_I + ℓ² ∇w_I·∇φ_p)]·V₀. Weights are recomputed with the same
   `@wp.func` (not cached).
5. `grid_update_pf` (nodes): mv += f·dt and φ_I ← φ_I − dt/(η m^φ)·F_I, in one kernel.
6. `g2p_pf` (particles, reads only): FLIP velocity, x += dt Σ w v_I, L, F ← (I + dt L)F,
   φ_p = max(φ_p, clamp(Σ w φ_I, 0, 1)) for irreversibility, then the convected particle domain update.
7. `stress_update_pf` (particles): ε, σ₀, σ, H as above.

Steps 3 and 5 exist because ∇φ needs the finished φ_I and the φ update needs the finished F_I,
and each of those is a device-wide barrier, which means a kernel boundary. Rejected alternatives: lagging the residual on the previous
step's grid φ (changes Algorithm 1), and node-centric gathers (need sorting, not GeoWarp style).

**Basis and geometry**
- cpGIMP from `WARP/src/warp_gimp.py` (import its `@wp.func`s and `update_gimp_lp_2d`; do not
  edit them). **Convected particle domain**, lp0 = (h/2, h/2). GeoWarp clamp at [dx, L − dx].
- Square cells h = 0.25 mm. Material 100 × 40 mm, 1 particle per cell (400 × 160 particle mesh,
  one Gauss point), seeded exactly like the reference.
- **Pad ring of 2 cells on all four sides**: background grid 404 × 164 cells, with the material
  offset by (2h, 2h).
- **Four-row notch**: remove the material rows iy ∈ {78, 79, 80, 81} (y ∈ [19.5, 20.5) mm in
  material coordinates) for x ≤ 50 mm. With two rows, the convected cpGIMP support bridges the
  faces near the tip under y-tension. This follows the precedent of `Yvonnet_plate_gimp.py`.
  Expect 64000 − 800 = 63200 particles.

**Time**
- dt = 0.5 · min(h / c, dt_PF), with c = sqrt(E/ρ) and dt_PF = `dt_crit_phase(h, 2)` as the
  reference (≈ 3.5e-8 s). T = 60 µs (≈ 1700 steps). `vtk_interval = 50`.
- Every choice is hard-coded. **No environment-variable overrides** (the reference's
  `PF_H_SCALE`, `PF_NSTEPS` and `PF_SIGMA` are dropped).

**Execution**
- float32 throughout. State lives on the device. The host copies only on output steps, plus once
  at the end. Atomics in P2G and the residual, reproducible to round-off.
- Per-step histories accumulated **on the device** into per-step arrays and read once at the
  end: φ_max (`wp.atomic_max`, the reference's `da`), the external work increment, and the
  viscous dissipation increment.

**Output**
- VTK: pyvista `.vtp` per output step + `write_pvd` → `WARP/vtk_output_PF_warp/simulation.pvd`.
  **Four fields**: `velocity`, `von_mises`, `damage` (φ), `history` (H). Append a new writer to
  `WARP/src/vtk_export.py` and leave the existing writers untouched.
- `energy_history_PF_warp.csv` (output steps): t, K, U, Γ_bulk, D_visc, W_ext, miss.
- `crack_tip_PF_warp.csv` (output steps): t, x_tip, φ_max. x_tip = max material-x over
  particles with φ > 0.9 within 4ℓ of the notch line, computed in numpy from the host copy that
  the output step already makes.
- **No checkpoint yet.** The reference's `checkpoint.npz` and resume belong to a later ticket.

**Energy budget** (follow `Yvonnet_plate.py` ~L45–70 and ~L393–481, single field)
- K = Σ ½ m v²; U = Σ ½ V₀ σ:ε (degraded); Γ_bulk = Σ V₀ · Gc/(4 c_w ℓ) · (w(φ_p) + ℓ²|∇φ_p|²);
  D_visc = Σ_t Σ_p V₀ η (Δφ_p)² / dt; W_ext = Σ_t dt Σ_I f_I^ext · v_I (post-update velocity).
- miss = (K + U + Γ_bulk + D_visc) − W_ext, reported as a % of W_ext.

**Checks** (printed at the end, pass/fail with measured values)
1. **Energy budget**: |miss| ≤ 10 % of W_ext at the end of the run.
2. **Crack path**: every particle with φ > 0.9 lies within |y − y_notch| < 4ℓ, and the final
   x_tip > 50 mm (the crack has grown from the notch tip).

Tolerances are first guesses. If a check fails, report the measured values and do not loosen the
tolerance.

## Out of scope

Plasticity, a tension/compression split, anisotropy and interfacial damage, checkpoint and resume,
float64, a numpy reference run of the cpGIMP model, performance tuning, editing any existing
`WARP/src/warp_*.py` module.
