# 01 — Elastoplastic Yvonnet plate (J2 + bulk + interfacial damage) on the GPU

Status: done (awaiting review)

**Spec:** the `## Spec` section below holds every physics, kernel, output and check decision.
Read `WARP/CLAUDE.md` and `WARP/docs/parallel.md` first. The theory is §6
("Explicit effective--stress return mapping") and Algorithm 1 of
`WARP/plastic_aniso_layer_PF_MPM_explicit.tex`. That file has already been revised to match
this spec. It is git-ignored and the researcher's own report: read it, never edit it, never
compile it.
Branch: `warp-first-try`. Run in the `pepaenv` conda env from `WARP/`.

**What to build:** A researcher runs `python Yvonnet_plate_plast_warp.py` from `WARP/` and gets
the same pre-notched Yvonnet plate as `Yvonnet_plate_warp.py`: symmetric 2 MPa tension, bulk
damage d and anisotropic interfacial damage α. The new part is J2 plasticity with linear
isotropic hardening, acting on the effective stress. The run writes a sparse ParaView series and
an energy-budget CSV, and ends by printing whether the energy-budget check passes.

Only new files, one appended writer and one doc addition. Every existing `WARP/src/warp_*.py`
module, `Yvonnet_plate_warp.py` and every other driver stay **untouched**.

- `WARP/src/warp_usl_aniso_plast.py` (new). The plane-stress return map `@wp.func`, the
  `stress_update_aniso_plast` kernel, and `run_warp_aniso_plast_solver(...)`. Every other step
  kernel is **imported unchanged** from `src.warp_usl_aniso`: `reset_grid_aniso`, `p2g_aniso`,
  `grid_two_field_project`, `damage_residual_two_field`, `grid_update_aniso` and `g2p_aniso`.
  `update_gimp_lp_2d` is imported from `src.warp_gimp`.
- `WARP/src/vtk_export.py`: **append** a writer `write_particles_vtp_aniso_plast` with the fields
  of `write_particles_vtp_aniso` plus `plastic_strain_eq` (p).
- `WARP/Yvonnet_plate_plast_warp.py` (new): a copy of `Yvonnet_plate_warp.py` with the changes in
  the spec.
- `WARP/tests/test_warp_usl_aniso_plast.py` (new).
- `WARP/docs/parallel.md`: add the new module to the module list (additive).

**Acceptance criteria:**
- [x] `python Yvonnet_plate_plast_warp.py` runs to completion (T = 30 µs) in `pepaenv` on the
      CUDA device. The plots call `plt.show()`, so run headless with `MPLBACKEND=Agg`.
- [x] `WARP/vtk_output_Yvonnet_plast_warp/simulation.pvd` opens with exactly six fields:
      velocity, von_mises, bulk_damage, interfacial_damage, history and plastic_strain_eq.
- [x] `energy_history_Yvonnet_plast_warp.csv` is written with a `W_p` column.
- [x] The energy-budget check prints the measured values and pass/fail. Whether or not it passes,
      report the following in the `## Comments` section of this file: the numbers, nsteps, dt,
      runtime (ms/step), max d, max α, max p, the number of particles that ever yielded, and the
      same max d / max α from a fresh run of `Yvonnet_plate_warp.py`, for comparison.
- [x] Inside the time loop there are no host↔device copies except on output steps. Max d,
      max α, max p, W_ext, D_visc and W_p are accumulated per step on the device.
- [x] `python Yvonnet_plate_warp.py`, `python pre_noched_plate_PF_warp.py` and
      `python CopperTaylorImpact_warp.py` still run.
- [x] `git diff -- WARP/src/` shows only the new module and the appended VTK writer.
- [x] The new pytests and all existing WARP pytests pass (`pytest tests` from `WARP/`).

**Blocked by:** None. Can start immediately.

## Spec

### Problem Statement

`WARP/Yvonnet_plate_warp.py` runs the anisotropic two-field phase-field plate (3D-printed layered
material) on the GPU as a purely elastic body. The source paper (Li, Yvonnet et al., CMAME 386
(2021) 114086) is elastoplastic: J2 plasticity acts on the effective stress and slows fracture,
because only the elastic energy drives damage. The researcher wants the elastoplastic version
of the same run, so the two can be compared directly.

### Solution

A new driver and one new backbone module. The elastic two-field step is reused kernel for kernel.
Only the stress update is new. It now carries a plastic state, the plastic strain εp (Voigt,
engineering shear) and the equivalent plastic strain p, and returns the trial stress onto the
yield surface with a plane-stress closest-point return that uses the full anisotropic C_a(α).

Rules:
- `WARP/CLAUDE.md`: the Warp backbone is frozen, so new modules only; `vtk_export.py` takes
  additive changes only; never import from `serial_py/` or `geowarp_clone/`.
- `WARP/docs/parallel.md`: float32 everywhere; one thread per particle; on-device per-step
  reductions; tolerances in tests, never equality.

Use the glossary terms of the root `CONTEXT.md`: **plastic strain**, **equivalent plastic
strain**, **yield stress**, **hardening modulus**, **plastic work**, **history field**,
**energy budget**, **bulk damage**, **interfacial damage**.

### Decisions

**Everything not listed here is identical to `Yvonnet_plate_warp.py` / `run_warp_aniso_solver`:**
- E = 30e9 Pa, ν = 0.25, ρ = 2450, θ = 60°, ξ = 30, gc_d = 10, gc_α = 1, ℓ_d = ℓ_α = 0.2e-3.
- h = 0.05e-3 m, 200 × 200 material cells, 2-cell pad ring, 1 particle per cell.
- The 4-row notch, σ = 2e6 Pa nominal traction from t = 0.
- The three-way CFL with SF = 0.5, η_α = η_d. Plasticity adds no stability limit, so dt is
  unchanged.
- T = 30e-6 s, vtk_interval = 50.
- cpGIMP with a convected particle domain, lp0 = h/2, pure FLIP.

The elastic driver's header and comments are stale: "20 us", "0.2 mm cells", "50 x 50",
"54 x 54". In the new file, write comments that match the code. Do not touch the old file.

**Plasticity parameters (hard-coded in the driver, new printed section
"Plasticity (J2, linear isotropic hardening)")**
- Yield stress σ_y = 10e6 Pa. This is just below the AT2 strength, about 12.6 MPa for these
  parameters. Print that estimate, sqrt(27 E gc_d / (256 ℓ_d)), next to σ_y.
- Hardening modulus k_h = 1.5e9 Pa (E/20).
- **Naming:** never call the hardening modulus `H`. `H` is the history field, both in the
  glossary and in the solver arrays. Use `k_h` / `hardening_modulus`.

**Constitutive update. This replaces `stress_update_aniso` and follows `.tex` §6 eqs.
(Pvm)–(degrade).** Per particle, Voigt order (11, 22, 12), engineering shear:
1. ε += dt·(L00, L11, L01 + L10) (total strain, as before); C = CA1 + M2 (1 − α)², with the
   new α from G2P.
2. Trial: σ_tr = C (ε − εp); q(σ) = sqrt(1.5 σᵀPσ), with P = (1/3)[[2,−1,0],[−1,2,0],[0,0,6]]
   a constant `wp.mat33`.
3. If q(σ_tr) ≤ σ_y + k_h p, the step is elastic: σ0 = σ_tr, and εp and p are unchanged.
4. Otherwise, solve for Δγ ≥ 0 by Newton, starting from Δγ = 0:
   - A = I + Δγ C P,   σ = A⁻¹ σ_tr   (`wp.inverse` on the mat33)
   - r = q(σ)(1 − ⅔ k_h Δγ) − (σ_y + k_h p)
   - dσ = −A⁻¹ C P σ,   dq = 1.5 (Pσ)·dσ / q,   dr = dq (1 − ⅔ k_h Δγ) − ⅔ k_h q
   - Δγ ← max(0, Δγ − r/dr). Stop when |r| ≤ 1e-5 σ_y, or after a **fixed maximum of 25
     iterations**, keeping the last iterate.

   Then Δεp = Δγ Pσ, Δp = ⅔ Δγ q(σ), εp += Δεp, p += Δp, and σ0 = σ.

   Verified offline in float32 and float64 at α = 0, 0.5, 0.95 and 1: 5–8 iterations, the yield
   error is ≤ 1e-6 σ_y, and C(ε − εp_new) equals σ0 to round-off. The design of the `@wp.func`
   is yours. Warp funcs return one value, so you could return a `wp.vec4` (σ0, Δγ), or inline
   the loop in the kernel.
5. εe = ε − εp (stored, `wp.vec3`); σ = (1 − d)² σ0; H = max(H, ½ σ0·εe); J = det F.
6. If the step yielded: atomic_add V0 · σ·Δεp (degraded Cauchy stress, Voigt dot product) into
   `w_p[step]`, and atomic_max p into `p_max[step]`. p_max is also written for elastic particles
   with p > 0, so do the atomic_max unconditionally.

εp33 = −(εp11 + εp22) is implied and never stored. Don't add μ: the return map uses only C_a(α).

**Kernel sequence per step.** Same order as `run_warp_aniso_solver`, with two differences:
- `damage_residual_two_field` gets the **elastic strain array `eps_e`** where it takes `eps`.
  Its G_α = (1 − α)(1 − d)² ε:M2:ε then uses εe, as the `.tex` requires. This is the start-of-step
  εe, written by the previous step's stress update, just as the elastic solver used the previous
  step's ε.
- The last kernel is `stress_update_aniso_plast` instead of `stress_update_aniso`.

**Energy budget**
- Elastic energy at output = ½ Σ V0 σ·εe (not σ·ε).
- Plastic work W_p is the cumulative sum of `w_p` per step, like D_visc.
- miss = K + U + Γ_d + Γ_α + D_visc + W_p − W_ext. The check is |miss| ≤ 10 % of W_ext at t_f.

**`run_warp_aniso_plast_solver` signature:** the `run_warp_aniso_solver` arguments plus
`sigma_y` and `k_h`. It returns that function's dict keys plus:
- `'plastic_work'`: cumulative, at the output times;
- `'p_max'`: per step;
- `'plastic_strain_eq'`: final p per particle.

`'miss'` includes W_p. εp and p start at zero inside the solver.

**Driver output**
- `vtk_output_Yvonnet_plast_warp/` (deleted at start, as before) and `write_pvd`.
- `energy_history_Yvonnet_plast_warp.csv` with header
  `time_us,K,U,Gamma_d,Gamma_alpha,D_visc,W_p,W_ext,miss,miss_pct_of_w_ext`.
- The energy-budget printout gains a "Plastic work W_p" line. The NOTE lists the plastic
  return's O(dt) work rule alongside the other sources of drift.
- Print max p and the count of particles with p > 0.
- The final figure has three panels: d, α and p (p with its own colour scale, from 0 to max p).

**Tests (`WARP/tests/test_warp_usl_aniso_plast.py`).** Build the layer model the same way as
`tests/test_warp_usl_aniso.py`. Launch the kernel on a handful of particles with prescribed L,
d, α and initial state. Use float32 tolerances.
1. σ_y = 1e30: the new kernel reproduces `stress_update_aniso` (σ, σ0, ε, H, J), and εp = p = 0.
2. Yielding states at α = 0, 0.5 and 1.0, with strains large enough to yield: q(σ0) = σ_y + k_h p
   (relative 1e-4), and σ0 = C_a(α)(ε − εp) (relative 1e-4 of |σ0|).
3. Isotropic check: CA1 = C_iso,plane-stress, M2 = 0, pure-shear strain. Then Δp equals the
   closed form (q_tr − σ_y − k_h p_n)/(3μ + k_h), with μ = E/(2(1+ν)).
4. Elastic step below yield: εp and p are unchanged, and w_p[step] stays 0.
5. Yielding step: w_p[step] > 0.

## Comments

### 2026-10-26 — implementation results (Claude)

**What landed.** `WARP/src/warp_usl_aniso_plast.py` (new: `von_mises_ps` and
`plane_stress_return` `@wp.func`s, `stress_update_aniso_plast`, `run_warp_aniso_plast_solver`),
`write_particles_vtp_aniso_plast` appended to `WARP/src/vtk_export.py`, the new driver
`WARP/Yvonnet_plate_plast_warp.py`, `WARP/tests/test_warp_usl_aniso_plast.py` (5 tests, one per
spec item) and the module entry in `WARP/docs/parallel.md`. `git diff -- WARP/src/` shows only the
appended writer (+26 lines) next to the new untracked module.

The return map `@wp.func` returns a `wp.vec4` (σ0, Δγ). Newton runs in a `while` loop capped at 25
iterations. After each Δγ update it recomputes A⁻¹, σ and r, so the returned σ always belongs to
the returned Δγ, including when the iteration cap is hit.

**Elastoplastic run** (`python Yvonnet_plate_plast_warp.py`, pepaenv, cuda:0, `MPLBACKEND=Agg`):

| quantity | elastoplastic | elastic (`Yvonnet_plate_warp.py`, fresh run) |
|---|---|---|
| nsteps, dt | 15031, 1.9959e-9 s | 15031, 1.9959e-9 s |
| runtime | 30.28 s, 2.014 ms/step (incl. output) | 29.43 s, 1.958 ms/step |
| max d | **0.2682** | **1.0000** |
| max α | 1.0000 | 1.0000 |
| max p | 1.7032e-2 | – |
| particles that ever yielded (p > 0) | 5227 of 39600 (13 %) | – |
| K at t_f (J/m) | 3.680e-1 | 7.547e-1 |
| U (1/2 Σ V0 σ·εe) | 1.176e-2 | 6.781e-3 |
| Γ_d | 1.113e-2 | 7.358e-2 |
| Γ_α | 4.586e-2 | 4.151e-2 |
| D_visc | 5.075e-2 | 6.028e-2 |
| W_p | 1.087e-1 | – |
| W_ext | 6.389e-1 | 9.972e-1 |
| miss | −4.269e-2 (**−6.68 % of W_ext, PASS**) | −6.036e-2 (−6.05 %, PASS) |
| max \|miss\| over t ≥ 10 snapshots | 9.94 % | 15.76 % |

VTK: `vtk_output_Yvonnet_plast_warp/simulation.pvd`, 302 frames, with exactly the fields
`velocity, von_mises, bulk_damage, interfacial_damage, history, plastic_strain_eq` (read back with
pyvista). The CSV header is `time_us,K,U,Gamma_d,Gamma_alpha,D_visc,W_p,W_ext,miss,miss_pct_of_w_ext`.

**Elastic vs plastic.** Plasticity strongly delays bulk fracture. In the elastic run d reaches 1,
the plate splits and its halves are accelerated apart (K = 0.75 J/m, W_ext ≈ 1 J/m). In the
elastoplastic run d stays at 0.27. Yielding caps the effective stress near σ_y + k_h p ≈ 10–36 MPa,
and only the elastic strain feeds H, so the history field stays below what bulk damage needs.
Plastic work (0.109 J/m) is about 1.5× the elastic run's whole bulk crack energy plus its viscous
dissipation. Interfacial damage still reaches α = 1 in both runs, with a similar Γ_α
(0.046 vs 0.042 J/m): the interface (gc_α = 1 N/m) is weak enough to crack even under the capped
stress.

**Things that surprised me.**
- The test data had to be oriented along the layer direction e1′. At α = 1, C_a = CA1 keeps only
  the stiffness along e1′, so an x-stretch at θ = 60° gives about 1 MPa and never yields. The
  yielding tests now load along e1′, plus noise.
- Plastic work closes the budget better than the elastic case does over the run (max |miss|
  9.9 % vs 15.8 %), but the end-of-run miss is about the same (−6.7 % vs −6.1 %).
- Regression runs: `Yvonnet_plate_warp.py` and `CopperTaylorImpact_warp.py` still pass all their
  checks. `pre_noched_plate_PF_warp.py` runs to completion. Its crack-path band check still FAILs
  with the same numbers as before (290 particles, 68 at ≥ 1 mm, x_tip = 72.375 mm), as recorded in
  `.scratch/warp-phase-field-plate/issues/01-phase-field-plate-warp.md`; that failure predates this
  work.
- `pytest tests` from `WARP/`: 32 passed (27 existing + 5 new).

