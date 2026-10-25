# 01 — Yvonnet plate (bulk + interfacial damage) on the GPU with cpGIMP

Status: done (awaiting review)

**Spec:** the `## Spec` section below — every physics, kernel-sequence, geometry, output and
check decision is there. Read `WARP/CLAUDE.md` and `WARP/docs/parallel.md` first.
Branch: `warp-first-try`. Run in the `pepaenv` conda env from `WARP/` (already verified: both
earlier Warp drivers run in it — no need to inspect the environment).

**What to build:** A researcher runs `python Yvonnet_plate_warp.py` from `WARP/` and gets the
Yvonnet pre-notched plate under symmetric 2 MPa tension with bulk damage and anisotropic
interfacial damage evolving by explicit updates, all on the GPU with the cpGIMP basis. The run
writes a sparse ParaView series and an energy-budget CSV, and ends by printing whether the
energy-budget check passes.

New files only, plus one appended writer. `warp_gimp.py`, `warp_hencky.py`, `warp_usl.py`,
`warp_usl_pf.py`, `warp_phasefield.py`, `warp_small_strain.py` and every existing driver stay
untouched.

- `WARP/src/warp_usl_aniso.py` (name proposed): the two-field reset, `p2g`, projection,
  `damage_residual_two_field`, grid update, `g2p`, `stress_update_aniso`, and
  `run_warp_aniso_solver(...)` returning a result dict in the style of `run_warp_pf_solver`.
- `WARP/src/vtk_export.py`: **append** a writer for `velocity`, `von_mises` (plane stress),
  `bulk_damage`, `interfacial_damage`, `history`.
- `WARP/Yvonnet_plate_warp.py`: mirrors `serial_py/Yvonnet_plate.py`'s structure and printouts,
  computes the model constants inline, hard-codes every choice, seeds with the WARP `Mesh` /
  `gauss_2D` / `lagrange_basis_Q4` / `ParticleSet`, applies the 2-cell pad ring and the four-row
  notch, and ends with the energy-budget check plus the initial and final configuration plots.
- Update the module list in `WARP/docs/parallel.md` (additive).

Add to `WARP/tests/` a small pytest for the new kernels on a tiny grid:
- With uniform d_p = d₀ and α_p = α₀, the two-field projection returns d₀ and α₀ at every node
  with pseudo-mass.
- With d = α = 0, H = 0 and ε = 0, `damage_residual_two_field` gives F^d_I = F^α_I = 0.
- `stress_update_aniso` with α = 0, d = 0 gives σ = C_iso,plane-stress : ε (C_a(0) is isotropic).

Compare with tolerances, never equality (`parallel.md`).

**Acceptance criteria:**
- [x] `python Yvonnet_plate_warp.py` runs to completion (T = 20 µs) in `pepaenv` on the CUDA
      device. (Plots call `plt.show()`; run headless with `MPLBACKEND=Agg`.)
- [x] `WARP/vtk_output_Yvonnet_warp/simulation.pvd` opens with exactly the five fields.
- [x] `energy_history_Yvonnet_warp.csv` is written.
- [x] The energy-budget check prints measured values and pass/fail. Report the numbers, nsteps,
      dt, the runtime (ms/step), max d and max α in the ticket comments, whether or not it passes.
- [x] No host↔device copies inside the time loop except on output steps; per-step max d, max α,
      W_ext and D_visc accumulated on the device.
- [x] `python pre_noched_plate_PF_warp.py` and `python CopperTaylorImpact_warp.py` still run
      (both take well under a minute on the GPU). `git diff -- WARP/src/` shows only new files
      and the appended VTK writer.
- [x] The new pytests and the existing WARP pytests pass (`pytest tests` from `WARP/`).

**Blocked by:** None. Can start immediately.

## Spec

### Problem Statement

`serial_py/Yvonnet_plate.py` is the anisotropic two-field phase-field plate (3D-printed layered
material): every particle carries **bulk damage** d and **interfacial damage** α, each with its own
toughness, length scale and viscosity, and α degrades the stiffness anisotropically along the layer
direction θ. It runs as Python loops on the CPU and is limited to 100 steps. The Warp port of the
isotropic AT2 plate (`WARP/pre_noched_plate_PF_warp.py`, `.scratch/warp-phase-field-plate/`) proved
the GPU pattern for one damage field. The researcher now wants the two-field Yvonnet plate on the
GPU, with the same cpGIMP basis and parallel pattern, as the next step towards the anisotropic
layered plate.

### Solution

A new driver `WARP/Yvonnet_plate_warp.py` and one new Warp backbone module (proposed name
`WARP/src/warp_usl_aniso.py`). No existing `WARP/src/warp_*.py` module is modified. The driver
seeds the plate as the serial driver does (with the geometry changes below), uploads it once, runs
an explicit USL step with explicit updates of both damage fields entirely on the device, writes a
sparse VTK series and an energy-budget CSV, and ends with an energy-budget check.

Rules: `WARP/CLAUDE.md` (the Warp backbone is **frozen**: new modules only; numpy helpers such as
`vtk_export.py` take additive changes only; never import from `serial_py/`), kernel conventions in
`WARP/docs/parallel.md`, reasoning in `docs/adr/0001-warp-is-a-gpu-staging-ground.md`. Use the
glossary terms of the root `CONTEXT.md`: **bulk damage** (d), **interfacial damage** (α),
**history field** (H), **pseudo-mass**, **crack surface energy**, **viscous dissipation**,
**energy budget**, **pad ring**, **notch**, **convected particle domain**.

### Decisions

**Physics (faithful port of `serial_py/Yvonnet_plate.py`)**
- Parameters are those of `DEFAULT_PARAMS` in `serial_py/src/yvonnet_model.py` — the values the
  serial driver actually runs. The numbers in the header comment of `Yvonnet_plate.py`
  (E = 10 GPa, h = 0.25 mm, gc_d = 4e3, σ = 1 MPa) are stale and must be ignored.
  - E = 30e9 Pa, ν = 0.25, ρ = 2450 kg/m³
  - θ = 60°, ξ = 30
  - gc_d = 10 N/m, gc_α = 1 N/m, ℓ_d = ℓ_α = 0.2e-3 m
  - plate 0.01 × 0.01 m, h = 0.2e-3 m (square cells), notch length 0.005 m
  - σ = 2e6 Pa, CFL safety factor 0.5
- **SI units** throughout (m, N, Pa, kg, s), as the reference, so printed numbers compare line by
  line with `Yvonnet_plate.py`. **Plane stress.** float32 on the device.
- **Model constants are computed inline in the driver** (no WARP copy of `yvonnet_model.py`),
  exactly as `build_model` does:
  - pref = E/(1−ν²); A1 = pref·[[0.8437,0,0],[0,0,0],[0,0,0]];
    A2 = pref·[[0.1563,ν,0],[ν,1,0],[0,0,(1−ν)/2]]
  - T_σ(θ) (Voigt order 11, 22, 12, as in `build_model`); CA1 = T A1 Tᵀ; M2 = T A2 Tᵀ;
    C_a(α) = CA1 + M2 (1−α)²
  - P_a = e1'⊗e1', ω_a = I + ξ P_a
  - C_max = max eig C_a(0); c_d = sqrt(C_max/ρ); dt_u = h/c_d
  - η_d = 2·ndim·gc_d·ℓ_d/(h·c_d), η_α = η_d (shared viscosity)
  - dt_d = η_d h²/(2 ndim gc_d ℓ_d); dt_α = η_α h²/(2 ndim gc_α ℓ_α (1+ξ))
  - dt = CFL · min(dt_u, dt_d, dt_α) ≈ 8.0e-9 s (the interfacial limit governs; dt_d = dt_u)
  - Print the same diagnostics the serial driver prints (C_a(0) vs isotropic, C_a(1), ω_a, the
    three limits, the viscosities, the ordering check).
- **Constitutive law — total strain, not incremental:**
  dε = sym(L)·dt (Voigt, engineering shear), ε += dε, σ₀ = C_a(α_new) : ε, σ = (1−d)² σ₀,
  **no κ**, H = max(H, ½ ε:σ₀), J = det F. σ₀ is recomputed from the total strain every step
  because C_a depends on α — the incremental `stress_update_pf` cannot be reused.
- **Residuals** (as the serial code, per particle p and node I, weights w, ∇w):
  - F^d_I += [(gc_d/ℓ_d) d_p w + gc_d ℓ_d ∇w·∇d − 2(1−d_p) w H_p] V₀
  - F^α_I += [(gc_α/ℓ_α) α_p w + gc_α ℓ_α ∇w·(ω_a ∇α) − G_α w] V₀,
    G_α = (1−α_p)(1−d_p)² ε:M2:ε
  - ∇d, ∇α are gathered from the grid fields at the start of the step.
- **Grid update**, forward Euler on the shared pseudo-mass m_I = Σ w V₀:
  d_I −= dt/(η_d m_I) F^d_I, α_I −= dt/(η_α m_I) F^α_I.
- **G2P:** pure FLIP velocity, x += dt Σ w v_I, L, F ← (I + dt L) F. Both fields are mapped back,
  clamped to [0, 1] and made irreversible (max with the old particle value).
- USL, no gravity, **no Dirichlet BCs**. Load: constant σ from t = 0, nominal traction (GeoWarp
  style, as in the PF port) t_p = (0, +σ) on the top material cell row and (0, −σ) on the bottom
  row, deposited in P2G as f_I += w_I t_p h.

**Step: kernel sequence (one thread per particle / per node, as in `parallel.md`)**
1. reset (nodes): m, mv, f, f_ext, pseudo-mass, d-numerator, α-numerator, F^d, F^α.
2. p2g (particles, 3×3 cpGIMP stencil, `atomic_add`): mass, momentum, internal force from the
   **degraded** stress, traction, pseudo-mass += w V₀, d-num += w V₀ d_p, α-num += w V₀ α_p.
3. two-field projection (nodes): d_I, α_I = numerator / m_I where m_I > tol, else 0.
4. `damage_residual_two_field` (particles): **one** gather over the 9 nodes of ∇d, ∇α and of the
   grid fields interpolated back, d̃ = Σ w d_I and α̃ = Σ w α_I; then scatter F^d and F^α. It also
   writes the per-particle crack surface energies, using d̃ and α̃ for the local term as the
   reference does:
   - Γ_d,p = gc_d V₀ (d̃² + ℓ_d² |∇d|²) / (2 ℓ_d)
   - Γ_α,p = gc_α V₀ (α̃² + ℓ_α² ∇α·ω_a∇α) / (2 ℓ_α)
5. grid update (nodes): momentum, d_I, α_I, and the external-work increment dt f_ext·v_new
   (post-update velocity) reduced per step on the device.
6. g2p (particles, reads the grid only): as above, plus per-step max d and max α
   (`wp.atomic_max`) and the viscous dissipation V₀[η_d (Δd)² + η_α (Δα)²]/dt of the **clamped**
   increments (`wp.atomic_add`), into per-step device arrays.
7. convected particle domain update: the existing `update_gimp_lp_2d` (import, do not edit).
8. `stress_update_aniso` (particles): total-strain update above; CA1 and M2 passed as `wp.mat33`.

A `run_warp_aniso_solver(...)` time loop in the style of `run_warp_pf_solver` returns a result dict
(VTK entries, energy series, per-step max d / max α, final fields, wall time).

**Basis and geometry (cpGIMP precedent of both earlier Warp ports)**
- cpGIMP from `WARP/src/warp_gimp.py` (import its `@wp.func`s; do not edit). **Convected particle
  domain**, lp0 = (h/2, h/2). GeoWarp clamp at [dx, L − dx].
- 1 particle per cell: 50 × 50 particle mesh, one Gauss point, seeded exactly like the reference.
- **Pad ring of 2 cells** on all four sides: background grid 54 × 54 cells, material offset by
  (2h, 2h).
- **Four-row notch**: remove material rows iy ∈ {23, 24, 25, 26} for x ≤ 0.005 m (the reference
  removes two rows, 24 and 25). With two rows the convected cpGIMP supports bridge the notch faces;
  this follows `Yvonnet_plate_gimp.py` and `pre_noched_plate_PF_warp.py`. Expect
  2500 − 100 = 2400 particles.

**Time**
- T = 20e-6 s (≈ 2500 steps at dt ≈ 8.0e-9 s; print the actual count). `vtk_interval = 50`.
- Every choice is hard-coded. No environment-variable overrides.

**Execution**
- float32 throughout. State lives on the device. The host copies only on output steps, plus once
  at the end. Atomics in P2G and the residual; reproducible to round-off.

**Output**
- VTK: pyvista `.vtp` per output step + `write_pvd` → `WARP/vtk_output_Yvonnet_warp/simulation.pvd`
  (`rmtree` the directory at start). **Five fields**: `velocity`, `von_mises` (plane stress,
  σ_zz = 0), `bulk_damage` (d), `interfacial_damage` (α), `history` (H). Append a new writer to
  `WARP/src/vtk_export.py`; leave the existing writers untouched.
- `energy_history_Yvonnet_warp.csv` (output steps): time_us, K, U, Γ_d, Γ_α, D_visc, W_ext, miss,
  miss_pct_of_w_ext. Energies in J/m (per unit thickness).
- Initial-configuration plot (particle layout, traction rows) and final-configuration plot (two
  panels: bulk damage and interfacial damage), particle scatter plots as in the PF driver.
- **No checkpoint, no resume, no crack-tip CSV.**

**Energy budget** (as `Yvonnet_plate.py` header and main loop, two fields)
- K = Σ ½ m v²; U = Σ ½ V₀ σ:ε (degraded); Γ_d, Γ_α as above; D_visc = Σ_t Σ_p V₀ [η_d (Δd)² +
  η_α (Δα)²]/dt; W_ext = Σ_t dt Σ_I f_I^ext·v_I (post-update velocity).
- miss = (K + U + Γ_d + Γ_α + D_visc) − W_ext, reported as a % of W_ext. Γ_d and Γ_α come from the
  start of the step, the other terms from the end (reason 4 in the serial header).

**Check** (printed at the end, pass/fail with measured values)
- **Energy budget only**: |miss| ≤ 10 % of W_ext at the end of the run. No crack-path check —
  θ = 60° layers are expected to deflect the crack. If the check fails, report the measured values
  and do not loosen the tolerance or change the physics to make it pass.

### Out of scope

Checkpoint and resume, a crack-path or crack-tip check, a tension/compression split, plasticity,
float64, a serial cross-check run, performance tuning, a WARP copy of `yvonnet_model.py`, editing
any existing `WARP/src/warp_*.py` module, any change under `serial_py/` (including the stale header
of `Yvonnet_plate.py`).

## Comments

### 2026-10-25 — implementation (agent)

**New files:**
- `WARP/src/warp_usl_aniso.py`, containing `reset_grid_aniso`, `p2g_aniso`, `grid_two_field_project`,
  `damage_residual_two_field`, `grid_update_aniso`, `g2p_aniso`, `stress_update_aniso` and
  `run_warp_aniso_solver`.
- The driver `WARP/Yvonnet_plate_warp.py`.
- `WARP/tests/test_warp_usl_aniso.py`.

**Changed files (additive only):**
- `WARP/src/vtk_export.py`: `write_particles_vtp_aniso` appended (25 lines).
- `WARP/docs/parallel.md`: one bullet added to the module list.

I wrote the three kernel pytests first and saw them fail with an import error; the kernels then
made them pass. Nothing under `serial_py/` and no existing `WARP/src/warp_*.py` was touched.
Not committed.

**Run:** cuda:0 (RTX 4060 Laptop), warp 1.18, 2400 particles (100 removed by the four-row notch),
54 × 54 grid.
- **nsteps = 2505**, **dt = 7.9835e-9 s**, T = 20 µs.
- The interfacial limit governs: dt_α = 1.5967e-8 s, dt_d = dt_u = 4.9497e-8 s.
- η_d = η_α = 9.8995e-3 Pa·s, C_max = 4.0e10 Pa, c_d = 4040.6 m/s.
- |C_a(0) − C_iso| = 3.8e-6 Pa (float64 round-off on the host).

| Check | Measured | Limit | Result |
|---|---|---|---|
| Energy budget | K + U + Γ_d + Γ_α + D = 0.414409 J/m vs W_ext = 0.451525 J/m; miss = −0.037116 J/m = **−8.22 %** of W_ext | \|miss\| ≤ 10 % | PASS |

**Energy budget at t_final = 20 µs** (J/m, per unit thickness), from `energy_history_Yvonnet_warp.csv`:

| Term | Value | Share of W_ext |
|---|---|---|
| Kinetic K | 3.27316e-1 | 72.5 % |
| Stored elastic U (degraded, on V₀) | 1.42900e-2 | 3.2 % |
| Crack surface energy, bulk Γ_d | 9.01886e-3 | 2.0 % |
| Crack surface energy, interfacial Γ_α | 4.52162e-2 | 10.0 % |
| Viscous dissipation D_visc | 1.85685e-2 | 4.1 % |
| External work W_ext | 4.51525e-1 | — |
| Miss | −3.71159e-2 | −8.22 % |

**Damage:**
- **Max d = 0.3238** (bulk damage never localises; no particle reaches d > 0.5).
- **Max α = 1.0000**, with 170 particles at α > 0.9.
- The interfacial crack starts at the notch tip and runs along the θ = 60° layer direction to the
  top-right edge of the plate. Weak bulk damage (d ≲ 0.3) sits in a band around it. This is the
  deflection the spec expects.
- Once the interface has debonded, the upper-right block is pulled away, so kinetic energy
  dominates at the end.

**The miss over the run:**
- It stays negative throughout: −6.4 % at 2 µs and −7.6 % at 8 µs.
- It peaks at **−12.8 % at 12.0 µs**, while α is localising.
- It then recovers to −8.22 % at the end.

The check is end-of-run only, as specified, so it passes. Mid-run the miss exceeds the 10 % limit
for roughly 10–18 µs. Γ_α and D_visc grow fastest in that window, which fits the one-step lag on
Γ and the irreversibility clamp listed in the serial header.

**Runtime:**
- Full run: 1.008 s of time loop for 2505 steps, **0.40 ms/step including output** (52 VTK frames
  plus the energy samples).
- Kernels only: a scratch copy of the driver with output disabled (not kept) gave
  **0.245 ms/step** over two runs.
- The run is reproducible to round-off: two runs agree on K, U, Γ_d, Γ_α and D_visc to 5–6
  significant digits.

**Deviations from the spec:** none in the physics, geometry, kernel sequence or check. Smaller
points:
- **Tolerances are the reference's SI values:** momentum mass 1e-24 kg, pseudo-mass 1e-30 m². They
  are new constants in `warp_usl_aniso.py` (`GRID_MASS_EPS_SI`, `PSEUDO_MASS_EPS_SI`). I did not
  reuse the PF port's mm/tonne constants, whose documented units differ.
- **The two crack surface energies are written per particle** by `damage_residual_two_field`
  (`crack_energy_d`, `crack_energy_a`). The host sums them on output steps.
- **W_ext is summed only on nodes with nonzero f_ext**, as in the PF port. The value is identical,
  because f_ext is zero elsewhere.
- **The energy CSV header** is `time_us,K,U,Gamma_d,Gamma_alpha,D_visc,W_ext,miss,miss_pct_of_w_ext`.
  Energies are in J/m.
- **Printouts mirror `Yvonnet_plate.py`**, including its E_d / E_a / E_visc labels in the budget
  block. The check line uses Γ / D.
- **`sigma0` (the undegraded σ₀) is kept as a device array**, as in `stress_update_pf`. No other
  kernel reads it.

**Acceptance criteria:**
- `python Yvonnet_plate_warp.py` exits 0 in pepaenv on cuda:0 (T = 20 µs, 2505 steps), run with
  `MPLBACKEND=Agg`.
- `WARP/vtk_output_Yvonnet_warp/simulation.pvd` has 52 time steps. I read it back through
  pyvista's PVD reader: it holds exactly `velocity`, `von_mises`, `bulk_damage`,
  `interfacial_damage` and `history` on 2400 points. I did not open it in the ParaView GUI.
- `energy_history_Yvonnet_warp.csv` is written (52 rows).
- The energy-budget check prints its measured values and PASS. The numbers are above.
- The loop only launches kernels; `.numpy()` runs only in `output()` and once after the loop.
  Max d, max α, W_ext and D_visc are accumulated per step on the device (`wp.atomic_max` /
  `wp.atomic_add`) and read once at the end.
- `python pre_noched_plate_PF_warp.py` exits 0. `python CopperTaylorImpact_warp.py` exits 0 with
  all three checks PASS, but only with `PYTHONIOENCODING=utf-8`: piped to a file, its existing
  `print(... μs ...)` raises UnicodeEncodeError under the Windows cp1252 console encoding. That is
  pre-existing and unrelated to this change. `git diff -- WARP/src/` shows only the appended
  `write_particles_vtp_aniso`; everything else is new files.
- `pytest tests`: 27 passed (the 3 new tests plus the 24 existing ones).

**Review** (`/code-review`, two axes):
- **Spec:** no physics or geometry defects. Every formula was checked line by line against
  `yvonnet_model.py` and `Yvonnet_plate.py`. I fixed the CSV header to the spec's column names.
- **Standards:** no hard violations. These judgement calls are left as they are:
  - The solver takes loose model constants instead of a model object. The spec says to compute
    the constants inline, with no WARP copy of `yvonnet_model.py`.
  - The test rebuilds CA1/M2/ω_a itself, for the same reason.
  - The 3×3 stencil prologue repeats within the module, which is the per-module pattern of the
    frozen siblings.
  - The result-dict keys `interf_damage` / `crack_interf` follow the serial `solver_results`.
