# 01 — Pre-notched AT2 phase-field plate on the GPU with cpGIMP

Status: needs-triage

**Spec:** `.scratch/warp-phase-field-plate/spec.md`. All physics, kernel-sequence, geometry,
output and check decisions are there. Read `WARP/CLAUDE.md` and `WARP/docs/parallel.md` first.
Branch: `warp-first-try`. Run in the `pepaenv` conda env from `WARP/`.

**What to build:** A researcher runs `python pre_noched_plate_PF_warp.py` from `WARP/` and gets
the pre-notched plate under symmetric 1 MPa tension. Bulk damage evolves by the explicit AT2
phase-field update, and everything is computed on the GPU with the cpGIMP basis. The run
writes a sparse ParaView series and two CSV histories, and ends by printing whether the
energy-budget and crack-path checks pass.

New files only. `warp_gimp.py`, `warp_hencky.py`, `warp_usl.py` and every existing driver stay untouched.

- `WARP/src/phasefield.py`: copy of the root `src/phasefield.py` (host-side constants), with a
  header noting where it was copied from.
- `WARP/src/warp_phasefield.py`: AT2 `@wp.func`s (g, g′, w, w′) and the bulk-damage kernels
  `grid_phase_project` and `damage_residual`.
- `WARP/src/warp_small_strain.py`: the elasticity matrix and `stress_update_pf` (ε, σ₀,
  degraded σ, history field).
- `WARP/src/warp_usl_pf.py`: `reset_grid_pf`, `p2g_pf` (with traction), `grid_update_pf`,
  `g2p_pf`, and `run_warp_pf_solver(...)` returning a result dict in the style of
  `run_warp_solver`. That dict carries `vtk_entries`, the energy-budget histories, the per-step
  φ_max, and the crack-tip samples.
- `WARP/src/vtk_export.py`: **append** `write_particles_vtp_pf(...)` writing `velocity`,
  `von_mises`, `damage` and `history`.
- `WARP/pre_noched_plate_PF_warp.py`: mirrors `pre_noched_plate_PF.py`'s structure and printouts,
  hard-codes every choice, and seeds with the WARP `Mesh` / `gauss_2D` / `lagrange_basis_Q4` /
  `ParticleSet`. It applies the 2-cell pad ring and the four-row notch, and ends with the two
  checks plus the initial and final configuration plots.
- Update the module list in `WARP/docs/parallel.md` with the new modules (additive).

Add to `WARP/tests/` a small pytest for the damage kernels on a tiny grid:
- With uniform φ_p = φ₀ on all particles, `grid_phase_project` returns φ₀ at every node that has
  pseudo-mass.
- With φ = 0 and H = 0, `damage_residual` gives F_I = 0.

**Acceptance criteria:**
- [ ] `python pre_noched_plate_PF_warp.py` runs to completion (T = 60 µs) in `pepaenv` on the
      CUDA device.
- [ ] `WARP/vtk_output_PF_warp/simulation.pvd` opens with the four fields only.
- [ ] `energy_history_PF_warp.csv` and `crack_tip_PF_warp.csv` are written.
- [ ] The two checks print measured values and pass/fail. Report the numbers, the runtime and the
      final crack-tip position in the ticket comments, whether or not the checks pass.
- [ ] No host↔device copies inside the time loop except on output steps. The φ_max, W_ext and
      D_visc histories are accumulated on the device.
- [ ] `python CopperTaylorImpact_warp.py` still runs. `git diff -- WARP/src/` shows only new files
      and the appended VTK writer.
- [ ] The new pytests and the existing cpGIMP pytests pass.

**Blocked by:** None. Can start immediately.

## Comments

### 2026-10-25 — implementation (agent)

Built `WARP/src/phasefield.py` (copy of the root module, with a provenance header),
`WARP/src/warp_phasefield.py` (AT2 g/g′/w/w′, `grid_phase_project`, `damage_residual`),
`WARP/src/warp_small_strain.py` (`PlaneStrainMaterial`, plane-strain C, `stress_update_pf`),
`WARP/src/warp_usl_pf.py` (`reset_grid_pf`, `p2g_pf` with nominal traction, `grid_update_pf`,
`g2p_pf`, `run_warp_pf_solver`), and the appended `write_particles_vtp_pf` in `WARP/src/vtk_export.py`.
Also the driver `WARP/pre_noched_plate_PF_warp.py`, `WARP/tests/test_warp_phasefield.py`, and the
module list in `WARP/docs/parallel.md`. The two damage-kernel pytests were written first and seen
failing (import error), then the kernels made them pass. Not committed.
**Not resolved: the crack-path check fails because the crack branches.** The energy-budget check passes.

**Checks** (cuda:0, warp 1.18, 5760 steps, dt = 1.0417e-8 s, T = 60 µs, 63200 particles):

| Check | Measured | Limit | Result |
|---|---|---|---|
| Energy budget | K + U + Γ + D = 3.36107 N·mm vs W_ext = 3.47550 N·mm; miss = −0.11443 N·mm = **3.29 %** of W_ext | ≤ 10 % | PASS |
| Crack path, band | 290 particles with φ > 0.9; **68 of them at \|y − y_notch\| ≥ 1 mm**, max 3.127 mm | all < 4ℓ = 1 mm | **FAIL** |
| Crack path, growth | final x_tip = **72.375 mm** | > 50 mm | PASS |

The second check is one boolean, so it reports FAIL as a whole.

**Why the band check fails.** The crack runs straight along y = 20 mm from the notch tip:
- It first exceeds φ > 0.9 at 23.4 µs (x_tip = 50.37 mm).
- It reaches ≈ 71.9 mm at ≈ 55 µs, a mean speed of ≈ 0.68 mm/µs.
- No particle with φ > 0.9 lies off the line up to the 54.2 µs output.

Between 54.2 and 58.3 µs the crack **branches symmetrically** at x ≈ 72 mm. At 60 µs the two arms reach (75.6, 16.9) and (74.9, 22.4) mm in material coordinates. Every off-band particle sits at x ∈ [71.9, 75.6] mm, on those arms. There is no spurious damage at free surfaces or notch faces.

This geometry and loading are the classic dynamic crack-branching benchmark: a 100 × 40 mm plate, a 50 mm notch, 1 MPa tension, E = 32 GPa, Gc = 3 J/m², ℓ = 0.25 mm. Branching is the expected physics there. So the check, as written ("every particle with φ > 0.9 lies within 4ℓ"), cannot hold for a run long enough to branch. The tolerance was not loosened.

**Energy budget at t = 60 µs** (N·mm per mm thickness), from `energy_history_PF_warp.csv`:

| Term | Value | Share of W_ext |
|---|---|---|
| Kinetic K | 2.88414 | 83.0 % |
| Stored elastic U (degraded, on V₀) | 0.19828 | 5.7 % |
| Crack surface energy Γ_bulk | 0.26954 | 7.8 % |
| Viscous dissipation D_visc | 0.00910 | 0.26 % |
| External work W_ext | 3.47550 | — |
| Miss | −0.11443 | −3.29 % |

The largest |miss| over the run is 4.9 %, at 20.8 µs. Viscous dissipation is ≈ 3 % of the crack surface energy, so η stabilises the march without dominating it.

**Runtime.**
- Full run: 15.6 s of time loop for 5760 steps, **2.71 ms/step including output**: 117 VTK frames plus energy and crack-tip samples, kernels cached.
- Kernels only: a scratch copy of the driver with output disabled (not kept) ran the loop in 1.08 s, **0.19 ms/step**. Output, mostly the pyvista `.vtp` writes, is ≈ 125 ms per frame.
- Runs are reproducible to round-off: K, U and Γ agree to 5–6 significant digits across three runs.

**Deviations from the spec, for review:**
- **Step count: 5760, not ≈ 1700.**
  - `dt_crit_phase(0.25, 2)` = 0.5 · 1e-9 · 0.0625 / (2 · 0.25 · 3e-3) = **2.083e-8 s**, not ≈ 3.5e-8 s.
  - So dt = 0.5 · min(6.92e-8, 2.08e-8) = 1.042e-8 s. The phase-field limit governs.
  - I kept the spec's formula and T = 60 µs, which gives 5760 steps, and 117 VTK frames at `vtk_interval = 50`.
- **Extra parameters on `run_warp_pf_solver`.** It takes `lp0` (as `run_warp_solver` does) and `crack_origin`, `crack_line_y`, `crack_band`, so it can compute the crack-tip samples on output steps. The φ > 0.9 threshold is `warp_usl_pf.CRACKED_PHI`. Particles are in grid coordinates and the crack tip is reported in material x (x − 2h).
- **`damage_residual` writes a per-particle `crack_energy` array**, V₀ (c/2)(w(φ_p) + ℓ²|∇φ_p|²). It is the only place ∇φ_p exists. The host sums it on output steps. As in `Yvonnet_plate.py` (reason 4), Γ_bulk comes from the phase at the start of the step, while K, U and D come from the end of it.
- **D_visc and φ_max are reduced inside `g2p_pf`** with `wp.atomic_add` and `wp.atomic_max` into per-step arrays. `parallel.md` now says this kind of per-step scalar reduction is allowed in G2P. W_ext is reduced the same way in `grid_update_pf`, only on nodes with nonzero f_ext.
- **Tolerances are the reference's:** momentum mass 1e-20 tonne, pseudo-mass 1e-30 mm². Both are representable in float32.
- **von Mises includes σ_zz = ν(σ_xx + σ_yy)** (plane strain), consistent with the WARP elastic writer. The root PF writer omits σ_zz.
- **The energy CSV has an extra last column, `miss_pct_of_w_ext`**, because the spec asks for the miss as a % of W_ext. Times are in µs and energies in N·mm.
- **Plots.** The driver's initial and final plots are particle scatter plots (layout and traction rows; final φ), as in the reference. `plot_mpm_domain` was not used, because drawing a 404 × 164 mesh is impractical. The plots call `plt.show()` like the Copper driver, so I ran headless with `MPLBACKEND=Agg`.

**Acceptance criteria:**
- [x] `python pre_noched_plate_PF_warp.py` runs to completion (T = 60 µs, 5760 steps) in `pepaenv` on cuda:0, exit 0.
- [x] `WARP/vtk_output_PF_warp/simulation.pvd`: 117 time steps. Read back through pyvista's PVD reader, the fields are exactly `velocity`, `von_mises`, `damage`, `history` on 63200 points. Not opened in the ParaView GUI.
- [x] `energy_history_PF_warp.csv` and `crack_tip_PF_warp.csv` are written (117 rows each).
- [x] Both checks print measured values and pass/fail: energy budget PASS, crack path **FAIL** (branching, see above). Numbers, runtime and final crack tip are reported here.
- [x] No host↔device copies in the time loop except on output steps: the loop only launches kernels, and `.numpy()` runs in `output()` and once at the end. φ_max, W_ext and D_visc are accumulated per step on the device and read once at the end.
- [x] `python CopperTaylorImpact_warp.py` still runs (exit 0, all three checks PASS). `git diff -- WARP/src/` shows only the 25-line appended `write_particles_vtp_pf`. Everything else is new files, and `warp_gimp.py`, `warp_hencky.py` and `warp_usl.py` are untouched.
- [x] `pytest tests`: 24 passed (the 2 new damage-kernel tests plus the 22 cpGIMP tests).

**Review** (`/code-review`, two axes). Fixed:
- Exact-equality asserts in the new tests, changed to tolerances per `parallel.md`.
- Glossary naming: `grid_mphi` → `grid_pseudo_mass`, `crack_density` → `crack_energy`.
- The driver now reuses `CRACKED_PHI` and a single `crack_band`.
- Unused `lam`/`mu` removed.

One defect came from my own scratch timing run: it overwrote `simulation.pvd` with an empty collection. The final driver run regenerated it.

**Decision needed:** how should the crack-path check treat branching? Options:
- (a) restrict the band test to x ≤ the branch point, or to t before branching;
- (b) replace it with a branching-aware check, e.g. branch onset x ≈ 72 mm, symmetry of the two arms;
- (c) keep the check and shorten T so the run ends before branching (the straight crack held up to ≈ 54 µs).
