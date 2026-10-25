# 01 — Elastic copper Taylor bar on the GPU with cpGIMP

Status: needs-triage

**Spec:** `.scratch/warp-first-try/spec.md` — all physics, geometry, output and check decisions are
there. Read `WARP/CLAUDE.md` and `WARP/docs/parallel.md` first. Branch: `warp-first-try`. Run in the
`pepaenv` conda env from `WARP/`.

**What to build:** A researcher runs `python CopperTaylorImpact_warp.py` from `WARP/` and gets an
elastic copper Taylor bar impacting a rigid wall, computed entirely on the GPU with the cpGIMP
basis. The run writes a sparse ParaView series and two CSV histories, and ends by printing whether
the energy, contact-time and rebound checks pass.

New files only (the numpy backbone and `CopperTaylorImpact.py` stay untouched):

- `WARP/src/warp_gimp.py` — cpGIMP weight/gradient `@wp.func` and the convected, clamped
  particle-domain update kernel, ported from `geowarp_clone/mpm/gimp.py`, float32, per-direction
  `lp0`, with a header crediting GeoWarp (Zhao et al. 2026, MIT).
- `WARP/src/warp_hencky.py` — an `ElasticMaterial` (ρ, E, ν → G, K, λ, μ, `wave_speed`) and the
  Hencky Cauchy-stress / elastic-energy-density `@wp.func`s.
- `WARP/src/warp_usl.py` — grid reset, P2G, grid update + rigid wall (with device-side wall-reaction
  accumulation), G2P, stress-update kernels, and
  `run_warp_solver(mesh, particles, material, dtime, time, alpha, vtk_output_dir, vtk_interval, track_pids)`
  returning a result dict in the style of `run_mpm_solver` (`vtk_entries`, `track_times`,
  `track_positions`, plus energy and wall-reaction histories).
- `WARP/src/vtk_export.py` — **append** `write_particles_vtp_elastic(...)` writing the four fields
  (`velocity`, `von_mises`, `pressure`, `J`); existing functions unchanged, `write_pvd` reused.
- `WARP/CopperTaylorImpact_warp.py` — mirrors `CopperTaylorImpact.py`'s structure and printouts,
  hard-codes every choice, seeds with the existing numpy `Mesh` / `gauss_2D` /
  `lagrange_basis_Q4` / `ParticleSet`, shifts the bar per the spec's cantilever-style layout, and
  ends with the three checks and the initial/final configuration plots.

Also add `WARP/tests/` with a small pytest for the cpGIMP func run through a tiny Warp kernel:
partition of unity (weights sum to 1) and gradients sum to 0 for a few particle positions and
half-widths, including one domain spanning a cell boundary and one clamped at the grid edge.

**Acceptance criteria:**
- [x] `python CopperTaylorImpact_warp.py` runs to completion in `pepaenv` on the CUDA device.
- [x] `WARP/vtk_output_warp/simulation.pvd` opens in ParaView with the four fields only.
- [x] `particle_history_warp.csv` and `energy_history_warp.csv` are written.
- [x] The three checks print measured values and pass/fail; report the numbers in the ticket
      comments whether or not they pass.
- [x] No host↔device copies inside the time loop except on output steps.
- [x] `python CopperTaylorImpact.py` still runs (numpy reference unaffected);
      `git diff -- WARP/src/` shows only new files and the appended VTK writer.
- [x] cpGIMP pytest passes.
- [ ] On pass, set `Freeze date:` in `WARP/CLAUDE.md` (the Warp backbone becomes additive).

**Blocked by:** None — can start immediately.

## Comments

### 2026-10-25 — implementation (agent)

Built `WARP/src/warp_gimp.py`, `warp_hencky.py`, `warp_usl.py`, appended
`write_particles_vtp_elastic` to `WARP/src/vtk_export.py`, the driver
`WARP/CopperTaylorImpact_warp.py`, and `WARP/tests/` (cpGIMP pytest, written and green first).
**Not resolved: the energy check fails.** Contact-time and rebound checks pass. Freeze date not set.

**Checks** (cuda:0, RTX 4060 Laptop, warp 1.18, 340 steps, dt = 1.184e-7 s, 40.3 µs):

| Check | Measured | Limit | Result |
|---|---|---|---|
| Energy drift | E_tot(0) = 31150.3 J/m → E_tot(end) = 27582.2 J/m (kin 25938.8, el 1643.4); **11.45 %** of E_kin(0) | ≤ 5 % | **FAIL** |
| Contact time | 13.62 µs (wall reaction > 1 % of peak 5.62e7 N/m, steps 0–114) vs 2L/c_bar = 13.47 µs; 1.1 % | ≤ 15 % | PASS |
| Rebound | mean v_y = 168.87 m/s = 0.889 V0, after contact ended | ≥ 0.8 V0 | PASS |

Runtime: time loop 0.20 s for 340 steps (0.60 ms/step including the 35 output steps); kernels
were already in Warp's cache. Final bar: length 24.98 mm, width 7.19 mm, J in [0.992, 1.012].

**Where the energy goes** (from `energy_history_warp.csv`): ~7.2 % of E_kin(0) is lost during
contact (0–13.6 µs), ~4.2 % as a slow drift in free flight afterwards. Diagnostic runs (scratch
copies of the driver, not committed):
- pure FLIP (`alpha = 1.0`): drift 10.77 % → the 1 % PIC blend accounts for only ~0.7 points;
- dt halved (0.25 · dx / c_p, 680 steps): drift 11.74 % → not time-integration error.
So the loss looks spatial: the rigid-wall clamp stopping wall-row nodal momentum (inelastic at
node level) plus USL grid-transfer dissipation at 1 mm cells (~7.6 cells across the bar). No
grid-refinement study was run (out of scope). The 5 % tolerance was not changed.

**Deviations from the spec, for review:**
- **Grid is 25 × 36 cells, not ≈ 25 × 31.** The bar rebounds at ~0.89 V0, so over 40 µs its top
  climbs to 30.90 mm. With 31 cells the clamp is at 30 mm and particles would index nodes off the
  grid. Keeping the 40 µs run meant a taller grid; the rest of the layout (wall row y = dx, pad ring,
  bar bottom at dx + 0.1 mm, clamp at [dx, L − dx]) is as specified.
- `run_warp_solver` takes one extra argument, `lp0` (n × 2 initial half-widths). The spec'd
  signature has no way to pass the particle domain. The rigid wall is hard-coded to node row 1
  (y = dx), as the cantilever-style layout says.
- GeoWarp's `+1e-8` index nudges are dropped (no-ops in float32, per `docs/parallel.md`). The
  clamp's `1e-6` offset is kept as in GeoWarp.

**Other acceptance items:** `simulation.pvd` (35 steps) read back through pyvista's PVD reader
with exactly `velocity`, `von_mises`, `pressure`, `J` (not opened in the ParaView GUI). Inside the
loop the host only launches kernels; `.numpy()` runs only in output steps and once at the end, and
the wall reaction is accumulated on the device into a per-step array that is read once.
`python CopperTaylorImpact.py` still runs (exit 0). `git diff -- WARP/src/` shows only the appended
writer; the rest are new files. `pytest tests` → 22 passed.

Environment notes: `pytest` was not in `pepaenv` and was pip-installed (9.1.1). When stdout is
piped, the drivers' `μ` printouts need `PYTHONIOENCODING=utf-8` (the console is cp1252); the numpy
driver has the same issue.

**Decision needed:** accept the scheme's ~11 % loss and revisit the energy tolerance, or try
refining (finer grid / more particles per cell) before freezing the Warp backbone.
