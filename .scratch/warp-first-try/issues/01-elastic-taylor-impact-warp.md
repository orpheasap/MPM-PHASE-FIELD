# 01 — Elastic copper Taylor bar on the GPU with cpGIMP

Status: ready-for-agent

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
- [ ] `python CopperTaylorImpact_warp.py` runs to completion in `pepaenv` on the CUDA device.
- [ ] `WARP/vtk_output_warp/simulation.pvd` opens in ParaView with the four fields only.
- [ ] `particle_history_warp.csv` and `energy_history_warp.csv` are written.
- [ ] The three checks print measured values and pass/fail; report the numbers in the ticket
      comments whether or not they pass.
- [ ] No host↔device copies inside the time loop except on output steps.
- [ ] `python CopperTaylorImpact.py` still runs (numpy reference unaffected);
      `git diff -- WARP/src/` shows only new files and the appended VTK writer.
- [ ] cpGIMP pytest passes.
- [ ] On pass, set `Freeze date:` in `WARP/CLAUDE.md` (the Warp backbone becomes additive).

**Blocked by:** None — can start immediately.

## Comments
