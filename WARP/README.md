# WARP — GPU-parallel MPM with NVIDIA Warp

The GPU staging ground of the MPM phase-field project. Explicit MPM kernels are written in
[NVIDIA Warp](https://nvidia.github.io/warp/) and grown here step by step, aiming at a Warp port
of the anisotropic phase-field plate. The root numpy project stays the reference that every Warp
result is checked against.

## Drivers

| Driver | Backbone | What it runs |
|---|---|---|
| `CopperTaylorImpact.py` | numpy (`src/solver1.py`) | Copper Taylor bar, MUSL, Johnson-Cook + Mie-Grüneisen, linear hat basis. Reference, frozen. |
| `CopperTaylorImpact_warp.py` | Warp (`src/warp_*.py`) | Elastic copper Taylor bar, USL with FLIP/PIC blend, Hencky elasticity, cpGIMP basis, rigid wall. |

Run from this directory, in the `pepaenv` env:

```bash
python CopperTaylorImpact_warp.py
```

VTK output is a `.vtp` series plus `simulation.pvd` for ParaView, written every `vtk_interval`
steps (not every step).

## Layout

- `src/` — the backbone: the frozen numpy routines and the Warp routines (`warp_*.py`).
- `docs/parallel.md` — how the Warp code is written (device state, float32, atomics, kernels).
- `docs/musl-ep-algorithm.md` — the MUSL elasto-plastic algorithm of the numpy reference.
- `geowarp_clone/` — local, git-ignored reference copy of
  [GeoWarp](https://doi.org/10.1016/j.advengsoft.2025.104072) (MIT), the source of the cpGIMP
  kernels. Not imported.

Rules for working here: `CLAUDE.md`. Glossary: `../CONTEXT.md`.
