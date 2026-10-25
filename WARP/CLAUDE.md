# WARP — GPU sub-project

`WARP/` is where this project's MPM is grown on the GPU with NVIDIA Warp. The aim is a Warp port
of the anisotropic phase-field plate; the elastic copper Taylor bar is the warm-up. The root numpy
project is the reference every Warp result is checked against — see
`../docs/adr/0001-warp-is-a-gpu-staging-ground.md`.

## Project rules

### Self-contained

- Drivers live at `WARP/` and are run with `WARP/` as the working directory.
- Code here imports only from `WARP/src/` (and third-party packages). Never import from the root
  `src/`; if a root helper is needed, copy it into `WARP/src/`.
- `geowarp_clone/` is a read-only, git-ignored reference copy of GeoWarp (MIT). Never import from
  it and never edit it. Routines ported from it are copied into `WARP/src/` with a header crediting
  GeoWarp (Zhao et al. 2026, MIT).

### The WARP backbone and when it freezes

`WARP/src/` holds two generations of code:

- **The numpy backbone** (`material.py`, `mesh2D.py`, `particle.py`, `solver1.py`,
  `vtk_export.py`, …) behind `CopperTaylorImpact.py`. It is already frozen: only additive changes,
  and `CopperTaylorImpact.py` must keep running unchanged.
- **The Warp backbone** (`warp_*.py`). It **may be refactored freely until the first milestone —
  the elastic Taylor impact passing its checks — is reached.** From then on it follows the same
  additive rule as the root `src/`: new capabilities arrive as new modules or functions, existing
  signatures and behaviour stay untouched, and every Warp driver keeps running. Record the freeze
  date here when it happens.

Freeze date: **2026-10-25** — the elastic Taylor bar passed all three checks (energy drift < 5 %
after refining to 0.5 mm cells, commit `7a1c629`). `warp_gimp.py`, `warp_hencky.py` and
`warp_usl.py` now take additive changes only; new capabilities go into new `warp_*.py` modules.

### Drivers hard-code their choices

As in the root: no runtime flags, basis selectors or config switches. A driver picks its kernels,
material, transfer scheme and output by hard-coding them; switching is done by editing the driver.

### Parallel conventions

Read `docs/parallel.md` before writing or changing any kernel.

## Environment

Run everything in the `pepaenv` conda env (warp-lang 1.18, pyvista, meshio; CUDA device present).
The default `python` on this machine has none of these.

## Docs

- `README.md` — what WARP is and how to run it.
- `docs/parallel.md` — device, precision, kernel and host-sync conventions.
- `docs/musl-ep-algorithm.md` — the MUSL elasto-plastic algorithm behind the numpy
  `CopperTaylorImpact.py`.
- Glossary: the root `../CONTEXT.md` (single context — use its terms, e.g. *rigid wall*,
  *particle domain*, *convected particle domain*, *pad ring*).
