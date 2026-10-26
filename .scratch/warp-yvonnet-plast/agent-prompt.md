Implement the ticket `.scratch/warp-yvonnet-plast/issues/01-yvonnet-plate-plast-warp.md` in this
repo, the elastoplastic Yvonnet plate on the GPU. The ticket is fully specified: build exactly
what its `## Spec` says, and stop and ask before departing from it.

Read these first, in this order:
1. `CLAUDE.md`, then `WARP/CLAUDE.md`, for the backbone freeze and the additive-only rule. There
   are no runtime flags; drivers hard-code their choices.
2. `WARP/docs/parallel.md`, the kernel conventions: float32, state on the device, host copies only
   on output steps.
3. The ticket.
4. `WARP/Yvonnet_plate_warp.py` and `WARP/src/warp_usl_aniso.py`. The new driver is a copy of the
   first, and the new module reuses the second's kernels unchanged.
5. §6 and Algorithm 1 of `WARP/plastic_aniso_layer_PF_MPM_explicit.tex`, the theory behind the
   return map. It is the researcher's report: read it only, never edit or compile it.
6. `CONTEXT.md`, the glossary. Use its terms in comments and docstrings, in particular *plastic
   strain*, *equivalent plastic strain*, *yield stress*, *hardening modulus*, *plastic work* and
   *history field*. Never name the hardening modulus `H`.

Hard constraints:
- Do not modify any existing `WARP/src/warp_*.py` file or any existing driver. `vtk_export.py`
  gets one appended function and nothing else.
- Never import from `serial_py/` or `WARP/geowarp_clone/`.
- Run everything from `WARP/` in the `pepaenv` conda env, with `MPLBACKEND=Agg` for the drivers.

Suggested order: write the tests for the return map and stress kernel, then the module, then the
driver, then the full run. Do the acceptance checks last.

When you finish, tick the acceptance criteria in the ticket and append your results under
`## Comments`: the numbers, the elastic-vs-plastic comparison, and anything that surprised you.
Set `Status: done (awaiting review)`. Do not commit unless asked.
