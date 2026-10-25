# WARP is a GPU staging ground; the serial numpy project stays the reference

The project is moving towards GPU-parallel simulation with NVIDIA Warp, ultimately to run the
anisotropic phase-field plate. Rather than port the serial numpy backbone in place or let the Warp
code replace it, the Warp work lives in a self-contained `WARP/` sub-project with its own backbone
and drivers. The serial numpy drivers stay runnable and authoritative: they are the oracle that
Warp results are checked against, and they are never retired in favour of their Warp ports.

(When this was decided the serial project sat at the repo root. On 2026-10-25 it moved into its
own `serial_py/` sub-project, laid out like `WARP/`; the decision is unchanged.)

## Considered Options

- **Port in place / successor** — WARP eventually absorbs the serial project. Rejected: it breaks
  the reproducibility rule (old drivers runnable indefinitely) and leaves no independent reference
  to check GPU results against.
- **Independent side project** — WARP with no tie to the serial project's physics. Rejected: the
  point of the Warp work is to run *this* project's models faster, and without a reference a
  float32, atomics-based solver has nothing to be wrong against.

## Consequences

- `WARP/` imports only from `WARP/src/`; `serial_py/` never imports from `WARP/`. Host-side
  helpers the Warp code needs (mesh, quadrature, seeding) are copied from `serial_py/src/` into
  `WARP/src/`, not shared.
- The domain is the same on both sides, so the repo stays single-context: one `CONTEXT.md` at the
  root. Parallel-execution conventions are implementation detail and live in
  `WARP/docs/parallel.md`.
- Warp results match the reference to round-off and tolerance, never bitwise.
- GeoWarp (MIT) is the source of the cpGIMP kernels. A local copy sits in `WARP/geowarp_clone/`
  as a read-only, git-ignored reference; ported routines are copied into `WARP/src/` with credit.
