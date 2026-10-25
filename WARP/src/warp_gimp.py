# cpGIMP basis for the Warp backbone.
#
# Ported from GeoWarp, mpm/gimp.py (get_GIMP_shape_function_and_gradient_2d, update_GIMP_lp_2d).
# GeoWarp — Zhao et al. (2026), MIT licence. https://doi.org/10.1016/j.advengsoft.2025.104072
#
# Changes from the original:
#   - retyped float64 / vec2d / mat33d -> float32 / vec2 / mat33 (docs/parallel.md);
#     GeoWarp's +1e-8 cell-index nudges are below float32 resolution and are dropped;
#   - the initial particle-domain half-width lp0 is a per-direction vec2 per particle
#     (GeoWarp's is one scalar for all particles and both directions);
#   - gimp_stencil_origin() factors out the lower-left support corner that GeoWarp computes
#     inline in every kernel, so P2G and G2P index the same 3x3 stencil.
import warp as wp

# (wx0, wy0, wx1, wy1, wx2, wy2, gwx0, gwy0, gwx1, gwy1, gwx2, gwy2) — GeoWarp's layout
vec12 = wp.types.vector(length=12, dtype=wp.float32)


@wp.func
def gimp_stencil_origin(xp: wp.vec2, inv_dx: wp.float32, lp: wp.vec2):
    """Lower-left node (ix, iy) of the particle's 3x3 cpGIMP stencil."""
    lbc = xp - lp
    return wp.vec2i(wp.int32(lbc[0] * inv_dx), wp.int32(lbc[1] * inv_dx))


@wp.func
def gimp_weights_gradients_2d(xp: wp.vec2, dx: wp.float32, inv_dx: wp.float32,
                              lpx: wp.float32, lpy: wp.float32):
    """1D cpGIMP weights and gradients of the 3 stencil nodes in x and y.

    Node (i, j) of the stencil has weight w_xi * w_yj and gradient
    (gw_xi * w_yj, w_xi * gw_yj). Square cells (one dx) are assumed.
    """
    lbc = xp - wp.vec2(lpx, lpy)
    lbc_int = wp.vec2i(wp.int32(lbc[0] * inv_dx), wp.int32(lbc[1] * inv_dx))
    lbc_base = wp.vec2(wp.float32(lbc_int[0]), wp.float32(lbc_int[1]))

    ruc = xp + wp.vec2(lpx, lpy)
    ruc_int = wp.vec2i(wp.int32(ruc[0] * inv_dx), wp.int32(ruc[1] * inv_dx))

    base_int = wp.vec2i(wp.int32(xp[0] * inv_dx), wp.int32(xp[1] * inv_dx))
    base = wp.vec2(wp.float32(base_int[0]), wp.float32(base_int[1]))

    fx = xp * inv_dx - base

    wx0 = wp.float32(0.0)
    wy0 = wp.float32(0.0)
    wx1 = wp.float32(0.0)
    wy1 = wp.float32(0.0)
    wx2 = wp.float32(0.0)
    wy2 = wp.float32(0.0)
    grad_wx0 = wp.float32(0.0)
    grad_wy0 = wp.float32(0.0)
    grad_wx1 = wp.float32(0.0)
    grad_wy1 = wp.float32(0.0)
    grad_wx2 = wp.float32(0.0)
    grad_wy2 = wp.float32(0.0)

    if ruc_int[0] == lbc_int[0]:
        # particle domain inside one cell in x: linear hat
        wx0 = wp.float32(1.0) - fx[0]
        wx1 = fx[0]
        grad_wx0 = -inv_dx
        grad_wx1 = inv_dx
    else:
        # particle domain spans a node in x — Coombs et al., section 2.4.2
        x_0 = lbc_base[0] * dx
        wx0 = wp.pow(dx + lpx - wp.abs(xp[0] - x_0), 2.0) / (4.0 * dx * lpx)
        grad_wx0 = -(dx + lpx - wp.abs(xp[0] - x_0)) / (2.0 * dx * lpx) * (xp[0] - x_0) / wp.abs(xp[0] - x_0)

        x_1 = (lbc_base[0] + 1.0) * dx
        wx1 = 1.0 - (wp.pow(xp[0] - x_1, 2.0) + wp.pow(lpx, 2.0)) / (2.0 * dx * lpx)
        grad_wx1 = -(xp[0] - x_1) / (dx * lpx)

        x_2 = (lbc_base[0] + 2.0) * dx
        wx2 = wp.pow(dx + lpx - wp.abs(xp[0] - x_2), 2.0) / (4.0 * dx * lpx)
        grad_wx2 = -(dx + lpx - wp.abs(xp[0] - x_2)) / (2.0 * dx * lpx) * (xp[0] - x_2) / wp.abs(xp[0] - x_2)

    if ruc_int[1] == lbc_int[1]:
        wy0 = wp.float32(1.0) - fx[1]
        wy1 = fx[1]
        grad_wy0 = -inv_dx
        grad_wy1 = inv_dx
    else:
        y_0 = lbc_base[1] * dx
        wy0 = wp.pow(dx + lpy - wp.abs(xp[1] - y_0), 2.0) / (4.0 * dx * lpy)
        grad_wy0 = -(dx + lpy - wp.abs(xp[1] - y_0)) / (2.0 * dx * lpy) * (xp[1] - y_0) / wp.abs(xp[1] - y_0)

        y_1 = (lbc_base[1] + 1.0) * dx
        wy1 = 1.0 - (wp.pow(xp[1] - y_1, 2.0) + wp.pow(lpy, 2.0)) / (2.0 * dx * lpy)
        grad_wy1 = -(xp[1] - y_1) / (dx * lpy)

        y_2 = (lbc_base[1] + 2.0) * dx
        wy2 = wp.pow(dx + lpy - wp.abs(xp[1] - y_2), 2.0) / (4.0 * dx * lpy)
        grad_wy2 = -(dx + lpy - wp.abs(xp[1] - y_2)) / (2.0 * dx * lpy) * (xp[1] - y_2) / wp.abs(xp[1] - y_2)

    return vec12(wx0, wy0, wx1, wy1, wx2, wy2,
                 grad_wx0, grad_wy0, grad_wx1, grad_wy1, grad_wx2, grad_wy2)


@wp.kernel
def update_gimp_lp_2d(F: wp.array(dtype=wp.mat33),
                      lp: wp.array(dtype=wp.vec2),
                      lp0: wp.array(dtype=wp.vec2),
                      x: wp.array(dtype=wp.vec2),
                      min_x: wp.float32,
                      max_x: wp.float32,
                      min_y: wp.float32,
                      max_y: wp.float32):
    """Convected particle domain, clamped so it never crosses [min, max] in either direction."""
    p = wp.tid()

    # Charlton et al. 2017, Eq. (38)-(39): stretch by the column norms of F
    Fp = F[p]
    U00 = wp.sqrt(Fp[0, 0] * Fp[0, 0] + Fp[1, 0] * Fp[1, 0] + Fp[2, 0] * Fp[2, 0])
    U11 = wp.sqrt(Fp[0, 1] * Fp[0, 1] + Fp[1, 1] * Fp[1, 1] + Fp[2, 1] * Fp[2, 1])

    lpx = lp0[p][0] * U00
    lpy = lp0[p][1] * U11
    xp = x[p]

    if xp[0] - lpx < min_x:
        lpx = xp[0] - min_x - wp.float32(1e-6)
    if xp[0] + lpx > max_x:
        lpx = max_x - xp[0] - wp.float32(1e-6)
    if xp[1] - lpy < min_y:
        lpy = xp[1] - min_y - wp.float32(1e-6)
    if xp[1] + lpy > max_y:
        lpy = max_y - xp[1] - wp.float32(1e-6)

    lp[p] = wp.vec2(lpx, lpy)
