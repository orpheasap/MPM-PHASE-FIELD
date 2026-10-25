# AT2 bulk damage for the Warp backbone: the constitutive functions and the two grid-side
# damage kernels of the explicit USL-MPM-PF step (Algorithm 1 of explicit_USL_MPM_PF.tex,
# as in the root pre_noched_plate_PF.py).
#
#   g(phi) = (1 - phi)^2,  w(phi) = phi^2,  c_w = 1/2
#   F_I    = sum_p [g'(phi_p) w_I H_p + c (1/2 w'(phi_p) w_I + l^2 grad w_I . grad phi_p)] V0_p
#   c      = Gc / (2 c_w l)
#
# grid_phase_project and damage_residual each sit behind a device-wide barrier: grad phi_p needs
# every phi_I finished, and the phi update needs every F_I finished. Kernel conventions:
# docs/parallel.md. Host-side constants: phasefield.py.
import warp as wp

from .warp_gimp import gimp_stencil_origin, gimp_weights_gradients_2d

# Nodes with less pseudo-mass than this carry no bulk damage (mm^2 per unit thickness),
# the reference's tol_phi.
PHI_MASS_EPS = wp.constant(1.0e-30)


@wp.func
def at2_g(phi: wp.float32):
    """Degradation g(phi) = (1 - phi)^2."""
    return (1.0 - phi) * (1.0 - phi)


@wp.func
def at2_dg(phi: wp.float32):
    """g'(phi) = -2 (1 - phi)."""
    return -2.0 * (1.0 - phi)


@wp.func
def at2_w(phi: wp.float32):
    """Local crack geometry function w(phi) = phi^2."""
    return phi * phi


@wp.func
def at2_dw(phi: wp.float32):
    """w'(phi) = 2 phi."""
    return 2.0 * phi


@wp.kernel
def grid_phase_project(grid_pseudo_mass: wp.array(dtype=wp.float32),
                       grid_phinum: wp.array(dtype=wp.float32),
                       grid_phi: wp.array(dtype=wp.float32)):
    """phi_I = (sum_p w_I V0 phi_p) / m_I^phi on nodes with pseudo-mass, 0 elsewhere."""
    n = wp.tid()
    m = grid_pseudo_mass[n]
    if m > PHI_MASS_EPS:
        grid_phi[n] = grid_phinum[n] / m
    else:
        grid_phi[n] = 0.0


@wp.kernel
def damage_residual(x: wp.array(dtype=wp.vec2),
                    vol0: wp.array(dtype=wp.float32),
                    phi: wp.array(dtype=wp.float32),
                    H: wp.array(dtype=wp.float32),
                    lp: wp.array(dtype=wp.vec2),
                    grid_phi: wp.array(dtype=wp.float32),
                    dx: wp.float32,
                    n_grid_x: wp.int32,
                    crack_coeff: wp.float32,
                    l2: wp.float32,
                    grid_F: wp.array(dtype=wp.float32),
                    crack_energy: wp.array(dtype=wp.float32)):
    """Gather grad phi_p from the grid, then scatter the damage residual F_I.

    Also writes the particle's crack surface energy V0 Gc/(4 c_w l) (w(phi_p) + l^2 |grad phi_p|^2)
    = V0 c/2 (...) into crack_energy, from the grid phase at the start of the step.
    """
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    grad_phi = wp.vec2(0.0, 0.0)
    for i in range(3):
        for j in range(3):
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            grad_phi += grad_w * grid_phi[node]

    V0 = vol0[p]
    phi_p = phi[p]
    dg_H = at2_dg(phi_p) * H[p]
    half_dw = 0.5 * at2_dw(phi_p)
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            F_I = (dg_H * w + crack_coeff * (half_dw * w + l2 * wp.dot(grad_w, grad_phi))) * V0
            wp.atomic_add(grid_F, node, F_I)

    crack_energy[p] = 0.5 * crack_coeff * V0 * (at2_w(phi_p) + l2 * wp.dot(grad_phi, grad_phi))
