# Explicit USL MPM with two damage fields on the GPU -- bulk damage d (AT2) and anisotropic
# interfacial damage alpha -- cpGIMP basis, small-strain total-strain plane stress, nominal
# traction, pure FLIP. Port of serial_py/Yvonnet_plate.py. SI units (m, N, Pa, kg, s).
#
#   sigma   = (1 - d)^2 C_a(alpha) : eps,   C_a(alpha) = CA1 + M2 (1 - alpha)^2   (no kappa)
#   H       = max(H, 1/2 eps : C_a(alpha) : eps)
#   F^d_I   = sum_p [(gc_d/l_d) d_p w + gc_d l_d grad w . grad d - 2 (1 - d_p) w H_p] V0
#   F^a_I   = sum_p [(gc_a/l_a) a_p w + gc_a l_a grad w . (omega_a grad a) - G_a w] V0
#   G_a     = (1 - a_p) (1 - d_p)^2 eps : M2 : eps
#   d_I    -= dt/(eta_d m_I) F^d_I,   a_I -= dt/(eta_a m_I) F^a_I   (shared pseudo-mass m_I)
#
# One step:
#   reset_grid_aniso -> p2g_aniso (mass, momentum, degraded internal force, traction,
#   pseudo-mass, d and alpha numerators) -> grid_two_field_project -> damage_residual_two_field
#   -> grid_update_aniso (momentum, d_I, alpha_I) -> g2p_aniso (FLIP, x, L, F, both fields
#   clamped and irreversible) -> convected particle domain update -> stress_update_aniso.
# Kernel conventions: docs/parallel.md. State stays on the device; the host reads it back only
# on output steps. The max d, max alpha, external-work and viscous-dissipation histories are
# accumulated per step on the device and read once at the end.
import time as _time

import numpy as np
import warp as wp

from .warp_gimp import gimp_stencil_origin, gimp_weights_gradients_2d, update_gimp_lp_2d
from .vtk_export import write_particles_vtp_aniso

# Nodes lighter than this carry no momentum (kg per unit thickness), the reference's tol.
GRID_MASS_EPS_SI = wp.constant(1.0e-24)

# Nodes with less pseudo-mass than this carry no damage (m^2 per unit thickness), the
# reference's tol_pm.
PSEUDO_MASS_EPS_SI = wp.constant(1.0e-30)


@wp.kernel
def reset_grid_aniso(grid_m: wp.array(dtype=wp.float32),
                     grid_mv: wp.array(dtype=wp.vec2),
                     grid_f: wp.array(dtype=wp.vec2),
                     grid_fext: wp.array(dtype=wp.vec2),
                     grid_pseudo_mass: wp.array(dtype=wp.float32),
                     grid_dnum: wp.array(dtype=wp.float32),
                     grid_anum: wp.array(dtype=wp.float32),
                     grid_Fd: wp.array(dtype=wp.float32),
                     grid_Fa: wp.array(dtype=wp.float32)):
    n = wp.tid()
    grid_m[n] = 0.0
    grid_mv[n] = wp.vec2(0.0, 0.0)
    grid_f[n] = wp.vec2(0.0, 0.0)
    grid_fext[n] = wp.vec2(0.0, 0.0)
    grid_pseudo_mass[n] = 0.0
    grid_dnum[n] = 0.0
    grid_anum[n] = 0.0
    grid_Fd[n] = 0.0
    grid_Fa[n] = 0.0


@wp.kernel
def p2g_aniso(x: wp.array(dtype=wp.vec2),
              v: wp.array(dtype=wp.vec2),
              mass: wp.array(dtype=wp.float32),
              vol0: wp.array(dtype=wp.float32),
              J: wp.array(dtype=wp.float32),
              sigma: wp.array(dtype=wp.vec3),
              traction: wp.array(dtype=wp.vec2),
              d: wp.array(dtype=wp.float32),
              alpha: wp.array(dtype=wp.float32),
              lp: wp.array(dtype=wp.vec2),
              dx: wp.float32,
              n_grid_x: wp.int32,
              grid_m: wp.array(dtype=wp.float32),
              grid_mv: wp.array(dtype=wp.vec2),
              grid_f: wp.array(dtype=wp.vec2),
              grid_fext: wp.array(dtype=wp.vec2),
              grid_pseudo_mass: wp.array(dtype=wp.float32),
              grid_dnum: wp.array(dtype=wp.float32),
              grid_anum: wp.array(dtype=wp.float32)):
    """sigma is the degraded Cauchy stress in Voigt order (xx, yy, xy).

    The traction is nominal (GeoWarp style): f_I += w_I t_p dx, one particle per cell along the
    loaded edge. grid_f collects internal + external force, grid_fext the external part alone.
    """
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    mp = mass[p]
    mvp = mp * v[p]
    V0 = vol0[p]
    Vp = V0 * J[p]
    sp = sigma[p]
    s2 = wp.mat22(sp[0], sp[2], sp[2], sp[1])
    t_dx = traction[p] * dx
    d_V0 = d[p] * V0
    a_V0 = alpha[p] * V0

    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            f_ext = w * t_dx
            wp.atomic_add(grid_m, node, w * mp)
            wp.atomic_add(grid_mv, node, w * mvp)
            wp.atomic_add(grid_f, node, -Vp * (s2 * grad_w) + f_ext)
            wp.atomic_add(grid_fext, node, f_ext)
            wp.atomic_add(grid_pseudo_mass, node, w * V0)
            wp.atomic_add(grid_dnum, node, w * d_V0)
            wp.atomic_add(grid_anum, node, w * a_V0)


@wp.kernel
def grid_two_field_project(grid_pseudo_mass: wp.array(dtype=wp.float32),
                           grid_dnum: wp.array(dtype=wp.float32),
                           grid_anum: wp.array(dtype=wp.float32),
                           grid_d: wp.array(dtype=wp.float32),
                           grid_alpha: wp.array(dtype=wp.float32)):
    """d_I, alpha_I = (sum_p w_I V0 field_p) / m_I on nodes with pseudo-mass, 0 elsewhere."""
    n = wp.tid()
    m = grid_pseudo_mass[n]
    if m > PSEUDO_MASS_EPS_SI:
        grid_d[n] = grid_dnum[n] / m
        grid_alpha[n] = grid_anum[n] / m
    else:
        grid_d[n] = 0.0
        grid_alpha[n] = 0.0


@wp.kernel
def damage_residual_two_field(x: wp.array(dtype=wp.vec2),
                              vol0: wp.array(dtype=wp.float32),
                              d: wp.array(dtype=wp.float32),
                              alpha: wp.array(dtype=wp.float32),
                              H: wp.array(dtype=wp.float32),
                              eps: wp.array(dtype=wp.vec3),
                              lp: wp.array(dtype=wp.vec2),
                              grid_d: wp.array(dtype=wp.float32),
                              grid_alpha: wp.array(dtype=wp.float32),
                              dx: wp.float32,
                              n_grid_x: wp.int32,
                              gc_d: wp.float32,
                              ell_d: wp.float32,
                              gc_a: wp.float32,
                              ell_a: wp.float32,
                              omega_a: wp.mat22,
                              M2: wp.mat33,
                              grid_Fd: wp.array(dtype=wp.float32),
                              grid_Fa: wp.array(dtype=wp.float32),
                              crack_energy_d: wp.array(dtype=wp.float32),
                              crack_energy_a: wp.array(dtype=wp.float32)):
    """Gather grad d, grad alpha and the interpolated grid fields d~, alpha~ in one pass, then
    scatter the two residuals F^d_I and F^alpha_I.

    Also writes the particle's crack surface energies from the grid fields at the start of the
    step, as the reference does:
      Gamma_d = gc_d V0 (d~^2 + l_d^2 |grad d|^2) / (2 l_d)
      Gamma_a = gc_a V0 (a~^2 + l_a^2 grad a . omega_a grad a) / (2 l_a)
    """
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    grad_d = wp.vec2(0.0, 0.0)
    grad_a = wp.vec2(0.0, 0.0)
    d_til = wp.float32(0.0)
    a_til = wp.float32(0.0)
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            dI = grid_d[node]
            aI = grid_alpha[node]
            grad_d += grad_w * dI
            grad_a += grad_w * aI
            d_til += w * dI
            a_til += w * aI

    V0 = vol0[p]
    d_p = d[p]
    a_p = alpha[p]
    eps_p = eps[p]
    omega_grad_a = omega_a * grad_a
    G_a = (1.0 - a_p) * (1.0 - d_p) * (1.0 - d_p) * wp.dot(eps_p, M2 * eps_p)
    local_d = (gc_d / ell_d) * d_p - 2.0 * (1.0 - d_p) * H[p]
    local_a = (gc_a / ell_a) * a_p - G_a
    flux_d = gc_d * ell_d * grad_d
    flux_a = gc_a * ell_a * omega_grad_a
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            wp.atomic_add(grid_Fd, node, (local_d * w + wp.dot(grad_w, flux_d)) * V0)
            wp.atomic_add(grid_Fa, node, (local_a * w + wp.dot(grad_w, flux_a)) * V0)

    crack_energy_d[p] = gc_d * V0 * (d_til * d_til + ell_d * ell_d * wp.dot(grad_d, grad_d)) \
        / (2.0 * ell_d)
    crack_energy_a[p] = gc_a * V0 * (a_til * a_til + ell_a * ell_a * wp.dot(grad_a, omega_grad_a)) \
        / (2.0 * ell_a)


@wp.kernel
def grid_update_aniso(grid_m: wp.array(dtype=wp.float32),
                      grid_mv: wp.array(dtype=wp.vec2),
                      grid_f: wp.array(dtype=wp.vec2),
                      grid_fext: wp.array(dtype=wp.vec2),
                      grid_pseudo_mass: wp.array(dtype=wp.float32),
                      grid_Fd: wp.array(dtype=wp.float32),
                      grid_Fa: wp.array(dtype=wp.float32),
                      dt: wp.float32,
                      inv_eta_d: wp.float32,
                      inv_eta_a: wp.float32,
                      step: wp.int32,
                      grid_d: wp.array(dtype=wp.float32),
                      grid_alpha: wp.array(dtype=wp.float32),
                      grid_v_old: wp.array(dtype=wp.vec2),
                      grid_v_new: wp.array(dtype=wp.vec2),
                      w_ext: wp.array(dtype=wp.float32)):
    """Momentum update (no Dirichlet BCs) and explicit forward-Euler updates of both grid damage
    fields on the shared pseudo-mass, in one pass over the nodes.

    The external work increment dt f_ext . v_I (post-update velocity) is summed into
    w_ext[step] on the device.
    """
    n = wp.tid()
    m = grid_m[n]
    if m > GRID_MASS_EPS_SI:
        mv = grid_mv[n]
        v_new = (mv + dt * grid_f[n]) / m
        grid_v_old[n] = mv / m
        grid_v_new[n] = v_new
        f_ext = grid_fext[n]
        if f_ext[0] != 0.0 or f_ext[1] != 0.0:
            wp.atomic_add(w_ext, step, dt * wp.dot(f_ext, v_new))
    else:
        grid_v_old[n] = wp.vec2(0.0, 0.0)
        grid_v_new[n] = wp.vec2(0.0, 0.0)

    m_pm = grid_pseudo_mass[n]
    if m_pm > PSEUDO_MASS_EPS_SI:
        grid_d[n] = grid_d[n] - dt * inv_eta_d * grid_Fd[n] / m_pm
        grid_alpha[n] = grid_alpha[n] - dt * inv_eta_a * grid_Fa[n] / m_pm


@wp.kernel
def g2p_aniso(x: wp.array(dtype=wp.vec2),
              v: wp.array(dtype=wp.vec2),
              F: wp.array(dtype=wp.mat33),
              L: wp.array(dtype=wp.mat22),
              d: wp.array(dtype=wp.float32),
              alpha: wp.array(dtype=wp.float32),
              vol0: wp.array(dtype=wp.float32),
              lp: wp.array(dtype=wp.vec2),
              grid_v_old: wp.array(dtype=wp.vec2),
              grid_v_new: wp.array(dtype=wp.vec2),
              grid_d: wp.array(dtype=wp.float32),
              grid_alpha: wp.array(dtype=wp.float32),
              dx: wp.float32,
              n_grid_x: wp.int32,
              dt: wp.float32,
              eta_d: wp.float32,
              eta_a: wp.float32,
              step: wp.int32,
              d_max: wp.array(dtype=wp.float32),
              alpha_max: wp.array(dtype=wp.float32),
              d_visc: wp.array(dtype=wp.float32)):
    """Pure FLIP velocity, x += dt sum w v_I, L and F update, and both particle fields
    field_p = max(field_p, clamp(sum w field_I, 0, 1)).

    max d_p and max alpha_p go into d_max[step] and alpha_max[step], and the viscous dissipation
    V0 [eta_d (dd)^2 + eta_a (da)^2] / dt of the clamped increments into d_visc[step], all on the
    device.
    """
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    v_pic = wp.vec2(0.0, 0.0)
    dv_flip = wp.vec2(0.0, 0.0)
    Lp = wp.mat22(0.0)
    d_grid = wp.float32(0.0)
    a_grid = wp.float32(0.0)
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            vn = grid_v_new[node]
            v_pic += w * vn
            dv_flip += w * (vn - grid_v_old[node])
            Lp += wp.outer(vn, grad_w)
            d_grid += w * grid_d[node]
            a_grid += w * grid_alpha[node]

    v[p] = v[p] + dv_flip
    x[p] = xp + dt * v_pic
    L[p] = Lp

    dF = wp.mat33(1.0 + dt * Lp[0, 0], dt * Lp[0, 1], 0.0,
                  dt * Lp[1, 0], 1.0 + dt * Lp[1, 1], 0.0,
                  0.0, 0.0, 1.0)
    F[p] = dF * F[p]

    d_old = d[p]
    d_new = wp.max(d_old, wp.clamp(d_grid, 0.0, 1.0))
    d[p] = d_new
    a_old = alpha[p]
    a_new = wp.max(a_old, wp.clamp(a_grid, 0.0, 1.0))
    alpha[p] = a_new
    wp.atomic_max(d_max, step, d_new)
    wp.atomic_max(alpha_max, step, a_new)
    dd = d_new - d_old
    da = a_new - a_old
    if dd > 0.0 or da > 0.0:
        wp.atomic_add(d_visc, step, vol0[p] * (eta_d * dd * dd + eta_a * da * da) / dt)


@wp.kernel
def stress_update_aniso(F: wp.array(dtype=wp.mat33),
                        L: wp.array(dtype=wp.mat22),
                        d: wp.array(dtype=wp.float32),
                        alpha: wp.array(dtype=wp.float32),
                        CA1: wp.mat33,
                        M2: wp.mat33,
                        dt: wp.float32,
                        eps: wp.array(dtype=wp.vec3),
                        sigma0: wp.array(dtype=wp.vec3),
                        sigma: wp.array(dtype=wp.vec3),
                        H: wp.array(dtype=wp.float32),
                        J: wp.array(dtype=wp.float32)):
    """Total-strain small-strain update (Voigt, engineering shear):

      eps += sym(L) dt,  sigma0 = C_a(alpha) : eps,  sigma = (1 - d)^2 sigma0  (no kappa),
      H = max(H, 1/2 eps : sigma0),  J = det F.

    sigma0 is recomputed from the total strain every step because C_a depends on alpha.
    """
    p = wp.tid()
    Lp = L[p]
    eps_p = eps[p] + wp.vec3(dt * Lp[0, 0], dt * Lp[1, 1], dt * (Lp[0, 1] + Lp[1, 0]))
    one_a = 1.0 - alpha[p]
    one_d = 1.0 - d[p]
    s0 = (CA1 + M2 * (one_a * one_a)) * eps_p
    eps[p] = eps_p
    sigma0[p] = s0
    sigma[p] = (one_d * one_d) * s0

    psi0 = 0.5 * wp.dot(s0, eps_p)
    if psi0 > H[p]:
        H[p] = psi0

    J[p] = wp.determinant(F[p])


def run_warp_aniso_solver(mesh, particles, CA1, M2, omega_a, gc_d, ell_d, gc_alpha, ell_alpha,
                          eta_d, eta_alpha, dtime, time, traction, lp0,
                          vtk_output_dir, vtk_interval):
    """Explicit USL MPM with bulk and interfacial damage of an anisotropic layered elastic body
    under nominal traction, entirely on the device. SI units, plane stress.

    mesh         — background grid (src.mesh2D.Mesh), square cells, origin at (0, 0). Particle
                   domains are clamped to [dx, L - dx] in both directions (GeoWarp style), so the
                   material should sit inside a pad ring.
    particles    — seeded src.particle.ParticleSet (positions, velocities, mass, volume, F = I).
                   Its positions, velocities and deformation_gradient are overwritten with the
                   final state at the end of the run.
    CA1, M2      — (3, 3) global-frame stiffness parts, C_a(alpha) = CA1 + M2 (1 - alpha)^2.
    omega_a      — (2, 2) anisotropic gradient tensor I + xi P_a.
    gc_d, ell_d, gc_alpha, ell_alpha, eta_d, eta_alpha
                 — toughness, length scale and viscosity of each damage field.
    traction     — (n, 2) nominal traction per particle, (0, 0) on unloaded particles.
    lp0          — (n, 2) initial particle-domain half-widths.

    Both damage fields and the history field start at zero. Output (VTK, energy budget) happens
    at t = 0 and every `vtk_interval` steps, plus the last step. Returns a dict in the style of
    run_warp_pf_solver.
    """
    if not np.isclose(mesh.deltax, mesh.deltay):
        raise ValueError("cpGIMP needs square background cells")

    dx = float(mesh.deltax)
    n_grid_x, n_grid_y = mesh.numx, mesh.numy
    n_nodes = (n_grid_x + 1) * (n_grid_y + 1)
    n_p = particles.count
    nsteps = int(np.floor(time / dtime))
    Lx, Ly = n_grid_x * dx, n_grid_y * dx

    f32 = np.float32
    F0 = np.zeros((n_p, 3, 3), f32)
    F0[:, 0, 0] = particles.deformation_gradient[:, 0]
    F0[:, 0, 1] = particles.deformation_gradient[:, 1]
    F0[:, 1, 0] = particles.deformation_gradient[:, 2]
    F0[:, 1, 1] = particles.deformation_gradient[:, 3]
    F0[:, 2, 2] = 1.0

    # ----- upload once -----
    x = wp.array(particles.positions.astype(f32), dtype=wp.vec2)
    v = wp.array(particles.velocities.astype(f32), dtype=wp.vec2)
    mass = wp.array(particles.mass.astype(f32), dtype=wp.float32)
    vol0 = wp.array(particles.volume.astype(f32), dtype=wp.float32)
    t_p = wp.array(np.asarray(traction, f32), dtype=wp.vec2)
    F = wp.array(F0, dtype=wp.mat33)
    lp0_d = wp.array(np.asarray(lp0, f32), dtype=wp.vec2)
    lp = wp.zeros(n_p, dtype=wp.vec2)
    L = wp.zeros(n_p, dtype=wp.mat22)
    J = wp.ones(n_p, dtype=wp.float32)
    eps = wp.zeros(n_p, dtype=wp.vec3)
    sigma0 = wp.zeros(n_p, dtype=wp.vec3)
    sigma = wp.zeros(n_p, dtype=wp.vec3)
    d = wp.zeros(n_p, dtype=wp.float32)
    alpha = wp.zeros(n_p, dtype=wp.float32)
    H = wp.zeros(n_p, dtype=wp.float32)
    crack_energy_d = wp.zeros(n_p, dtype=wp.float32)
    crack_energy_a = wp.zeros(n_p, dtype=wp.float32)

    grid_m = wp.zeros(n_nodes, dtype=wp.float32)
    grid_mv = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_f = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_fext = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_pseudo_mass = wp.zeros(n_nodes, dtype=wp.float32)
    grid_dnum = wp.zeros(n_nodes, dtype=wp.float32)
    grid_anum = wp.zeros(n_nodes, dtype=wp.float32)
    grid_Fd = wp.zeros(n_nodes, dtype=wp.float32)
    grid_Fa = wp.zeros(n_nodes, dtype=wp.float32)
    grid_d = wp.zeros(n_nodes, dtype=wp.float32)
    grid_alpha = wp.zeros(n_nodes, dtype=wp.float32)
    grid_v_old = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_v_new = wp.zeros(n_nodes, dtype=wp.vec2)

    d_max = wp.zeros(nsteps, dtype=wp.float32)
    alpha_max = wp.zeros(nsteps, dtype=wp.float32)
    w_ext = wp.zeros(nsteps, dtype=wp.float32)
    d_visc = wp.zeros(nsteps, dtype=wp.float32)

    CA1_d = wp.mat33(*np.asarray(CA1, f32).ravel())
    M2_d = wp.mat33(*np.asarray(M2, f32).ravel())
    omega_d = wp.mat22(*np.asarray(omega_a, f32).ravel())
    gc_d_f, ell_d_f, gc_a_f, ell_a_f = f32(gc_d), f32(ell_d), f32(gc_alpha), f32(ell_alpha)
    eta_d_f, eta_a_f = f32(eta_d), f32(eta_alpha)
    inv_eta_d, inv_eta_a = f32(1.0 / eta_d), f32(1.0 / eta_alpha)
    dt_f, dx_f = f32(dtime), f32(dx)
    clamp = (f32(dx), f32(Lx - dx), f32(dx), f32(Ly - dx))

    wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])

    mass_h = particles.mass
    vol0_h = particles.volume
    vtk_entries = []
    out_steps, kinetic, elastic, crack_bulk, crack_interf = [], [], [], [], []

    def output(step):
        t = step * dtime
        xh, vh = x.numpy().astype(np.float64), v.numpy().astype(np.float64)
        sh, epsh = sigma.numpy().astype(np.float64), eps.numpy().astype(np.float64)
        out_steps.append(step)
        kinetic.append(0.5 * np.sum(mass_h * np.sum(vh ** 2, axis=1)))
        elastic.append(0.5 * np.sum(vol0_h * np.sum(sh * epsh, axis=1)))
        crack_bulk.append(np.sum(crack_energy_d.numpy().astype(np.float64)))
        crack_interf.append(np.sum(crack_energy_a.numpy().astype(np.float64)))

        if vtk_output_dir is not None:
            fname = write_particles_vtp_aniso(xh, vh, sh, d.numpy(), alpha.numpy(), H.numpy(),
                                              step, t, vtk_output_dir)
            vtk_entries.append((t, fname))

    output(0)
    wp.synchronize()
    t_start = _time.perf_counter()

    for step in range(nsteps):
        wp.launch(reset_grid_aniso, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_dnum,
                          grid_anum, grid_Fd, grid_Fa])
        wp.launch(p2g_aniso, dim=n_p,
                  inputs=[x, v, mass, vol0, J, sigma, t_p, d, alpha, lp, dx_f, n_grid_x,
                          grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_dnum,
                          grid_anum])
        wp.launch(grid_two_field_project, dim=n_nodes,
                  inputs=[grid_pseudo_mass, grid_dnum, grid_anum, grid_d, grid_alpha])
        wp.launch(damage_residual_two_field, dim=n_p,
                  inputs=[x, vol0, d, alpha, H, eps, lp, grid_d, grid_alpha, dx_f, n_grid_x,
                          gc_d_f, ell_d_f, gc_a_f, ell_a_f, omega_d, M2_d,
                          grid_Fd, grid_Fa, crack_energy_d, crack_energy_a])
        wp.launch(grid_update_aniso, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_Fd, grid_Fa,
                          dt_f, inv_eta_d, inv_eta_a, step, grid_d, grid_alpha,
                          grid_v_old, grid_v_new, w_ext])
        wp.launch(g2p_aniso, dim=n_p,
                  inputs=[x, v, F, L, d, alpha, vol0, lp, grid_v_old, grid_v_new, grid_d,
                          grid_alpha, dx_f, n_grid_x, dt_f, eta_d_f, eta_a_f, step,
                          d_max, alpha_max, d_visc])
        wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])
        wp.launch(stress_update_aniso, dim=n_p,
                  inputs=[F, L, d, alpha, CA1_d, M2_d, dt_f, eps, sigma0, sigma, H, J])

        if (step + 1) % vtk_interval == 0 or step + 1 == nsteps:
            output(step + 1)

    wp.synchronize()
    wall_time = _time.perf_counter() - t_start

    # ----- one final copy for post-processing -----
    Fh = F.numpy()
    particles.positions = x.numpy().astype(np.float64)
    particles.velocities = v.numpy().astype(np.float64)
    particles.deformation_gradient = np.column_stack(
        [Fh[:, 0, 0], Fh[:, 0, 1], Fh[:, 1, 0], Fh[:, 1, 1]]).astype(np.float64)

    out_steps = np.array(out_steps)
    w_ext_cum = np.concatenate([[0.0], np.cumsum(w_ext.numpy().astype(np.float64))])[out_steps]
    d_visc_cum = np.concatenate([[0.0], np.cumsum(d_visc.numpy().astype(np.float64))])[out_steps]
    kinetic, elastic = np.array(kinetic), np.array(elastic)
    crack_bulk, crack_interf = np.array(crack_bulk), np.array(crack_interf)

    return {
        'vtk_entries':   vtk_entries,
        'energy_times':  out_steps * dtime,
        'kinetic':       kinetic,
        'elastic':       elastic,
        'crack_bulk':    crack_bulk,
        'crack_interf':  crack_interf,
        'visc':          d_visc_cum,
        'w_ext':         w_ext_cum,
        'miss':          kinetic + elastic + crack_bulk + crack_interf + d_visc_cum - w_ext_cum,
        'step_times':    (np.arange(nsteps) + 1) * dtime,
        'd_max':         d_max.numpy().astype(np.float64),
        'alpha_max':     alpha_max.numpy().astype(np.float64),
        'bulk_damage':   d.numpy().astype(np.float64),
        'interf_damage': alpha.numpy().astype(np.float64),
        'history':       H.numpy().astype(np.float64),
        'nsteps':        nsteps,
        'wall_time':     wall_time,
    }
