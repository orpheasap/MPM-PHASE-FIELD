# Explicit USL MPM with AT2 bulk damage on the GPU: cpGIMP basis, small-strain hypoelasticity,
# nominal traction, pure FLIP. Port of the root pre_noched_plate_PF.py (Algorithm 1).
#
# One step:
#   reset_grid_pf -> p2g_pf (mass, momentum, degraded internal force, traction, pseudo-mass,
#   phase numerator) -> grid_phase_project -> damage_residual -> grid_update_pf (momentum and
#   grid phase) -> g2p_pf (FLIP, x, L, F, irreversible phase) -> convected particle domain
#   update -> stress_update_pf.
# Kernel conventions: docs/parallel.md. State stays on the device; the host reads it back only
# on output steps. The phi_max, external-work and viscous-dissipation histories are
# accumulated per step on the device and read once at the end.
import time as _time

import numpy as np
import warp as wp

from .warp_gimp import gimp_stencil_origin, gimp_weights_gradients_2d, update_gimp_lp_2d
from .warp_phasefield import PHI_MASS_EPS, damage_residual, grid_phase_project
from .warp_small_strain import stress_update_pf
from .vtk_export import write_particles_vtp_pf

# Nodes lighter than this carry no momentum (tonne per unit thickness), the reference's tol.
GRID_MASS_EPS_PF = wp.constant(1.0e-20)

# A particle with more bulk damage than this counts as cracked for the crack-tip samples.
CRACKED_PHI = 0.9


@wp.kernel
def reset_grid_pf(grid_m: wp.array(dtype=wp.float32),
                  grid_mv: wp.array(dtype=wp.vec2),
                  grid_f: wp.array(dtype=wp.vec2),
                  grid_fext: wp.array(dtype=wp.vec2),
                  grid_pseudo_mass: wp.array(dtype=wp.float32),
                  grid_phinum: wp.array(dtype=wp.float32),
                  grid_F: wp.array(dtype=wp.float32)):
    n = wp.tid()
    grid_m[n] = 0.0
    grid_mv[n] = wp.vec2(0.0, 0.0)
    grid_f[n] = wp.vec2(0.0, 0.0)
    grid_fext[n] = wp.vec2(0.0, 0.0)
    grid_pseudo_mass[n] = 0.0
    grid_phinum[n] = 0.0
    grid_F[n] = 0.0


@wp.kernel
def p2g_pf(x: wp.array(dtype=wp.vec2),
           v: wp.array(dtype=wp.vec2),
           mass: wp.array(dtype=wp.float32),
           vol0: wp.array(dtype=wp.float32),
           J: wp.array(dtype=wp.float32),
           sigma: wp.array(dtype=wp.vec3),
           traction: wp.array(dtype=wp.vec2),
           phi: wp.array(dtype=wp.float32),
           lp: wp.array(dtype=wp.vec2),
           dx: wp.float32,
           n_grid_x: wp.int32,
           grid_m: wp.array(dtype=wp.float32),
           grid_mv: wp.array(dtype=wp.vec2),
           grid_f: wp.array(dtype=wp.vec2),
           grid_fext: wp.array(dtype=wp.vec2),
           grid_pseudo_mass: wp.array(dtype=wp.float32),
           grid_phinum: wp.array(dtype=wp.float32)):
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
    phi_V0 = phi[p] * V0

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
            wp.atomic_add(grid_phinum, node, w * phi_V0)


@wp.kernel
def grid_update_pf(grid_m: wp.array(dtype=wp.float32),
                   grid_mv: wp.array(dtype=wp.vec2),
                   grid_f: wp.array(dtype=wp.vec2),
                   grid_fext: wp.array(dtype=wp.vec2),
                   grid_pseudo_mass: wp.array(dtype=wp.float32),
                   grid_F: wp.array(dtype=wp.float32),
                   dt: wp.float32,
                   inv_eta: wp.float32,
                   step: wp.int32,
                   grid_phi: wp.array(dtype=wp.float32),
                   grid_v_old: wp.array(dtype=wp.vec2),
                   grid_v_new: wp.array(dtype=wp.vec2),
                   w_ext: wp.array(dtype=wp.float32)):
    """Momentum update (no Dirichlet BCs) and explicit grid phase update
    phi_I <- phi_I - dt/(eta m_I^phi) F_I, in one pass over the nodes.

    The external work increment dt f_ext . v_I (post-update velocity) is summed into
    w_ext[step] on the device.
    """
    n = wp.tid()
    m = grid_m[n]
    if m > GRID_MASS_EPS_PF:
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

    m_phi = grid_pseudo_mass[n]
    if m_phi > PHI_MASS_EPS:
        grid_phi[n] = grid_phi[n] - dt * inv_eta * grid_F[n] / m_phi


@wp.kernel
def g2p_pf(x: wp.array(dtype=wp.vec2),
           v: wp.array(dtype=wp.vec2),
           F: wp.array(dtype=wp.mat33),
           L: wp.array(dtype=wp.mat22),
           phi: wp.array(dtype=wp.float32),
           vol0: wp.array(dtype=wp.float32),
           lp: wp.array(dtype=wp.vec2),
           grid_v_old: wp.array(dtype=wp.vec2),
           grid_v_new: wp.array(dtype=wp.vec2),
           grid_phi: wp.array(dtype=wp.float32),
           dx: wp.float32,
           n_grid_x: wp.int32,
           dt: wp.float32,
           eta: wp.float32,
           step: wp.int32,
           phi_max: wp.array(dtype=wp.float32),
           d_visc: wp.array(dtype=wp.float32)):
    """Pure FLIP velocity, x += dt sum w v_I, L and F update, and the particle phase
    phi_p = max(phi_p, clamp(sum w phi_I, 0, 1)).

    max phi_p goes into phi_max[step] and the viscous dissipation V0 eta (d phi_p)^2 / dt of the
    clamped increment into d_visc[step], both on the device.
    """
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    v_pic = wp.vec2(0.0, 0.0)
    dv_flip = wp.vec2(0.0, 0.0)
    Lp = wp.mat22(0.0)
    phi_grid = wp.float32(0.0)
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            vn = grid_v_new[node]
            v_pic += w * vn
            dv_flip += w * (vn - grid_v_old[node])
            Lp += wp.outer(vn, grad_w)
            phi_grid += w * grid_phi[node]

    v[p] = v[p] + dv_flip
    x[p] = xp + dt * v_pic
    L[p] = Lp

    dF = wp.mat33(1.0 + dt * Lp[0, 0], dt * Lp[0, 1], 0.0,
                  dt * Lp[1, 0], 1.0 + dt * Lp[1, 1], 0.0,
                  0.0, 0.0, 1.0)
    F[p] = dF * F[p]

    phi_old = phi[p]
    phi_new = wp.max(phi_old, wp.clamp(phi_grid, 0.0, 1.0))
    phi[p] = phi_new
    wp.atomic_max(phi_max, step, phi_new)
    d_phi = phi_new - phi_old
    if d_phi > 0.0:
        wp.atomic_add(d_visc, step, vol0[p] * eta * d_phi * d_phi / dt)


def run_warp_pf_solver(mesh, particles, material, pf, dtime, time, traction, lp0,
                       vtk_output_dir, vtk_interval, crack_origin, crack_line_y, crack_band):
    """Explicit USL MPM with AT2 bulk damage of an elastic body under nominal traction,
    entirely on the device.

    mesh         — background grid (src.mesh2D.Mesh), square cells, origin at (0, 0). Particle
                   domains are clamped to [dx, L - dx] in both directions (GeoWarp style), so the
                   material should sit inside a pad ring.
    particles    — seeded src.particle.ParticleSet (positions, velocities, mass, volume, F = I).
                   Its positions, velocities and deformation_gradient are overwritten with the
                   final state at the end of the run.
    material     — src.warp_small_strain.PlaneStrainMaterial.
    pf           — src.phasefield.PhaseFieldModel; only AT2 is implemented on the device.
    traction     — (n, 2) nominal traction per particle, (0, 0) on unloaded particles.
    lp0          — (n, 2) initial particle-domain half-widths.
    crack_origin, crack_line_y, crack_band
                 — crack-tip samples: x_tip is the largest x - crack_origin[0] over particles
                   with phi > 0.9 and |y - crack_origin[1] - crack_line_y| < crack_band.

    Bulk damage and the history field start at zero. Output (VTK, energy budget, crack tip)
    happens at t = 0 and every `vtk_interval` steps, plus the last step. Returns a dict in the
    style of run_warp_solver.
    """
    if not np.isclose(mesh.deltax, mesh.deltay):
        raise ValueError("cpGIMP needs square background cells")
    if pf.model != 'AT2':
        raise ValueError("the Warp bulk-damage kernels implement AT2 only")

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
    phi = wp.zeros(n_p, dtype=wp.float32)
    H = wp.zeros(n_p, dtype=wp.float32)
    crack_energy = wp.zeros(n_p, dtype=wp.float32)

    grid_m = wp.zeros(n_nodes, dtype=wp.float32)
    grid_mv = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_f = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_fext = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_pseudo_mass = wp.zeros(n_nodes, dtype=wp.float32)
    grid_phinum = wp.zeros(n_nodes, dtype=wp.float32)
    grid_F = wp.zeros(n_nodes, dtype=wp.float32)
    grid_phi = wp.zeros(n_nodes, dtype=wp.float32)
    grid_v_old = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_v_new = wp.zeros(n_nodes, dtype=wp.vec2)

    phi_max = wp.zeros(nsteps, dtype=wp.float32)
    w_ext = wp.zeros(nsteps, dtype=wp.float32)
    d_visc = wp.zeros(nsteps, dtype=wp.float32)

    C = wp.mat33(*np.asarray(material.elasticity_matrix, f32).ravel())
    kappa, eta, inv_eta = f32(pf.kappa), f32(pf.eta), f32(1.0 / pf.eta)
    crack_coeff, l2 = f32(pf.crack_force_coeff), f32(pf.ell ** 2)
    dt_f, dx_f = f32(dtime), f32(dx)
    clamp = (f32(dx), f32(Lx - dx), f32(dx), f32(Ly - dx))

    wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])

    mass_h = particles.mass
    vol0_h = particles.volume
    vtk_entries = []
    out_steps, kinetic, elastic, crack_bulk = [], [], [], []
    tip_x, tip_phi_max = [], []

    def output(step):
        t = step * dtime
        xh, vh = x.numpy().astype(np.float64), v.numpy().astype(np.float64)
        sh, epsh = sigma.numpy().astype(np.float64), eps.numpy().astype(np.float64)
        phih, Hh = phi.numpy(), H.numpy()
        out_steps.append(step)
        kinetic.append(0.5 * np.sum(mass_h * np.sum(vh ** 2, axis=1)))
        elastic.append(0.5 * np.sum(vol0_h * np.sum(sh * epsh, axis=1)))
        crack_bulk.append(np.sum(crack_energy.numpy().astype(np.float64)))

        xm = xh - np.asarray(crack_origin)
        cracked = (phih > CRACKED_PHI) & (np.abs(xm[:, 1] - crack_line_y) < crack_band)
        tip_x.append(xm[cracked, 0].max() if cracked.any() else np.nan)
        tip_phi_max.append(float(phih.max()))

        if vtk_output_dir is not None:
            fname = write_particles_vtp_pf(xh, vh, sh, material.nu, phih, Hh,
                                           step, t, vtk_output_dir)
            vtk_entries.append((t, fname))

    output(0)
    wp.synchronize()
    t_start = _time.perf_counter()

    for step in range(nsteps):
        wp.launch(reset_grid_pf, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_phinum, grid_F])
        wp.launch(p2g_pf, dim=n_p,
                  inputs=[x, v, mass, vol0, J, sigma, t_p, phi, lp, dx_f, n_grid_x,
                          grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_phinum])
        wp.launch(grid_phase_project, dim=n_nodes, inputs=[grid_pseudo_mass, grid_phinum, grid_phi])
        wp.launch(damage_residual, dim=n_p,
                  inputs=[x, vol0, phi, H, lp, grid_phi, dx_f, n_grid_x, crack_coeff, l2,
                          grid_F, crack_energy])
        wp.launch(grid_update_pf, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_F, dt_f, inv_eta,
                          step, grid_phi, grid_v_old, grid_v_new, w_ext])
        wp.launch(g2p_pf, dim=n_p,
                  inputs=[x, v, F, L, phi, vol0, lp, grid_v_old, grid_v_new, grid_phi, dx_f,
                          n_grid_x, dt_f, eta, step, phi_max, d_visc])
        wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])
        wp.launch(stress_update_pf, dim=n_p,
                  inputs=[F, L, phi, C, kappa, dt_f, eps, sigma0, sigma, H, J])

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
    kinetic, elastic, crack_bulk = np.array(kinetic), np.array(elastic), np.array(crack_bulk)

    return {
        'vtk_entries':  vtk_entries,
        'energy_times': out_steps * dtime,
        'kinetic':      kinetic,
        'elastic':      elastic,
        'crack_bulk':   crack_bulk,
        'visc':         d_visc_cum,
        'w_ext':        w_ext_cum,
        'miss':         kinetic + elastic + crack_bulk + d_visc_cum - w_ext_cum,
        'step_times':   (np.arange(nsteps) + 1) * dtime,
        'phi_max':      phi_max.numpy().astype(np.float64),
        'tip_times':    out_steps * dtime,
        'tip_x':        np.array(tip_x),
        'tip_phi_max':  np.array(tip_phi_max),
        'phase':        phi.numpy().astype(np.float64),
        'history':      H.numpy().astype(np.float64),
        'nsteps':       nsteps,
        'wall_time':    wall_time,
    }
