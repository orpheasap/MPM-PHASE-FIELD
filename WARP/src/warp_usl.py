# Explicit USL MPM on the GPU with NVIDIA Warp: cpGIMP basis, Hencky elasticity, rigid wall.
#
# One step: reset grid -> P2G -> grid momentum update + rigid wall -> G2P (FLIP/PIC blend,
# position and F update) -> convected particle domain update -> stress update.
# Kernel conventions: docs/parallel.md. State stays on the device; the host reads it back only
# on output steps.
import time as _time

import numpy as np
import warp as wp

from .warp_gimp import gimp_stencil_origin, gimp_weights_gradients_2d, update_gimp_lp_2d
from .warp_hencky import hencky_cauchy_stress, hencky_energy_density
from .vtk_export import write_particles_vtp_elastic

# Nodes lighter than this carry no momentum (kg per unit thickness).
GRID_MASS_EPS = wp.constant(1.0e-12)


@wp.kernel
def reset_grid(grid_m: wp.array(dtype=wp.float32),
               grid_mv: wp.array(dtype=wp.vec2),
               grid_f: wp.array(dtype=wp.vec2)):
    n = wp.tid()
    grid_m[n] = 0.0
    grid_mv[n] = wp.vec2(0.0, 0.0)
    grid_f[n] = wp.vec2(0.0, 0.0)


@wp.kernel
def p2g(x: wp.array(dtype=wp.vec2),
        v: wp.array(dtype=wp.vec2),
        mass: wp.array(dtype=wp.float32),
        vol0: wp.array(dtype=wp.float32),
        J: wp.array(dtype=wp.float32),
        sigma: wp.array(dtype=wp.mat33),
        lp: wp.array(dtype=wp.vec2),
        dx: wp.float32,
        n_grid_x: wp.int32,
        grid_m: wp.array(dtype=wp.float32),
        grid_mv: wp.array(dtype=wp.vec2),
        grid_f: wp.array(dtype=wp.vec2)):
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    mp = mass[p]
    mvp = mp * v[p]
    Vp = vol0[p] * J[p]
    sp = sigma[p]
    s2 = wp.mat22(sp[0, 0], sp[0, 1], sp[1, 0], sp[1, 1])

    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            wp.atomic_add(grid_m, node, w * mp)
            wp.atomic_add(grid_mv, node, w * mvp)
            wp.atomic_add(grid_f, node, -Vp * (s2 * grad_w))


@wp.kernel
def grid_update_rigid_wall(grid_m: wp.array(dtype=wp.float32),
                           grid_mv: wp.array(dtype=wp.vec2),
                           grid_f: wp.array(dtype=wp.vec2),
                           dt: wp.float32,
                           n_grid_x: wp.int32,
                           wall_row: wp.int32,
                           step: wp.int32,
                           grid_v_old: wp.array(dtype=wp.vec2),
                           grid_v_new: wp.array(dtype=wp.vec2),
                           wall_reaction: wp.array(dtype=wp.float32)):
    """Momentum update, then the frictionless one-sided rigid wall on node row `wall_row`.

    The momentum the wall removes, divided by dt, is the wall reaction (N per unit thickness);
    it is summed into wall_reaction[step] on the device.
    """
    n = wp.tid()
    m = grid_m[n]
    if m <= GRID_MASS_EPS:
        grid_v_old[n] = wp.vec2(0.0, 0.0)
        grid_v_new[n] = wp.vec2(0.0, 0.0)
        return

    mv = grid_mv[n]
    mv_new = mv + dt * grid_f[n]
    if n // (n_grid_x + 1) == wall_row and mv_new[1] < 0.0:
        wp.atomic_add(wall_reaction, step, -mv_new[1] / dt)
        mv_new = wp.vec2(mv_new[0], 0.0)

    grid_v_old[n] = mv / m
    grid_v_new[n] = mv_new / m


@wp.kernel
def g2p(x: wp.array(dtype=wp.vec2),
        v: wp.array(dtype=wp.vec2),
        F: wp.array(dtype=wp.mat33),
        lp: wp.array(dtype=wp.vec2),
        grid_v_old: wp.array(dtype=wp.vec2),
        grid_v_new: wp.array(dtype=wp.vec2),
        dx: wp.float32,
        n_grid_x: wp.int32,
        dt: wp.float32,
        alpha: wp.float32):
    p = wp.tid()
    inv_dx = 1.0 / dx
    xp = x[p]
    s = gimp_weights_gradients_2d(xp, dx, inv_dx, lp[p][0], lp[p][1])
    origin = gimp_stencil_origin(xp, inv_dx, lp[p])

    v_pic = wp.vec2(0.0, 0.0)
    dv_flip = wp.vec2(0.0, 0.0)
    L = wp.mat22(0.0)
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            grad_w = wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
            node = (origin[0] + i) + (origin[1] + j) * (n_grid_x + 1)
            vn = grid_v_new[node]
            v_pic += w * vn
            dv_flip += w * (vn - grid_v_old[node])
            L += wp.outer(vn, grad_w)

    v[p] = alpha * (v[p] + dv_flip) + (1.0 - alpha) * v_pic
    x[p] = xp + dt * v_pic

    dF = wp.mat33(1.0 + dt * L[0, 0], dt * L[0, 1], 0.0,
                  dt * L[1, 0], 1.0 + dt * L[1, 1], 0.0,
                  0.0, 0.0, 1.0)
    F[p] = dF * F[p]


@wp.kernel
def stress_update(F: wp.array(dtype=wp.mat33),
                  lam: wp.float32,
                  mu: wp.float32,
                  sigma: wp.array(dtype=wp.mat33),
                  J: wp.array(dtype=wp.float32),
                  psi: wp.array(dtype=wp.float32)):
    p = wp.tid()
    Fp = F[p]
    sigma[p] = hencky_cauchy_stress(Fp, lam, mu)
    J[p] = wp.determinant(Fp)
    psi[p] = hencky_energy_density(Fp, lam, mu)


def run_warp_solver(mesh, particles, material, dtime, time, alpha,
                    vtk_output_dir, vtk_interval, track_pids, lp0):
    """Explicit USL MPM of an elastic body against a rigid wall, entirely on the device.

    mesh       — background grid (src.mesh2D.Mesh), square cells, origin at (0, 0). Cantilever-style
                 layout: the rigid wall is the node row at y = dx, the row below it is a pad ring,
                 and particle domains are clamped to [dx, L - dx] in both directions.
    particles  — seeded src.particle.ParticleSet (positions, velocities, mass, volume, F = I).
                 Its positions, velocities and deformation_gradient are overwritten with the
                 final state at the end of the run.
    material   — src.warp_hencky.ElasticMaterial.
    lp0        — (n, 2) initial particle-domain half-widths.

    Output (VTK, tracked particles, energy) happens at t = 0 and every `vtk_interval` steps,
    plus the last step. Returns a dict of results.
    """
    if not np.isclose(mesh.deltax, mesh.deltay):
        raise ValueError("cpGIMP needs square background cells")

    dx = float(mesh.deltax)
    n_grid_x, n_grid_y = mesh.numx, mesh.numy
    n_nodes = (n_grid_x + 1) * (n_grid_y + 1)
    n_p = particles.count
    nsteps = int(np.floor(time / dtime))
    wall_row = 1
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
    F = wp.array(F0, dtype=wp.mat33)
    lp0_d = wp.array(np.asarray(lp0, f32), dtype=wp.vec2)
    lp = wp.zeros(n_p, dtype=wp.vec2)
    sigma = wp.zeros(n_p, dtype=wp.mat33)
    J = wp.zeros(n_p, dtype=wp.float32)
    psi = wp.zeros(n_p, dtype=wp.float32)

    grid_m = wp.zeros(n_nodes, dtype=wp.float32)
    grid_mv = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_f = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_v_old = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_v_new = wp.zeros(n_nodes, dtype=wp.vec2)
    wall_reaction = wp.zeros(nsteps, dtype=wp.float32)

    lam, mu = f32(material.lam), f32(material.mu)
    dt_f, dx_f, alpha_f = f32(dtime), f32(dx), f32(alpha)
    clamp = (f32(dx), f32(Lx - dx), f32(dx), f32(Ly - dx))

    def update_domain_and_stress():
        wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])
        wp.launch(stress_update, dim=n_p, inputs=[F, lam, mu, sigma, J, psi])

    update_domain_and_stress()

    mass_h = particles.mass
    vol0_h = particles.volume
    vtk_entries = []
    track_times, track_pos = [], {pid: [] for pid in (track_pids or [])}
    energy_times, kinetic, elastic = [], [], []

    def output(step):
        t = step * dtime
        xh, vh = x.numpy(), v.numpy()
        psih = psi.numpy()
        energy_times.append(t)
        kinetic.append(0.5 * np.sum(mass_h * np.sum(vh.astype(np.float64) ** 2, axis=1)))
        elastic.append(np.sum(vol0_h * psih))
        if vtk_output_dir is not None:
            fname = write_particles_vtp_elastic(xh, vh, sigma.numpy(), J.numpy(),
                                                step, t, vtk_output_dir)
            vtk_entries.append((t, fname))
        if track_pids is not None:
            track_times.append(t)
            for pid in track_pids:
                track_pos[pid].append(xh[pid].astype(np.float64))

    output(0)
    wp.synchronize()
    t_start = _time.perf_counter()

    for step in range(nsteps):
        wp.launch(reset_grid, dim=n_nodes, inputs=[grid_m, grid_mv, grid_f])
        wp.launch(p2g, dim=n_p,
                  inputs=[x, v, mass, vol0, J, sigma, lp, dx_f, n_grid_x, grid_m, grid_mv, grid_f])
        wp.launch(grid_update_rigid_wall, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, dt_f, n_grid_x, wall_row, step,
                          grid_v_old, grid_v_new, wall_reaction])
        wp.launch(g2p, dim=n_p,
                  inputs=[x, v, F, lp, grid_v_old, grid_v_new, dx_f, n_grid_x, dt_f, alpha_f])
        update_domain_and_stress()

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

    kinetic, elastic = np.array(kinetic), np.array(elastic)
    result = {
        'vtk_entries':   vtk_entries,
        'energy_times':  np.array(energy_times),
        'kinetic':       kinetic,
        'elastic':       elastic,
        'total':         kinetic + elastic,
        'wall_times':    (np.arange(nsteps) + 1) * dtime,
        'wall_reaction': wall_reaction.numpy().astype(np.float64),
        'nsteps':        nsteps,
        'wall_time':     wall_time,
        'J':             J.numpy().astype(np.float64),
    }
    if track_pids is not None:
        result['track_times'] = np.array(track_times)
        result['track_positions'] = {pid: np.array(pos) for pid, pos in track_pos.items()}
    return result
