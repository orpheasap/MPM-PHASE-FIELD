"""
3D MUSL-EP solver — Johnson-Cook plasticity + Mie-Grüneisen EOS, no damage.

This is the full elasto-plastic formulation from README.md (steps 22-34),
ported to 3D from the 2D plane-strain solver in solver1.py. The only physics
difference vs solver1 is that the deviatoric split and von Mises contraction
are now done with the true out-of-plane component (d_zz, s_zz) carried
explicitly, rather than reconstructed from the plane-strain traceless
condition.

Grid bookkeeping (NodeState3D, trilinear shape functions, particle-element
map) is shared with the linear-elastic solver in solver3d.py.

Stress is stored in Voigt form: [xx, yy, zz, xy, yz, xz].
Damage is held at D = 0 throughout (step 33 omitted, matching solver1).
"""
import numpy as np
from src.vtk_export import write_particles_vtp_3d
from src.solver3d import NodeState3D, get_mpm3d_shape, build_particle_element_map


# ── Constitutive helpers ──────────────────────────────────────────────────────

def _von_mises_dev(s):
    """Von Mises equivalent stress from a 3D deviatoric stress in Voigt form
    s = [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz].
        sigma_eq = sqrt(3/2 * s:s)
    """
    return np.sqrt(1.5 * (
        s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
    ))


def _jc_flow_stress(mat, eps_p, eps_dot):
    """Johnson-Cook flow stress — no thermal/damage factor (README 29d)."""
    eps_dot_star = max(eps_dot / mat.eps_dot_0, 1.0)
    return (mat.A + mat.B * max(eps_p, 0.0) ** mat.n) * (1.0 + mat.C * np.log(eps_dot_star))


def _mg_pressure(mat, detF, e):
    """Mie-Grüneisen EOS — README step 30 (D = 0).

    eta   = rho/rho0 = 1/detF
    mu    = eta - 1                       (positive = compression)
    -p    = rho0*c0^2 * mu*(eta - Gamma0/2*mu)/(eta - S_alpha*mu)^2 + Gamma0*e
    Returned value is p_hat (the quantity added to the deviatoric stress).
    """
    eta   = 1.0 / detF
    mu    = eta - 1.0
    denom = (eta - mat.S_alpha * mu) ** 2
    if denom < 1e-20:                      # guard against the Hugoniot limit
        return -mat.Gamma0 * e
    p_H = mat.initial_density * mat.c0**2 * mu * (eta - 0.5 * mat.Gamma0 * mu) / denom
    return -(p_H + mat.Gamma0 * e)


def _constitutive_update(mat, pid, particles, dtime, Lp):
    """3D MUSL-EP constitutive update — README steps 24-32 (no damage)."""
    I3 = np.eye(3)

    # 24: Deformation gradient
    F    = (I3 + Lp * dtime) @ particles.deformation_gradient[pid].reshape(3, 3)
    detF = np.linalg.det(F)
    particles.deformation_gradient[pid] = F.ravel()

    # 25: Volume
    particles.volume[pid] = detF * particles.initial_volume[pid]

    # 27: Strain-rate tensor and polar decomposition via SVD
    Dp         = 0.5 * (Lp + Lp.T)
    U_s, _, Vt = np.linalg.svd(F)
    R          = U_s @ Vt

    # 28: Un-rotated strain rate and its deviatoric part (true 3D)
    dp     = R.T @ Dp @ R
    d_vol  = (dp[0, 0] + dp[1, 1] + dp[2, 2]) / 3.0
    dp_dev = dp - d_vol * I3

    eps_dot = np.sqrt(2.0 / 3.0 * (
        dp_dev[0, 0]**2 + dp_dev[1, 1]**2 + dp_dev[2, 2]**2
        + 2.0 * (dp_dev[0, 1]**2 + dp_dev[1, 2]**2 + dp_dev[0, 2]**2)
    ))

    # 29a: No damage — undamaged shear modulus
    Gp = mat.G

    # 29b: Elastic trial deviatoric stress (Voigt)
    s       = particles.stress_dev[pid]
    d_voigt = np.array([dp_dev[0, 0], dp_dev[1, 1], dp_dev[2, 2],
                        dp_dev[0, 1], dp_dev[1, 2], dp_dev[0, 2]])
    s_trial = s + 2.0 * Gp * dtime * d_voigt

    # 29c: Trial von Mises stress
    sigma_trial_eq = _von_mises_dev(s_trial)

    # 29d: JC flow stress (no damage)
    eps_p   = particles.eps_p[pid]
    sigma_f = _jc_flow_stress(mat, eps_p, eps_dot)

    # 29e-f: Elastic or plastic radial return
    delta_eps_p = 0.0
    if sigma_trial_eq <= sigma_f:
        particles.stress_dev[pid] = s_trial
    else:
        delta_eps_p               = (sigma_trial_eq - sigma_f) / (3.0 * Gp)
        particles.eps_p[pid]      = eps_p + delta_eps_p
        particles.stress_dev[pid] = (sigma_f / sigma_trial_eq) * s_trial

    s_upd = particles.stress_dev[pid]

    # 30: Internal-energy increment (Taylor-Quinney) and Mie-Grüneisen pressure
    particles.e[pid] += detF * mat.chi * sigma_f * delta_eps_p
    p_hat = _mg_pressure(mat, detF, particles.e[pid])

    # 31: Assemble un-rotated stress and rotate back to the global frame
    sigma_unrot = np.array([
        [s_upd[0] + p_hat, s_upd[3],          s_upd[5]         ],
        [s_upd[3],          s_upd[1] + p_hat, s_upd[4]         ],
        [s_upd[5],          s_upd[4],          s_upd[2] + p_hat],
    ])
    sigma_rot = R @ sigma_unrot @ R.T
    particles.stress[pid] = np.array([
        sigma_rot[0, 0], sigma_rot[1, 1], sigma_rot[2, 2],
        sigma_rot[0, 1], sigma_rot[1, 2], sigma_rot[0, 2],
    ])
    # Step 33 (damage) omitted — this solver has no damage.


# ── Main solver ───────────────────────────────────────────────────────────────

def run_mpm_solver_3d_ep(mesh, particles, material,
                         g=9.81, dtime=1e-5, time=1e-2, alpha=0.99,
                         g_dir=1, node_state=None,
                         vtk_output_dir=None, vtk_interval=10, track_pids=None):
    """3D MUSL-EP explicit MPM solver (Johnson-Cook + Mie-Grüneisen, no damage).

    Material must have c0, Gamma0, S_alpha set (mg_enabled = True).
    g_dir: index of the body-force / floor-contact axis (default 1 = y).
    """
    if not material.mg_enabled:
        raise ValueError("Material must have c0, Gamma0, S_alpha set for the EP solver.")

    node_state = node_state or NodeState3D(mesh.nodeCount)
    if node_state.node_count != mesh.nodeCount:
        raise ValueError("NodeState3D size must match mesh.nodeCount")

    _, mpoints = build_particle_element_map(particles.positions, mesh)

    nsteps      = int(np.floor(time / dtime))
    t           = 0.0
    vtk_entries = []
    track_times = []
    track_pos   = {pid: [] for pid in (track_pids or [])}

    dx, dy, dz = mesh.deltax, mesh.deltay, mesh.deltaz

    for istep in range(nsteps):

        # ── 7: Reset grid ─────────────────────────────────────────────────────
        node_state.reset()

        # ── 8-12: P2G ─────────────────────────────────────────────────────────
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                sigma = particles.stress[pid]
                mp    = particles.mass[pid]
                Vp    = particles.volume[pid]
                vp    = particles.velocities[pid]
                for idn in mesh.element[e]:
                    x = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm3d_shape(x, dx, dy, dz)
                    if N == 0.0:
                        continue
                    node_state.mass[idn]     += N * mp
                    node_state.momentum[idn] += N * mp * vp
                    node_state.internal_force[idn, 0] -= Vp * (
                        sigma[0]*dNdx[0] + sigma[3]*dNdx[1] + sigma[5]*dNdx[2])
                    node_state.internal_force[idn, 1] -= Vp * (
                        sigma[3]*dNdx[0] + sigma[1]*dNdx[1] + sigma[4]*dNdx[2])
                    node_state.internal_force[idn, 2] -= Vp * (
                        sigma[5]*dNdx[0] + sigma[4]*dNdx[1] + sigma[2]*dNdx[2])
                    node_state.external_force[idn, g_dir] -= g * N * mp

        # ── 14-15: Update momenta and contact BC ──────────────────────────────
        f_total  = node_state.internal_force + node_state.external_force
        mv_tilde = node_state.momentum + f_total * dtime
        if mesh.bNodes.size > 0:           # frictionless floor: one-sided contact
            node_state.momentum[mesh.bNodes, g_dir] = np.maximum(
                node_state.momentum[mesh.bNodes, g_dir], 0.0)
            mv_tilde[mesh.bNodes, g_dir] = np.maximum(
                mv_tilde[mesh.bNodes, g_dir], 0.0)

        # ── 16-21: MUSL double mapping ────────────────────────────────────────
        m_safe  = np.where(node_state.mass > 0, node_state.mass, 1.0)
        v_old   = node_state.momentum / m_safe[:, None]
        v_tilde = mv_tilde             / m_safe[:, None]

        positions_old = particles.positions.copy()

        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                node_data = [
                    (idn, *get_mpm3d_shape(positions_old[pid] - mesh.node[idn], dx, dy, dz))
                    for idn in mesh.element[e]
                ]
                for idn, N, _ in node_data:
                    if N != 0.0:
                        particles.positions[pid] += dtime * N * v_tilde[idn]
                particles.velocities[pid] *= alpha
                for idn, N, _ in node_data:
                    if N != 0.0:
                        particles.velocities[pid] += N * (v_tilde[idn] - alpha * v_old[idn])

        mv_new = np.zeros_like(node_state.momentum)
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                for idn in mesh.element[e]:
                    x = positions_old[pid] - mesh.node[idn]
                    N, _ = get_mpm3d_shape(x, dx, dy, dz)
                    if N != 0.0:
                        mv_new[idn] += N * particles.mass[pid] * particles.velocities[pid]

        if mesh.bNodes.size > 0:
            mv_new[mesh.bNodes, g_dir] = np.maximum(mv_new[mesh.bNodes, g_dir], 0.0)

        # ── 22-34: G2P and constitutive update ────────────────────────────────
        v_new = mv_new / m_safe[:, None]

        k = 0.0
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                Lp = np.zeros((3, 3))
                for idn in mesh.element[e]:
                    x = positions_old[pid] - mesh.node[idn]
                    _, dNdx = get_mpm3d_shape(x, dx, dy, dz)
                    Lp += np.outer(v_new[idn], dNdx)

                _constitutive_update(material, pid, particles, dtime, Lp)

                k += 0.5 * particles.mass[pid] * np.dot(
                    particles.velocities[pid], particles.velocities[pid])

        if istep % vtk_interval == 0:
            if vtk_output_dir is not None:
                fname = write_particles_vtp_3d(
                    particles.positions, particles.velocities, particles.stress,
                    istep, t, vtk_output_dir,
                    eps_p=particles.eps_p, damage=particles.D)
                vtk_entries.append((t, fname))
            if track_pids is not None:
                track_times.append(t)
                for pid in track_pids:
                    track_pos[pid].append(particles.positions[pid].copy())

        _, mpoints = build_particle_element_map(particles.positions, mesh)
        t += dtime

    result = {'vtk_entries': vtk_entries}
    if track_pids is not None:
        result['track_times']     = np.array(track_times)
        result['track_positions'] = {pid: np.array(pos) for pid, pos in track_pos.items()}
    return result
