"""
MUSL-EP solver — Mie-Grüneisen EOS, no damage.

EOS (README step 30):
    eta   = rho / rho0 = 1 / det(F)
    mu    = eta - 1
    p_H   = rho0 * c0^2 * mu * (eta - Gamma0/2 * mu) / (eta - S_alpha * mu)^2
    p_hat = p_H  +  Gamma0 * e_p

Energy update (adiabatic, Taylor-Quinney, feeds p_hat next step):
    From e = Cv*rho0*(T - Tr)  and  rho*Cp*ΔT = chi*sigma_f*Δeps_p  (Cp ≈ Cv):
        Δe = rho0*Cv*ΔT = rho0/rho * chi*sigma_f*Δeps_p = detF * chi*sigma_f*Δeps_p
    So:
        e_p += detF * chi * sigma_f * delta_eps_p      [J/m³]
"""
import numpy as np
from src.vtk_export import write_particles_vtp


class NodeState:
    def __init__(self, node_count):
        self.node_count     = node_count
        self.mass           = np.zeros(node_count)
        self.momentum       = np.zeros((node_count, 2))
        self.internal_force = np.zeros((node_count, 2))
        self.external_force = np.zeros((node_count, 2))

    def reset(self):
        self.mass.fill(0.0)
        self.momentum.fill(0.0)
        self.internal_force.fill(0.0)
        self.external_force.fill(0.0)


def get_mpm2d_shape(x, deltax, deltay):
    xi  = x[0] / deltax
    eta = x[1] / deltay
    if abs(xi) >= 1.0 or abs(eta) >= 1.0:
        return 0.0, np.zeros(2)
    Nx = 1.0 - abs(xi)
    Ny = 1.0 - abs(eta)
    N  = Nx * Ny
    dNdx = np.zeros(2)
    dNdx[0] = -(np.sign(xi)  if xi  != 0 else 0.0) * Ny / deltax
    dNdx[1] = -(np.sign(eta) if eta != 0 else 0.0) * Nx / deltay
    return N, dNdx


def build_particle_element_map(xp, mesh):
    pElems  = np.zeros(len(xp), dtype=int)
    mpoints = [[] for _ in range(mesh.elemCount)]
    for p, point in enumerate(xp):
        ix = int(np.floor(point[0] / mesh.deltax))
        iy = int(np.floor(point[1] / mesh.deltay))
        ix = max(0, min(ix, mesh.numx - 1))
        iy = max(0, min(iy, mesh.numy - 1))
        e  = ix + iy * mesh.numx
        pElems[p] = e
        mpoints[e].append(p)
    return pElems, mpoints


# ── Constitutive helpers ──────────────────────────────────────────────────────

def _von_mises_dev(s_xx, s_yy, s_xy):
    """Von Mises equivalent stress from deviatoric components (plane strain).
    s_zz = -(s_xx + s_yy) from the traceless condition."""
    s_zz = -(s_xx + s_yy)
    return np.sqrt(1.5 * (s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * s_xy**2))


def _jc_flow_stress(mat, eps_p, eps_dot):
    """Johnson-Cook flow stress — no damage factor."""
    eps_dot_star = max(eps_dot / mat.eps_dot_0, 1.0)
    return (mat.A + mat.B * max(eps_p, 0.0) ** mat.n) * (1.0 + mat.C * np.log(eps_dot_star))


def _mg_pressure(mat, detF, e):
    """Mie-Grüneisen EOS — README step 30 (D = 0).

    eta   = rho / rho0 = 1 / detF
    mu    = eta - 1                     (positive = compression)
    p_hat = rho0*c0^2 * mu * (eta - Gamma0/2 * mu) / (eta - S_alpha*mu)^2
            + Gamma0 * e
    """
    eta   = 1.0 / detF
    mu    = eta - 1.0
    denom = (eta - mat.S_alpha * mu) ** 2
    if denom < 1e-20:                   # guard against Hugoniot limit
        return -mat.Gamma0 * e
    p_H = mat.initial_density * mat.c0**2 * mu * (eta - 0.5 * mat.Gamma0 * mu) / denom
    return -(p_H + mat.Gamma0 * e)


def _constitutive_update(mat, pid, particles, dtime, Lp):
    """MUSL constitutive update — Mie-Grüneisen EOS, no damage."""
    I2 = np.eye(2)

    # 24: Deformation gradient
    F    = (I2 + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
    detF = np.linalg.det(F)
    particles.deformation_gradient[pid] = F.ravel()

    # 25: Volume
    particles.volume[pid] = detF * particles.initial_volume[pid]

    # 27: Strain-rate tensor and polar decomposition via SVD
    Dp         = 0.5 * (Lp + Lp.T)
    U_s, _, Vt = np.linalg.svd(F)
    R          = U_s @ Vt

    # 28: Un-rotated strain rate and deviatoric part (plane strain: d_zz = 0)
    dp       = R.T @ Dp @ R
    d_vol    = (dp[0, 0] + dp[1, 1]) / 3.0
    dp_dev   = dp - d_vol * I2
    d_dev_zz = -(dp_dev[0, 0] + dp_dev[1, 1])

    eps_dot = np.sqrt(2.0 / 3.0 * (
        dp_dev[0, 0]**2 + dp_dev[1, 1]**2 + d_dev_zz**2 + 2.0 * dp_dev[0, 1]**2
    ))

    # 29a: No damage — undamaged shear modulus
    Gp = mat.G

    # 29b: Elastic trial deviatoric stress
    s       = particles.stress_dev[pid]
    s_trial = s + 2.0 * Gp * dtime * np.array([dp_dev[0, 0], dp_dev[1, 1], dp_dev[0, 1]])

    # 29c: Trial von Mises stress
    sigma_trial_eq = _von_mises_dev(s_trial[0], s_trial[1], s_trial[2])

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

    # 30: Mie-Grüneisen EOS
    # Adiabatic energy update: Δe = detF * chi * sigma_f * delta_eps_p  (README step 32)
    particles.e[pid] += detF * mat.chi * sigma_f * delta_eps_p
    p_hat = _mg_pressure(mat, detF, particles.e[pid])

    # 31: Assemble un-rotated stress and rotate back to global frame
    sigma_unrot = np.array([[s_upd[0] + p_hat, s_upd[2]          ],
                             [s_upd[2],          s_upd[1] + p_hat]])
    sigma_rot = R @ sigma_unrot @ R.T
    particles.stress[pid] = np.array([sigma_rot[0, 0], sigma_rot[1, 1], sigma_rot[0, 1]])

    # Step 33 (damage) removed — this solver has no damage


# ── Main solver ───────────────────────────────────────────────────────────────

def run_mpm_solver(mesh, particles, material,
                   g=9.81, dtime=1e-5, time=1e-2, alpha=0.99,
                   node_state=None, vtk_output_dir=None, vtk_interval=10,
                   track_pids=None):
    """
    MUSL explicit MPM solver — Mie-Grüneisen EOS, no damage.

    Material must have c0, Gamma0, S_alpha set (mg_enabled = True).
    Internal energy e_p is tracked per particle and updated from plastic
    dissipation each step (Taylor-Quinney: e += chi * sigma_f * delta_eps_p).
    """
    if not material.mg_enabled:
        raise ValueError("Material must have c0, Gamma0, S_alpha set for MieGruNoD solver.")

    node_state = node_state or NodeState(mesh.nodeCount)
    if node_state.node_count != mesh.nodeCount:
        raise ValueError("NodeState size must match mesh.nodeCount")

    _, mpoints = build_particle_element_map(particles.positions, mesh)

    nsteps      = int(np.floor(time / dtime))
    t           = 0.0
    vtk_entries = []
    track_times = []
    track_pos   = {pid: [] for pid in (track_pids or [])}
    #ta, ka      = [], []

    for istep in range(nsteps):

        # ── 7: Reset grid ─────────────────────────────────────────────────────
        node_state.reset()

        # ── 8-12: P2G ─────────────────────────────────────────────────────────
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                sigma  = particles.stress[pid]
                mp     = particles.mass[pid]
                Vp     = particles.volume[pid]
                vp     = particles.velocities[pid]
                is_dir = particles.dirichlet_particles[pid]
                for idn in mesh.element[e]:
                    x = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)
                    if N == 0.0:
                        continue
                    node_state.mass[idn]     += N * mp
                    node_state.momentum[idn] += N * mp * vp
                    node_state.internal_force[idn, 0] -= Vp * (sigma[0]*dNdx[0] + sigma[2]*dNdx[1])
                    node_state.internal_force[idn, 1] -= Vp * (sigma[2]*dNdx[0] + sigma[1]*dNdx[1])
                    node_state.external_force[idn, 1] -= g * N * mp

        # ── 14-15: Update momenta and Dirichlet BCs ───────────────────────────
        f_total  = node_state.internal_force + node_state.external_force
        mv_tilde = node_state.momentum + f_total * dtime
        #for nodes in [mesh.lNodes, mesh.rNodes, mesh.tNodes]:
        #    if nodes.size > 0:
        #        node_state.momentum[nodes] = 0.0
        #        mv_tilde[nodes]            = 0.0
        if mesh.bNodes.size > 0:                         # frictionless floor: one-sided contact
            node_state.momentum[mesh.bNodes, 1] = np.maximum(node_state.momentum[mesh.bNodes, 1], 0.0)
            mv_tilde[mesh.bNodes, 1]            = np.maximum(mv_tilde[mesh.bNodes, 1],            0.0)

        # ── 16-21: MUSL double mapping ────────────────────────────────────────
        m_safe  = np.where(node_state.mass > 0, node_state.mass, 1.0)
        v_old   = node_state.momentum / m_safe[:, None]
        v_tilde = mv_tilde             / m_safe[:, None]

        positions_old = particles.positions.copy()

        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                node_data = [
                    (idn, *get_mpm2d_shape(positions_old[pid] - mesh.node[idn],
                                           mesh.deltax, mesh.deltay))
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
                    N, _ = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)
                    if N != 0.0:
                        mv_new[idn] += N * particles.mass[pid] * particles.velocities[pid]

        #for nodes in [mesh.lNodes, mesh.rNodes, mesh.tNodes]:
        #    if nodes.size > 0:
        #        mv_new[nodes] = 0.0
        if mesh.bNodes.size > 0:                         # frictionless floor: one-sided contact
            mv_new[mesh.bNodes, 1] = np.maximum(mv_new[mesh.bNodes, 1], 0.0)

        # ── 22-34: G2P and constitutive update ───────────────────────────────
        v_new = mv_new / m_safe[:, None]

        k = 0.0
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                Lp = np.zeros((2, 2))
                for idn in mesh.element[e]:
                    x = positions_old[pid] - mesh.node[idn]
                    _, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)
                    Lp += np.outer(v_new[idn], dNdx)

                _constitutive_update(material, pid, particles, dtime, Lp)

                k += 0.5 * particles.mass[pid] * np.dot(
                    particles.velocities[pid], particles.velocities[pid]
                )

        #ta.append(t)
        #ka.append(k)

        if istep % vtk_interval == 0:
            if vtk_output_dir is not None:
                fname = write_particles_vtp(
                    particles.positions, particles.velocities,
                    particles.stress, particles.stress_dev,
                    particles.eps_p, particles.D,
                    istep, t, vtk_output_dir,
                )
                vtk_entries.append((t, fname))
            if track_pids is not None:
                track_times.append(t)
                for pid in track_pids:
                    track_pos[pid].append(particles.positions[pid].copy())

        _, mpoints = build_particle_element_map(particles.positions, mesh)
        t += dtime

    result = {
        #'time':        np.array(ta),
        #'kinetic':     np.array(ka),
        'vtk_entries': vtk_entries,
    }
    if track_pids is not None:
        result['track_times']     = np.array(track_times)
        result['track_positions'] = {pid: np.array(pos) for pid, pos in track_pos.items()}
    return result
