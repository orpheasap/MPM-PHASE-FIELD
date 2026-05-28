import numpy as np
from src.vtk_export import write_particles_vtp


class NodeState:
    """Container for nodal MPM state variables."""

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
    """Tent-function shape function and gradient for particle offset x."""
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
    """Map particles to elements and elements to particle lists."""
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
    """Von Mises equivalent stress from 2D deviatoric components (plane strain).
    s_zz = -(s_xx + s_yy) from the traceless condition."""
    s_zz = -(s_xx + s_yy)
    return np.sqrt(1.5 * (s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * s_xy**2))


def _jc_flow_stress(mat, eps_p, eps_dot, D):
    """Johnson-Cook flow stress (isothermal): [A + B*eps_p^n][1 + C*ln(eps_dot*)][1-D]"""
    eps_dot_star = max(eps_dot / mat.eps_dot_0, 1.0)
    return (
        (mat.A + mat.B * max(eps_p, 0.0) ** mat.n)
        * (1.0 + mat.C * np.log(eps_dot_star))
        * max(1.0 - D, 0.0)
    )


def _constitutive_update(mat, pid, particles, dtime, Lp):
    """MUSL-EP constitutive update (steps 24-34) for a single particle."""
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
    R          = U_s @ Vt                      # rotation from polar decomp F = R U

    # 28: Un-rotated strain rate and deviatoric part (plane strain: d_zz = 0)
    dp       = R.T @ Dp @ R
    d_vol    = (dp[0, 0] + dp[1, 1]) / 3.0    # tr_3D(d)/3; d_zz = 0 in plane strain
    dp_dev   = dp - d_vol * I2                 # 2x2; d_dev_zz = -d_vol (implicit)
    d_dev_zz = -(dp_dev[0, 0] + dp_dev[1, 1]) # traceless condition

    # Equivalent strain rate (von Mises, includes out-of-plane d_dev_zz)
    eps_dot = np.sqrt(2.0 / 3.0 * (
        dp_dev[0, 0]**2 + dp_dev[1, 1]**2 + d_dev_zz**2 + 2.0 * dp_dev[0, 1]**2
    ))

    D  = particles.D[pid]
    Gp = (1.0 - D) * mat.G                    # 29a: damaged shear modulus

    # 29b: Elastic trial deviatoric stress
    s       = particles.stress_dev[pid]
    s_trial = s + 2.0 * Gp * dtime * np.array([dp_dev[0, 0], dp_dev[1, 1], dp_dev[0, 1]])

    # 29c: Trial von Mises stress
    sigma_trial_eq = _von_mises_dev(s_trial[0], s_trial[1], s_trial[2])

    # 29d: JC flow stress
    eps_p   = particles.eps_p[pid]
    sigma_f = _jc_flow_stress(mat, eps_p, eps_dot, D)

    # 29e-f: Elastic or plastic radial return
    delta_eps_p = 0.0
    if sigma_trial_eq <= sigma_f or Gp < 1e-14:
        # 29e: elastic
        particles.stress_dev[pid] = s_trial
    else:
        # 29f: plastic — radial return
        delta_eps_p               = (sigma_trial_eq - sigma_f) / (3.0 * Gp)
        particles.eps_p[pid]      = eps_p + delta_eps_p
        particles.stress_dev[pid] = (sigma_f / sigma_trial_eq) * s_trial

    s_upd = particles.stress_dev[pid]

    # 30: Linear EOS pressure (positive = compression)
    p_hat = -mat.K * (1.0 - detF) * (1.0 - D)

    # 31: Assemble un-rotated stress and rotate back to global frame
    sigma_unrot = np.array([[s_upd[0] + p_hat, s_upd[2]          ],
                             [s_upd[2],          s_upd[1] + p_hat]])
    sigma_rot = R @ sigma_unrot @ R.T
    particles.stress[pid] = np.array([sigma_rot[0, 0], sigma_rot[1, 1], sigma_rot[0, 1]])

    # 33: Damage update
    if mat.damage_enabled and delta_eps_p > 0.0:
        sigma_eq    = _von_mises_dev(s_upd[0], s_upd[1], s_upd[2])
        triaxiality = -p_hat / sigma_eq if sigma_eq > 1e-14 else 0.0

        eps_dot_p_star = max((delta_eps_p / dtime) / mat.eps_dot_0, 1.0)
        eps_f = max(
            (mat.D1 + mat.D2 * np.exp(mat.D3 * triaxiality))
            * (1.0 + mat.D4 * np.log(eps_dot_p_star)),
            1e-14,
        )
        particles.D_init[pid] += delta_eps_p / eps_f
        if particles.D_init[pid] >= 1.0:
            particles.D[pid] = min(10.0 * (particles.D_init[pid] - 1.0), 1.0)


# ── Main solver ───────────────────────────────────────────────────────────────

def run_mpm_solver(mesh, particles, material, traction,
                   g=9.81, dtime=1e-5, time=1e-2, alpha=0.99,
                   node_state=None, vtk_output_dir=None, vtk_interval=10,
                   v_dirichlet=None):
    """
    MUSL explicit MPM solver with Johnson-Cook elasto-plastic constitutive model.

    Parameters
    ----------
    mesh         : Mesh object
    particles    : ParticleSet object
    material     : Material object
    traction     : Applied downward traction for Neumann BC (N/m)
    g            : Gravity magnitude (m/s^2)
    dtime        : Time step (s)
    time         : Total simulation time (s)
    alpha        : FLIP/PIC blending (1 = pure FLIP, 0 = pure PIC)
    v_dirichlet  : Prescribed velocity [vx, vy] for particles.dirichlet_particles.
                   None or [0,0] fixes them in place.
    """
    if v_dirichlet is None:
        v_dirichlet = np.zeros(2)
    v_dirichlet = np.asarray(v_dirichlet, dtype=float)
    node_state = node_state or NodeState(mesh.nodeCount)
    if node_state.node_count != mesh.nodeCount:
        raise ValueError("NodeState size must match mesh.nodeCount")

    _, mpoints = build_particle_element_map(particles.positions, mesh)

    nsteps      = int(np.floor(time / dtime))
    t           = 0.0
    vtk_entries = []
    ta, ka      = [], []

    for istep in range(nsteps):

        # ── 7: Reset grid ─────────────────────────────────────────────────────
        node_state.reset()

        # ── 8-12: P2G ─────────────────────────────────────────────────────────
        dirichlet_nodes_set = set()
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                sigma      = particles.stress[pid]
                mp         = particles.mass[pid]
                Vp         = particles.volume[pid]
                vp         = particles.velocities[pid]
                is_dir     = particles.dirichlet_particles[pid]
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
                    if particles.neumann_particles[pid]:
                        node_state.external_force[idn, 1] -= traction * N * Vp
                    if is_dir:
                        dirichlet_nodes_set.add(idn)
        dirichlet_nodes = (np.fromiter(dirichlet_nodes_set, dtype=int)
                           if dirichlet_nodes_set else np.array([], dtype=int))

        # ── 14-15: Update momenta and Dirichlet BCs ───────────────────────────
        f_total  = node_state.internal_force + node_state.external_force
        mv_tilde = node_state.momentum + f_total * dtime
        for nodes in [mesh.lNodes, mesh.rNodes, mesh.bNodes, mesh.tNodes]:
            if nodes.size > 0:
                node_state.momentum[nodes] = 0.0
                mv_tilde[nodes]            = 0.0

        # Prescribed particle velocity (dirichlet_particles)
        if dirichlet_nodes.size > 0:
            mv_tilde[dirichlet_nodes] = node_state.mass[dirichlet_nodes, None] * v_dirichlet

        # ── 16-21: MUSL double mapping ────────────────────────────────────────
        m_safe  = np.where(node_state.mass > 0, node_state.mass, 1.0)
        v_old   = node_state.momentum / m_safe[:, None]
        v_tilde = mv_tilde             / m_safe[:, None]

        # All shape-function evaluations in steps 17-23 use positions at time t
        positions_old = particles.positions.copy()

        # 17-18: Update particle positions and velocities
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                # Pre-compute N for all nodes at the OLD position
                node_data = [
                    (idn, *get_mpm2d_shape(positions_old[pid] - mesh.node[idn],
                                           mesh.deltax, mesh.deltay))
                    for idn in mesh.element[e]
                ]
                # 17: position update
                for idn, N, _ in node_data:
                    if N != 0.0:
                        particles.positions[pid] += dtime * N * v_tilde[idn]
                # 18: velocity update — FLIP/PIC blend
                #     v_p^new = alpha*v_p^old + sum_I N_I*(v~_I - alpha*v_I^old)
                particles.velocities[pid] *= alpha
                for idn, N, _ in node_data:
                    if N != 0.0:
                        particles.velocities[pid] += N * (v_tilde[idn] - alpha * v_old[idn])

        # 19: Second P2G — nodal momentum from updated particle momenta,
        #     evaluated at OLD positions (x_p^t per the algorithm)
        mv_new = np.zeros_like(node_state.momentum)
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                for idn in mesh.element[e]:
                    x = positions_old[pid] - mesh.node[idn]
                    N, _ = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)
                    if N != 0.0:
                        mv_new[idn] += N * particles.mass[pid] * particles.velocities[pid]

        # 20: Apply BCs to second mapping
        for nodes in [mesh.lNodes, mesh.rNodes, mesh.bNodes, mesh.tNodes]:
            if nodes.size > 0:
                mv_new[nodes] = 0.0

        if dirichlet_nodes.size > 0:
            mv_new[dirichlet_nodes] = node_state.mass[dirichlet_nodes, None] * v_dirichlet

        # ── 22-34: G2P and constitutive update ───────────────────────────────
        # 22: Nodal velocity from second mapping
        v_new = mv_new / m_safe[:, None]

        k = 0.0
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                # 23: Velocity gradient at OLD positions, NEW nodal velocities
                Lp = np.zeros((2, 2))
                for idn in mesh.element[e]:
                    x = positions_old[pid] - mesh.node[idn]
                    _, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)
                    Lp += np.outer(v_new[idn], dNdx)

                # 24-34: Constitutive update
                _constitutive_update(material, pid, particles, dtime, Lp)

                k += 0.5 * particles.mass[pid] * np.dot(
                    particles.velocities[pid], particles.velocities[pid]
                )

        #ta.append(t)
        #ka.append(k)

        if vtk_output_dir is not None and istep % vtk_interval == 0:
            fname = write_particles_vtp(
                particles.positions, particles.velocities,
                particles.stress, particles.stress_dev,
                particles.eps_p, particles.D,
                istep, t, vtk_output_dir,
            )
            vtk_entries.append((t, fname))

        _, mpoints = build_particle_element_map(particles.positions, mesh)
        t += dtime

    return {
        'time':       np.array(ta),
        'kinetic':    np.array(ka),
        'vtk_entries': vtk_entries,
    }
