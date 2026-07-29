"""
3D MUSL explicit MPM solver — linear (Hookean) elastic constitutive model.

Constitutive update (per assignment screenshot — simple hypoelastic, no rotation):
    L_p   = sum_I v_I (x) grad(phi_I)          velocity gradient
    D_p   = 0.5 (L_p + L_p^T)                  rate of deformation
    deps  = dt * D_p                           strain increment (tensor)
    dsig  = lambda*tr(deps)*I + 2*mu*deps      isotropic Hooke increment
    sigma_p^{t+dt} = sigma_p^t + dsig

Stress is stored in Voigt form: [xx, yy, zz, xy, yz, xz].
The deformation gradient and particle volume are still advected
(F = (I + L_p dt) F, V = det(F) V0) so that internal forces use the
current volume.
"""
import numpy as np
from src.vtk_export import write_particles_vtp_3d


class NodeState3D:
    def __init__(self, node_count):
        self.node_count     = node_count
        self.mass           = np.zeros(node_count)
        self.momentum       = np.zeros((node_count, 3))
        self.internal_force = np.zeros((node_count, 3))
        self.external_force = np.zeros((node_count, 3))

    def reset(self):
        self.mass.fill(0.0)
        self.momentum.fill(0.0)
        self.internal_force.fill(0.0)
        self.external_force.fill(0.0)


def get_mpm3d_shape(x, deltax, deltay, deltaz):
    """Trilinear MPM grid (tent) shape function and gradient.

    x is the particle position relative to grid node I.
    Returns N (scalar) and dNdx (3,)."""
    xi  = x[0] / deltax
    eta = x[1] / deltay
    zet = x[2] / deltaz
    if abs(xi) >= 1.0 or abs(eta) >= 1.0 or abs(zet) >= 1.0:
        return 0.0, np.zeros(3)
    Nx = 1.0 - abs(xi)
    Ny = 1.0 - abs(eta)
    Nz = 1.0 - abs(zet)
    N  = Nx * Ny * Nz
    dNdx = np.zeros(3)
    dNdx[0] = -(np.sign(xi)  if xi  != 0 else 0.0) * Ny * Nz / deltax
    dNdx[1] = -(np.sign(eta) if eta != 0 else 0.0) * Nx * Nz / deltay
    dNdx[2] = -(np.sign(zet) if zet != 0 else 0.0) * Nx * Ny / deltaz
    return N, dNdx


def build_particle_element_map(xp, mesh):
    pElems  = np.zeros(len(xp), dtype=int)
    mpoints = [[] for _ in range(mesh.elemCount)]
    for p, point in enumerate(xp):
        ix = int(np.floor(point[0] / mesh.deltax))
        iy = int(np.floor(point[1] / mesh.deltay))
        iz = int(np.floor(point[2] / mesh.deltaz))
        ix = max(0, min(ix, mesh.numx - 1))
        iy = max(0, min(iy, mesh.numy - 1))
        iz = max(0, min(iz, mesh.numz - 1))
        e  = ix + iy * mesh.numx + iz * mesh.numx * mesh.numy
        pElems[p] = e
        mpoints[e].append(p)
    return pElems, mpoints


# ── Constitutive update — linear Hooke ────────────────────────────────────────

def _constitutive_update(mat, pid, particles, dtime, Lp):
    I3 = np.eye(3)
    lam = mat.K - 2.0 * mat.G / 3.0      # Lame's first parameter
    mu  = mat.G                          # shear modulus

    # Advect deformation gradient and volume (for internal-force integration)
    F    = (I3 + Lp * dtime) @ particles.deformation_gradient[pid].reshape(3, 3)
    detF = np.linalg.det(F)
    particles.deformation_gradient[pid] = F.ravel()
    particles.volume[pid] = detF * particles.initial_volume[pid]

    # Rate of deformation and strain increment (screenshot)
    Dp   = 0.5 * (Lp + Lp.T)
    deps = dtime * Dp

    # Isotropic linear-elastic stress increment
    tr   = deps[0, 0] + deps[1, 1] + deps[2, 2]
    dsig = lam * tr * I3 + 2.0 * mu * deps

    # Update Cauchy stress in Voigt form [xx, yy, zz, xy, yz, xz]
    s = particles.stress[pid]
    s[0] += dsig[0, 0]
    s[1] += dsig[1, 1]
    s[2] += dsig[2, 2]
    s[3] += dsig[0, 1]
    s[4] += dsig[1, 2]
    s[5] += dsig[0, 2]


# ── Main solver ───────────────────────────────────────────────────────────────

def run_mpm_solver_3d(mesh, particles, material,
                      g=9.81, dtime=1e-5, time=1e-2, alpha=0.99,
                      g_dir=1, node_state=None,
                      vtk_output_dir=None, vtk_interval=10, track_pids=None):
    """3D MUSL explicit MPM solver with a linear-elastic constitutive model.

    g_dir: index of the body-force / floor-contact axis (default 1 = y).
    The floor at y = 0 (mesh.bNodes) acts as a one-sided frictionless wall.
    """
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
                    # f_int = -V * sigma . grad(N)
                    node_state.internal_force[idn, 0] -= Vp * (
                        sigma[0]*dNdx[0] + sigma[3]*dNdx[1] + sigma[5]*dNdx[2])
                    node_state.internal_force[idn, 1] -= Vp * (
                        sigma[3]*dNdx[0] + sigma[1]*dNdx[1] + sigma[4]*dNdx[2])
                    node_state.internal_force[idn, 2] -= Vp * (
                        sigma[5]*dNdx[0] + sigma[4]*dNdx[1] + sigma[2]*dNdx[2])
                    node_state.external_force[idn, g_dir] -= g * N * mp

        # ── 14-15: Update momenta and BCs ─────────────────────────────────────
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
                    particles.positions, particles.velocities,
                    particles.stress, istep, t, vtk_output_dir)
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
