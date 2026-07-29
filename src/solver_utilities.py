import numpy as np


class NodeState:
    """Container for nodal MPM state variables."""

    def __init__(self, node_count):
        self.node_count = node_count
        self.mass = np.zeros(node_count)
        self.momentum = np.zeros((node_count, 2))
        self.internal_force = np.zeros((node_count, 2))
        self.external_force = np.zeros((node_count, 2))
        # ----- Phase-field grid quantities -----
        self.pseudo_mass = np.zeros(node_count)        # m_I^phi = sum_p N_I V_p^0
        self.phase_num = np.zeros(node_count)          # sum_p N_I V_p^0 phi_p  (numerator for grid phi)
        self.phase = np.zeros(node_count)              # grid phase field phi_I^t
        self.damage_force = np.zeros(node_count)       # F_I (energetic damage residual)

    def reset(self):
        self.mass.fill(0.0)
        self.momentum.fill(0.0)
        self.internal_force.fill(0.0)
        self.external_force.fill(0.0)
        self.pseudo_mass.fill(0.0)
        self.phase_num.fill(0.0)
        self.phase.fill(0.0)
        self.damage_force.fill(0.0)


def get_mpm2d_shape(x, deltax, deltay):
    """
    Return the MPM nodal shape function and gradient for a particle
    relative to a grid node at the origin.

    The shape function is a tensor product of 1D tent functions.
    """
    xi = x[0] / deltax
    eta = x[1] / deltay

    if abs(xi) >= 1.0 or abs(eta) >= 1.0:
        return 0.0, np.zeros(2)

    Nx = 1.0 - abs(xi)
    Ny = 1.0 - abs(eta)
    N = Nx * Ny

    dNdx = np.zeros(2)
    dNdx[0] = -(np.sign(xi) if xi != 0 else 0.0) * Ny / deltax
    dNdx[1] = -(np.sign(eta) if eta != 0 else 0.0) * Nx / deltay

    return N, dNdx


def build_particle_element_map(xp, mesh):
    """
    Build a mapping from particles to element indices and
    from elements to particle indices.
    """
    pElems = np.zeros(len(xp), dtype=int)
    mpoints = [[] for _ in range(mesh.elemCount)]

    for p, point in enumerate(xp):
        ix = int(np.floor(point[0] / mesh.deltax))
        iy = int(np.floor(point[1] / mesh.deltay))
        ix = max(0, min(ix, mesh.numx - 1))
        iy = max(0, min(iy, mesh.numy - 1))
        e = ix + iy * mesh.numx
        pElems[p] = e
        mpoints[e].append(p)

    return pElems, mpoints
