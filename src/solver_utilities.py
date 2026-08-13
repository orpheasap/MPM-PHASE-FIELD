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


def get_gimp2d_shape(x, deltax, deltay, lpx, lpy):
    """
    Return the uGIMP nodal shape function and gradient for a particle
    relative to a grid node at the origin.

    The particle occupies a rectangular domain of half-widths (lpx, lpy),
    so the support reaches deltax + lpx in x and deltay + lpy in y --
    wider than the single cell of get_mpm2d_shape.  The shape function is a
    tensor product of the 1D four-branch GIMP kernels, written out for both
    directions here rather than in a helper.

    Outside the support the weight and the gradient are exactly zero, so a
    caller can skip a non-contributing node on `if N == 0.0`.
    """
    dx = x[0]
    dy = x[1]

    if abs(dx) >= deltax + lpx or abs(dy) >= deltay + lpy:
        return 0.0, np.zeros(2)

    # ----- x direction -----
    if abs(dx) < lpx:
        Nx = 1.0 - (dx * dx + lpx * lpx) / (2.0 * deltax * lpx)
        dNx = -dx / (deltax * lpx)
    elif abs(dx) < deltax - lpx:
        Nx = 1.0 - abs(dx) / deltax
        dNx = -np.sign(dx) / deltax
    else:
        Nx = (deltax + lpx - abs(dx)) ** 2 / (4.0 * deltax * lpx)
        dNx = -np.sign(dx) * (deltax + lpx - abs(dx)) / (2.0 * deltax * lpx)

    # ----- y direction -----
    if abs(dy) < lpy:
        Ny = 1.0 - (dy * dy + lpy * lpy) / (2.0 * deltay * lpy)
        dNy = -dy / (deltay * lpy)
    elif abs(dy) < deltay - lpy:
        Ny = 1.0 - abs(dy) / deltay
        dNy = -np.sign(dy) / deltay
    else:
        Ny = (deltay + lpy - abs(dy)) ** 2 / (4.0 * deltay * lpy)
        dNy = -np.sign(dy) * (deltay + lpy - abs(dy)) / (2.0 * deltay * lpy)

    N = Nx * Ny

    dNdx = np.zeros(2)
    dNdx[0] = dNx * Ny
    dNdx[1] = Nx * dNy

    return N, dNdx


def _gimp_neighbour_cells(ix, iy, numx, numy):
    """
    Return the cell indices of cell (ix, iy) and its neighbours on a
    structured numx-by-numy grid: nine cells in the interior, six along an
    edge, four at a corner.  Neighbours that fall off the grid are dropped
    rather than clamped, so no index ever points past the end of the grid.
    """
    cells = []
    for jy in range(iy - 1, iy + 2):
        if jy < 0 or jy >= numy:
            continue
        for jx in range(ix - 1, ix + 2):
            if jx < 0 or jx >= numx:
                continue
            cells.append(jx + jy * numx)

    return cells


def build_gimp_element_map(mesh):
    """
    Build the uGIMP connectivity: for each background cell, the sorted node
    set of that cell together with its up-to-eight neighbouring cells.

    A particle's uGIMP support never reaches beyond the ring of cells around
    the one it sits in, so this is the node set to scatter to and gather from.
    Entries are ragged -- border cells have fewer neighbours and hence fewer
    nodes.  The background grid is Eulerian, so this is built once, before the
    time loop, and never rebuilt.
    """
    emap = []

    for e in range(mesh.elemCount):
        ix = e % mesh.numx
        iy = e // mesh.numx
        nodes = []
        for cell in _gimp_neighbour_cells(ix, iy, mesh.numx, mesh.numy):
            nodes.extend(mesh.element[cell].tolist())
        emap.append(np.unique(np.array(nodes, dtype=int)))

    return emap
