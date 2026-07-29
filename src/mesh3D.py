import numpy as np


def box_node_array(Lx, Ly, Lz, nnx, nny, nnz):
    """
    Generate a structured grid of nodes filling the box [0,Lx]x[0,Ly]x[0,Lz].

    Node ordering is x-fastest, then y, then z:
        index(i, j, k) = i + j*nnx + k*nnx*nny

    Returns:
    X: node coordinates (nnx*nny*nnz, 3)
    """
    xs = np.linspace(0.0, Lx, nnx)
    ys = np.linspace(0.0, Ly, nny)
    zs = np.linspace(0.0, Lz, nnz)

    # meshgrid with 'ij' then reorder so x varies fastest
    X = np.zeros((nnx * nny * nnz, 3))
    idx = 0
    for k in range(nnz):
        for j in range(nny):
            for i in range(nnx):
                X[idx] = [xs[i], ys[j], zs[k]]
                idx += 1
    return X


def make_hex_elements(noX, noY, noZ):
    """
    Build the connectivity for a structured hexahedral (B8) mesh.

    Node ordering per element matches lagrange_basis._B8_NODES:
        bottom face (z-): n0, n0+1, n0+1+nnx, n0+nnx
        top face    (z+): same four + nnx*nny
    """
    nnx = noX + 1
    nny = noY + 1
    nxy = nnx * nny

    element = np.zeros((noX * noY * noZ, 8), dtype=int)
    e = 0
    for ez in range(noZ):
        for ey in range(noY):
            for ex in range(noX):
                n0 = ex + ey * nnx + ez * nxy
                element[e] = [
                    n0,
                    n0 + 1,
                    n0 + 1 + nnx,
                    n0 + nnx,
                    n0 + nxy,
                    n0 + 1 + nxy,
                    n0 + 1 + nnx + nxy,
                    n0 + nnx + nxy,
                ]
                e += 1
    return element


class Mesh3D:
    """
    Structured hexahedral (B8) background grid for 3D MPM.

    Conventions mirror the 2D Mesh class so the solver code stays parallel.
    """

    def __init__(self, Lx, Ly, Lz, noX, noY, noZ):
        nnx = noX + 1
        nny = noY + 1
        nnz = noZ + 1

        deltax = Lx / noX
        deltay = Ly / noY
        deltaz = Lz / noZ

        node    = box_node_array(Lx, Ly, Lz, nnx, nny, nnz)
        element = make_hex_elements(noX, noY, noZ)

        eps = 1e-12
        lNodes = np.where(np.abs(node[:, 0])       < eps)[0]   # x = 0
        rNodes = np.where(np.abs(node[:, 0] - Lx)  < eps)[0]   # x = Lx
        bNodes = np.where(np.abs(node[:, 1])       < eps)[0]   # y = 0  (impact wall)
        tNodes = np.where(np.abs(node[:, 1] - Ly)  < eps)[0]   # y = Ly
        kNodes = np.where(np.abs(node[:, 2])       < eps)[0]   # z = 0
        fNodes = np.where(np.abs(node[:, 2] - Lz)  < eps)[0]   # z = Lz

        self.node      = node
        self.element   = element
        self.deltax    = deltax
        self.deltay    = deltay
        self.deltaz    = deltaz
        self.elemCount = len(element)
        self.nodeCount = len(node)
        self.numx      = noX
        self.numy      = noY
        self.numz      = noZ
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.lNodes = lNodes
        self.rNodes = rNodes
        self.bNodes = bNodes
        self.tNodes = tNodes
        self.kNodes = kNodes
        self.fNodes = fNodes
        self.dxInv  = 1.0 / deltax
        self.dyInv  = 1.0 / deltay
        self.dzInv  = 1.0 / deltaz
