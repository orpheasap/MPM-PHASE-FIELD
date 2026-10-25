import numpy as np
from .lagrange_basis import lagrange_basis_Q4

def square_node_array(pt1, pt2, pt3, pt4, numnod_u, numnod_v):
    """
    Generates a quadrilateral array of nodes between the counterclockwise
    ordering of nodes pt1 - pt4. There are numnod_u nodes in the u direction
    (pt1 - pt2) and numnod_v nodes in the v direction (pt2 - pt3).
    
    Parameters:
    pt1, pt2, pt3, pt4: Node coordinates as [x, y] lists or arrays
    numnod_u: Number of nodes in u direction
    numnod_v: Number of nodes in v direction
    
    Returns:
    X: Array of node positions (numnod_v * numnod_u, 2)
    """
    xi_pts = np.linspace(-1, 1, numnod_u)
    eta_pts = np.linspace(-1, 1, numnod_v)
    
    x_pts = np.array([pt1[0], pt2[0], pt3[0], pt4[0]])
    y_pts = np.array([pt1[1], pt2[1], pt3[1], pt4[1]])
    
    X = np.zeros((numnod_v * numnod_u, 2))
    
    for r in range(numnod_v):
        eta = eta_pts[r]
        for c in range(numnod_u):
            xi = xi_pts[c]
            # Get interpolation basis at xi, eta
            N, _ = lagrange_basis_Q4([xi, eta])
            X[r * numnod_u + c, :] = [np.dot(x_pts, N), np.dot(y_pts, N)]
    return X


def make_elem(node_pattern, num_u, num_v, inc_u, inc_v):
    """
    Creates a connectivity list.
    
    Parameters:
    node_pattern: Array of node indices for one element
    num_u: Number of elements in u direction
    num_v: Number of elements in v direction
    inc_u: Increment array for u direction
    inc_v: Increment array for v direction
    
    Returns:
    element: Connectivity matrix (num_u * num_v, len(node_pattern))
    """
    inc = np.zeros(len(node_pattern))
    e = 0
    element = np.zeros((num_u * num_v, len(node_pattern)), dtype=int)
    
    for row in range(num_v):
        for col in range(num_u):
            element[e, :] = node_pattern + inc
            inc += inc_u
            e += 1
        inc = (row + 1) * inc_v
    
    return element


class Mesh:
    """
    Structured mesh class for 2D rectangle.

    """
    def __init__(self, Lx, Ly, noX0, noY0):
        """
        Build structured mesh for rectangle (Lx, Ly)
        
        Parameters:
        Lx, Ly: Dimensions of the rectangle
        noX0, noY0: Number of elements along x and y directions
        """
        noX = noX0
        noY = noY0
        
        deltax = Lx / noX0
        deltay = Ly / noY0
        
        # Build the mesh
        nnx = noX + 1
        nny = noY + 1
        node = square_node_array([0, 0], [Lx, 0], [Lx, Ly], [0, Ly], nnx, nny)
        inc_u = 1
        inc_v = nnx
        node_pattern = np.array([0, 1, nnx + 1, nnx])  # 0-based indexing
        element = make_elem(node_pattern, noX, noY, inc_u, inc_v)
        
        # Find boundaries
        eps = 1e-12
        lNodes = np.where(np.abs(node[:, 0]) < eps)[0]
        rNodes = np.where(np.abs(node[:, 0] - Lx) < eps)[0]
        tNodes = np.where(np.abs(node[:, 1] - Ly) < eps)[0]
        bNodes = np.where(np.abs(node[:, 1]) < eps)[0]
        
        # Build nodal support
        nodalSup = [[] for _ in range(len(node))]
        for e in range(len(element)):
            sctr = element[e]
            for i in range(4):
                nodei = sctr[i]
                nodalSup[nodei].append(e)
        
        # Assign attributes
        self.node = node
        self.element = element
        self.deltax = deltax
        self.deltay = deltay
        self.elemCount = len(element)
        self.nodeCount = len(node)
        self.numx = noX
        self.numy = noY
        self.lNodes = lNodes
        self.rNodes = rNodes
        self.tNodes = tNodes
        self.bNodes = bNodes
        self.dxInv = 1 / deltax
        self.dyInv = 1 / deltay
        self.nodalSup = nodalSup