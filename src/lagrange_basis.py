import numpy as np

def lagrange_basis_Q4(coord):
    """
    Lagrange basis functions for quadralateral (Q4) element.
    
    Parameters:
    coord: [xi, eta] coordinates in the reference element [-1, 1] x [-1, 1]
    
    Returns:
    N: Shape functions (4,)
    dNdxi: Derivatives of shape functions w.r.t. xi and eta (4, 2)
    """
    xi = coord[0]
    eta = coord[1]
    
    N = (1/4) * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    
    dNdxi = (1/4) * np.array([
        [-(1 - eta), -(1 - xi)],
        [1 - eta, -(1 + xi)],
        [1 + eta, 1 + xi],
        [-(1 + eta), 1 - xi]
    ])

    return N, dNdxi


# Reference-element node coordinates for the 8-node hexahedron (B8).
# Ordering matches the structured hex connectivity built in mesh3D.Mesh3D:
#   bottom face (zeta = -1):  1,2,3,4   top face (zeta = +1):  5,6,7,8
_B8_NODES = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=float)


def lagrange_basis_B8(coord):
    """
    Trilinear Lagrange basis functions for the 8-node hexahedron (B8).

    Parameters:
    coord: [xi, eta, zeta] coordinates in the reference cube [-1, 1]^3

    Returns:
    N:     Shape functions (8,)
    dNdxi: Derivatives w.r.t. (xi, eta, zeta)  (8, 3)
    """
    xi, eta, zeta = coord[0], coord[1], coord[2]
    xi_i   = _B8_NODES[:, 0]
    eta_i  = _B8_NODES[:, 1]
    zeta_i = _B8_NODES[:, 2]

    N = 0.125 * (1 + xi_i * xi) * (1 + eta_i * eta) * (1 + zeta_i * zeta)

    dNdxi = np.empty((8, 3))
    dNdxi[:, 0] = 0.125 * xi_i   * (1 + eta_i * eta) * (1 + zeta_i * zeta)
    dNdxi[:, 1] = 0.125 * eta_i  * (1 + xi_i * xi)   * (1 + zeta_i * zeta)
    dNdxi[:, 2] = 0.125 * zeta_i * (1 + xi_i * xi)   * (1 + eta_i * eta)

    return N, dNdxi
