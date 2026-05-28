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
