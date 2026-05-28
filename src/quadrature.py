import numpy as np


def gauss_legendre(x1, x2, n):
    """
    Computes Gaussian quadrature points and weights for Gauss-Legendre quadrature.
    
    Parameters:
    x1, x2: Integral limits
    n: Number of Gaussian points
    
    Returns:
    x: Gaussian points (n,)
    w: Gaussian weights (n,)
    """
    eps = 1e-15
    m = (n + 1) // 2
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    
    x = np.zeros(n)
    w = np.zeros(n)
    
    for i in range(1, m + 1):
        z = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        z1 = 0.0
        
        while abs(z - z1) > eps:
            pj = 1.0
            pj_1 = 0.0
            for j in range(1, n + 1):
                pj_2 = pj_1
                pj_1 = pj
                pj = ((2.0 * j - 1.0) * z * pj_1 - (j - 1.0) * pj_2) / j
            pp = n * (z * pj - pj_1) / (z**2 - 1.0)
            z1 = z
            z = z1 - pj / pp
        
        # Compute polynomials and derivative at the root z
        pj = 1.0
        pj_1 = 0.0
        for j in range(1, n + 1):
            pj_2 = pj_1
            pj_1 = pj
            pj = ((2.0 * j - 1.0) * z * pj_1 - (j - 1.0) * pj_2) / j
        pp = n * (z * pj - pj_1) / (z**2 - 1.0)
        
        x[i-1] = xm - xl * z
        x[n - i] = xm + xl * z
        w[i-1] = 2.0 * xl / ((1.0 - z**2) * pp**2)
        w[n - i] = w[i-1]
    
    return x, w


def gauss_2D(n):
    """
    Computes 2D Gauss quadrature points and weights using tensor product.
    
    Parameters:
    n: Quadrature order in each direction
    sdim: Spatial dimension (must be 2)
    
    Returns:
    W: Quadrature weights (n**2,)
    Q: Quadrature points (n**2, 2)
    """
    
    r1pt, r1wt = gauss_legendre(-1, 1, n)
    
    num_points = n ** 2
    Q = np.zeros((num_points, 2))
    W = np.zeros(num_points)
    
    d = 0
    for i in range(n):
        for j in range(n):
            Q[d, :] = [r1pt[i], r1pt[j]]
            W[d] = r1wt[i] * r1wt[j]
            d += 1
    
    return W, Q


if __name__ == "__main__":
    # Example usage
    n = 3
    x1, x2 = -1, 1
    x, w = gauss_legendre(x1, x2, n)
    print("Gauss-Legendre points:", x)
    print("Gauss-Legendre weights:", w)
    
    W, Q = gauss_2D(n)
    print("2D Gauss quadrature points:\n", Q)
    print("2D Gauss quadrature weights:", W)