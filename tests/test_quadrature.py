import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.quadrature import gauss_legendre
from src.quadrature import gauss_2D


def test_gauss_legendre():
    """Test Gauss-Legendre quadrature with n=3 points over [-1, 1]."""
    x1 = -1
    x2 = 1
    n = 3
    
    x, w = gauss_legendre(x1, x2, n)
    
    print("Gauss-Legendre quadrature test:")
    print(f"Points (x): {x}")
    print(f"Weights (w): {w}")
    print(f"Sum of weights: {np.sum(w)}")  # Should be approximately x2 - x1 = 2
    
    # Expected values for n=3 over [-1,1]:
    # x ≈ [-0.7746, 0, 0.7746]
    # w ≈ [0.5556, 0.8889, 0.5556]
    expected_x = np.array([-0.774596669241483, 0.0, 0.774596669241483])
    expected_w = np.array([0.555555555555556, 0.888888888888889, 0.555555555555556])
    
    print(f"Expected x: {expected_x}")
    print(f"Expected w: {expected_w}")
    print(f"x close to expected: {np.allclose(x, expected_x, atol=1e-10)}")
    print(f"w close to expected: {np.allclose(w, expected_w, atol=1e-10)}")


def test_gauss_2D():
    """Test 2D Gauss quadrature with quadorder=2."""
    W, Q = gauss_2D(2)
    
    print("2D Gauss quadrature test:")
    print(f"Points (Q):\n{Q}")
    print(f"Weights (W): {W}")
    print(f"Sum of weights: {np.sum(W)}")  # Should be approximately (x2-x1)^2 = 4
    
    # Expected points for quadorder=2 over [1,-1] x [1,-1]:
    # 1D points: approx [-0.577, 0.577]
    # 2D: all combinations
    expected_Q = np.array([
        [-0.577, -0.577],
        [-0.577,  0.577],
        [ 0.577, -0.577],
        [ 0.577,  0.577]
    ])
    expected_W = np.array([1.0, 1.0, 1.0, 1.0])  # Products of 1D weights
    
    print(f"Expected Q:\n{expected_Q}")
    print(f"Expected W: {expected_W}")
    print(f"Q close to expected: {np.allclose(Q, expected_Q, atol=1e-3)}")
    print(f"W close to expected: {np.allclose(W, expected_W, atol=1e-10)}")


if __name__ == "__main__":
    test_gauss_legendre()
    test_gauss_2D()