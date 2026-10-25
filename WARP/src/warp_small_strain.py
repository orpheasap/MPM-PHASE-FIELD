# Small-strain hypoelastic plane-strain elasticity with AT2 degradation, for the Warp backbone.
# Port of the stress update of the root pre_noched_plate_PF.py (Voigt, engineering shear):
#
#   d eps = sym(L) dt,  eps += d eps,  sigma0 += C : d eps,  sigma = (g(phi) + kappa) sigma0
#   H = max(H, 1/2 sigma0 : eps)       (history field, no tension/compression split)
#   J = det F
#
# The constitutive step is its own kernel so that a plastic version can replace it in a new
# driver; once plasticity exists, psi0 must come from the elastic strain.
import numpy as np
import warp as wp

from .warp_phasefield import at2_g


class PlaneStrainMaterial:
    """Isotropic linear-elastic material in plane strain: E, nu, density, Voigt C (3 x 3)."""

    def __init__(self, E, nu, rho):
        self.E = E
        self.nu = nu
        self.density = rho
        self.elasticity_matrix = plane_strain_elasticity_matrix(E, nu)


def plane_strain_elasticity_matrix(E, nu):
    """C in Voigt order (xx, yy, xy) acting on engineering shear strain."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, 0.5 - nu]])


@wp.kernel
def stress_update_pf(F: wp.array(dtype=wp.mat33),
                     L: wp.array(dtype=wp.mat22),
                     phi: wp.array(dtype=wp.float32),
                     C: wp.mat33,
                     kappa: wp.float32,
                     dt: wp.float32,
                     eps: wp.array(dtype=wp.vec3),
                     sigma0: wp.array(dtype=wp.vec3),
                     sigma: wp.array(dtype=wp.vec3),
                     H: wp.array(dtype=wp.float32),
                     J: wp.array(dtype=wp.float32)):
    p = wp.tid()
    Lp = L[p]
    d_eps = wp.vec3(dt * Lp[0, 0], dt * Lp[1, 1], dt * (Lp[0, 1] + Lp[1, 0]))
    eps_p = eps[p] + d_eps
    s0 = sigma0[p] + C * d_eps
    eps[p] = eps_p
    sigma0[p] = s0
    sigma[p] = (at2_g(phi[p]) + kappa) * s0

    psi0 = 0.5 * wp.dot(s0, eps_p)
    if psi0 > H[p]:
        H[p] = psi0

    J[p] = wp.determinant(F[p])
