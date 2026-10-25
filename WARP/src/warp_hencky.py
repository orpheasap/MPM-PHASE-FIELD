# Hencky (logarithmic-strain) elasticity for the Warp backbone, plane strain with F33 = 1.
#
# The stress follows GeoWarp's assemble_residual_2d_hencky (mpm/mpm_utils.py), Zhao et al. (2026),
# MIT licence, with the plastic return removed and b = F F^T taken directly from F instead of
# being stored and updated. Retyped to float32.
import numpy as np
import warp as wp


class ElasticMaterial:
    """Isotropic elastic material: density, Young's modulus, Poisson's ratio."""

    def __init__(self, rho, E, nu):
        self.density = rho
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.mu = self.G
        self.lam = self.K - 2.0 * self.G / 3.0
        # P-wave speed for the CFL condition
        self.wave_speed = np.sqrt((self.K + 4.0 * self.G / 3.0) / rho)


@wp.func
def hencky_principal_strain(F: wp.mat33):
    """Principal Hencky strains e_i = 1/2 ln(lambda_i^2) of b = F F^T, and their directions."""
    b = F * wp.transpose(F)
    Q, d = wp.eig3(b)
    e = wp.vec3(0.5 * wp.log(wp.max(d[0], 1.0e-8)),
                0.5 * wp.log(wp.max(d[1], 1.0e-8)),
                0.5 * wp.log(wp.max(d[2], 1.0e-8)))
    return e, Q


@wp.func
def hencky_cauchy_stress(F: wp.mat33, lam: wp.float32, mu: wp.float32):
    """Cauchy stress sigma = tau / J, Kirchhoff tau = lam tr(e) I + 2 mu e in the principal frame."""
    e, Q = hencky_principal_strain(F)
    tr_e = e[0] + e[1] + e[2]
    tau = wp.mat33(0.0)
    for a in range(3):
        n = wp.vec3(Q[0, a], Q[1, a], Q[2, a])
        tau += (lam * tr_e + 2.0 * mu * e[a]) * wp.outer(n, n)
    return tau / wp.determinant(F)


@wp.func
def hencky_energy_density(F: wp.mat33, lam: wp.float32, mu: wp.float32):
    """Elastic energy per unit reference volume: mu e:e + 1/2 lam (tr e)^2."""
    e, Q = hencky_principal_strain(F)
    tr_e = e[0] + e[1] + e[2]
    return mu * wp.dot(e, e) + 0.5 * lam * tr_e * tr_e
