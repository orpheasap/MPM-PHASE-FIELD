"""
Phase-field fracture helpers for the explicit USL-MPM-PF scheme.

Implements the constitutive functions of Section 4 of
`explicit_USL_MPM_PF.tex`:

    free energy        psi   = g(phi) * psi_e0 + Gc * gamma(phi, grad phi)
    degradation        g(phi)  = (1 - phi)^2,   g'(phi) = -2 (1 - phi)
    crack density      gamma   = 1/(4 c_w l) ( w(phi) + l^2 |grad phi|^2 )

with the Ambrosio-Tortorelli family (AT1 / AT2) for the local crack
geometry function w(phi) and its normalisation constant c_w.

These quantities feed the explicit nodal phase-field update (Eq. 5.x of
the report):

    phi_I^{t+dt} = phi_I^t - dt/(eta m_I^phi) * F_I

where F_I is the energetic damage residual assembled from the particles.
"""

import numpy as np


class PhaseFieldModel:
    """Constitutive container for the variational phase-field fracture model.

    Parameters
    ----------
    Gc : float
        Critical energy release rate (fracture toughness), N/mm.
    ell : float
        Regularisation length scale, mm.
    eta : float
        Phase-field viscosity (artificial), MPa*s. Renders the damage
        sub-problem parabolic so it can be marched with forward Euler.
    model : {'AT2', 'AT1'}
        Ambrosio-Tortorelli crack-density choice.
    kappa : float
        Small residual stiffness preventing zero stiffness when phi -> 1.
    """

    def __init__(self, Gc, ell, eta, model='AT2', kappa=1e-7):
        self.Gc = float(Gc)
        self.ell = float(ell)
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.model = model.upper()

        if self.model == 'AT2':
            self.c_w = 0.5
        elif self.model == 'AT1':
            self.c_w = 2.0 / 3.0
        else:
            raise ValueError(f"Unknown phase-field model '{model}' (use 'AT1' or 'AT2').")

    # ----- Degradation function g(phi) = (1 - phi)^2 -----
    @staticmethod
    def g(phi):
        return (1.0 - phi) ** 2

    @staticmethod
    def dg(phi):
        """g'(phi) = -2 (1 - phi)."""
        return -2.0 * (1.0 - phi)

    # ----- Local crack geometry function w(phi) -----
    def w(self, phi):
        if self.model == 'AT2':
            return phi ** 2
        return phi                     # AT1

    def dw(self, phi):
        """w'(phi):  AT2 -> 2 phi,  AT1 -> 1."""
        if self.model == 'AT2':
            return 2.0 * phi
        return np.ones_like(phi) if np.ndim(phi) else 1.0

    # ----- Derived quantities -----
    @property
    def crack_force_coeff(self):
        """Gc / (2 c_w ell), prefactor of the gradient/local fracture terms."""
        return self.Gc / (2.0 * self.c_w * self.ell)

    def sigma_c(self, E):
        """Homogeneous critical (failure) stress, Eq. (4.x)."""
        if self.model == 'AT1':
            return np.sqrt(3.0 * E * self.Gc / (8.0 * self.ell))
        return (3.0 / 16.0) * np.sqrt(3.0 * E * self.Gc / self.ell)

    def dt_crit_phase(self, h, ndim=2):
        """Parabolic (diffusion) stability limit of the explicit phase update.

            dt <= c_w eta h^2 / (ndim ell Gc).
        """
        return self.c_w * self.eta * h ** 2 / (ndim * self.ell * self.Gc)


def elastic_energy_density(eff_stress, strain):
    """Undamaged elastic energy density psi_e0 = 1/2 eps : C : eps.

    Using Voigt vectors with engineering shear strain, this equals
    1/2 * sigma0 . eps_vec (sigma0 = C : eps the effective stress).
    """
    return 0.5 * float(np.dot(eff_stress, strain))
