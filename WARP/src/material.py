import numpy as np


class Material:
    def __init__(
        self,
        rho, E, nu,
        # Johnson-Cook flow stress (no thermal softening)
        A, B, n, C, eps_dot_0,
        # Johnson-Cook damage (optional, no thermal term)
        D1=None, D2=None, D3=None, D4=None,
        # Mie-Grüneisen EOS parameters (optional)
        c0=None, Gamma0=None, S_alpha=None, chi=0.9,
    ):
        # Elastic
        self.initial_density = rho
        self.density = rho
        self.E = E
        self.nu = nu
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))

        # Johnson-Cook flow stress: sigma_f = [A + B*eps_p^n][1 + C*ln(eps_dot*)]
        self.A = A
        self.B = B
        self.n = n
        self.C = C
        self.eps_dot_0 = eps_dot_0

        # Linear EOS — P-wave speed for CFL condition
        self.wave_speed = np.sqrt((self.K + 4.0 * self.G / 3.0) / rho)

        # Johnson-Cook damage: eps_f = [D1 + D2*exp(D3*sigma*)][1 + D4*ln(eps_dot*)]
        # Disabled if any constant is None
        self.damage_enabled = all(d is not None for d in [D1, D2, D3, D4])
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.D4 = D4

        # Mie-Grüneisen EOS: p = rho0*c0^2*(eta-1)*[eta - Gamma0/2*(eta-1)] /
        #                         [eta - S_alpha*(eta-1)]^2  +  Gamma0 * e
        # chi: Taylor-Quinney coefficient for adiabatic energy update (e += chi*sigma_f*deps_p)
        self.mg_enabled = all(x is not None for x in [c0, Gamma0, S_alpha])
        self.c0      = c0
        self.Gamma0  = Gamma0
        self.S_alpha = S_alpha
        self.chi     = chi
