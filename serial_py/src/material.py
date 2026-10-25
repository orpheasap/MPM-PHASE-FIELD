import numpy as np

class Material:
    def __init__(self, E, nu, rho, stressState='PLANE_STRESS'):
        self.E = E
        self.nu = nu
        self.density = rho
        self.stressState = stressState
        self.elasticity_matrix = self.elasticityMatrix(stressState)

    def elasticityMatrix(self, stressState='PLANE_STRESS'):
        """Return the isotropic elasticity matrix for the given stress state."""
        self.stressState = stressState
        E0 = self.E
        nu0 = self.nu

        if stressState == 'PLANE_STRESS':   # plane stress
            C = E0 / (1 - nu0**2) * np.array([
                [1.0, nu0, 0.0],
                [nu0, 1.0, 0.0],
                [0.0, 0.0, (1 - nu0) / 2.0],
            ])
        elif stressState == 'PLANE_STRAIN':  # plane strain
            C = E0 / ((1 + nu0) * (1 - 2 * nu0)) * np.array([
                [1 - nu0, nu0, 0.0],
                [nu0, 1 - nu0, 0.0],
                [0.0, 0.0, 0.5 - nu0],
            ])
        else:                                  # 3D
            C = np.zeros((6, 6), dtype=float)
            C[0:3, 0:3] = E0 / ((1 + nu0) * (1 - 2 * nu0)) * np.array([
                [1 - nu0, nu0, nu0],
                [nu0, 1 - nu0, nu0],
                [nu0, nu0, 1 - nu0],
            ])
            C[3:6, 3:6] = E0 / (2 * (1 + nu0)) * np.eye(3)
        self.elasticity_matrix = C
        return C