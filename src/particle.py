import numpy as np


class ParticleSet:
    """Container for particle state arrays used by the MUSL-EP solver."""

    def __init__(self, count):
        self.count = count

        # Kinematics
        self.positions   = np.zeros((count, 2))
        self.velocities  = np.zeros((count, 2))
        self.mass        = np.ones(count)
        self.volume      = np.ones(count)
        # F stored row-major: [F11, F12, F21, F22]
        self.deformation_gradient = np.tile(np.array([1., 0., 0., 1.]), (count, 1))

        # Rotated Cauchy stress in Voigt notation: [sigma_xx, sigma_yy, sigma_xy]
        self.stress = np.zeros((count, 3))

        # Un-rotated deviatoric Cauchy stress, Voigt: [s_xx, s_yy, s_xy]
        self.stress_dev = np.zeros((count, 3))

        # Plasticity
        self.eps_p  = np.zeros(count)   # equivalent plastic strain

        # Damage
        self.D_init = np.zeros(count)   # damage initiation variable
        self.D      = np.zeros(count)   # damage variable in [0, 1]

        # Internal energy density [J/m³] — used by Mie-Grüneisen EOS
        self.e = np.zeros(count)

        # Reference state and BCs
        self.initial_positions = np.zeros((count, 2))
        self.initial_volume    = np.ones(count)
        self.pElems            = np.zeros(count, dtype=int)
        self.mpoints           = []
        self.neumann_particles = np.zeros(count, dtype=bool)
        self.dirichlet_particles = np.zeros(count, dtype=bool)

    # Current density — always consistent with mass and volume
    @property
    def density(self):
        return self.mass / self.volume

    def set_initial_state(self):
        self.initial_positions = self.positions.copy()
        self.initial_volume    = self.volume.copy()

    def displacement(self):
        return self.positions - self.initial_positions
