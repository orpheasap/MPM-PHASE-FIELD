# =============================================================================
#  yvonnet_model.py -- single source of truth for the physical inputs and
#  derived quantities of the Yvonnet-style anisotropic two-field PF-MPM plate.
#
#  Both Yvonnet_plate.py (fresh run) and resume_Yvonnet.py (checkpoint resume)
#  build their material model, background grid and solver constants by calling
#  build_model(params) with the same `params` dict, so the two scripts can
#  never drift out of sync. Yvonnet_plate.py owns DEFAULT_PARAMS and writes
#  the dict it actually used into the checkpoint; resume_Yvonnet.py only ever
#  reads params back out of the checkpoint.
# =============================================================================
import numpy as np

from src.material import Material
from src.mesh2D import Mesh


DEFAULT_PARAMS = dict(
    # --- Material (SI) ---
    E=30.0e9, nu=0.25, rho=2450.0,
    # --- Anisotropic layer model ---
    theta_deg=60.0, xi=30.0,
    # --- Fracture (phase-field) properties ---
    gc_d=10.0, gc_alpha=1.0,
    # --- Background grid ---
    Lx=0.01, Ly=0.01, hx=0.1e-3, hy=0.1e-3,
    ell_d=0.2e-3, ell_alpha=0.2e-3,
    # --- Loading / geometry (kept for record; baked into particle state) ---
    sigma=2.0e6, notch_length=0.005,
    # --- Explicit time-stepping safety factor ---
    CFL=0.5,
)


def build_model(params):
    """Rebuild the material model, background grid and solver constants
    from a `params` dict (see DEFAULT_PARAMS for the required keys)."""
    E, nu, rho = params['E'], params['nu'], params['rho']
    theta_deg, xi = params['theta_deg'], params['xi']
    gc_d, gc_alpha = params['gc_d'], params['gc_alpha']
    Lx, Ly, hx, hy = params['Lx'], params['Ly'], params['hx'], params['hy']
    ell_d, ell_alpha = params['ell_d'], params['ell_alpha']
    CFL = params['CFL']

    material = Material(E, nu, rho, stressState='PLANE_STRESS')

    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    pref  = E / (1.0 - nu**2)

    # Fitted effective stiffness in the LAYER frame, C'(alpha) = A1 + A2 (1-a)^2
    A1 = pref * np.array([
        [0.8437, 0.0, 0.0],
        [0.0,    0.0, 0.0],
        [0.0,    0.0, 0.0],
    ])
    A2 = pref * np.array([
        [0.1563, nu,  0.0],
        [nu,     1.0, 0.0],
        [0.0,    0.0, (1.0 - nu) / 2.0],
    ])

    # Stress Voigt rotation matrix T_sig(theta), Voigt order {11,22,12}
    T_sig = np.array([
        [c*c,  s*s, -2.0*c*s],
        [s*s,  c*c,  2.0*c*s],
        [c*s, -c*s,  c*c - s*s],
    ])

    CA1 = T_sig @ A1 @ T_sig.T             # alpha-independent part
    M2  = T_sig @ A2 @ T_sig.T             # multiplies (1-alpha)^2

    def C_aniso(alpha):
        """Global-frame damage-dependent effective stiffness C_a(alpha, theta) [Pa]."""
        return CA1 + M2 * (1.0 - alpha) ** 2

    P_a     = np.array([[c*c, s*c], [s*c, s*s]])   # projector onto layer direction e1'
    omega_a = np.eye(2) + xi * P_a                  # anisotropic gradient tensor

    numx = int(round(Lx / hx))
    numy = int(round(Ly / hy))
    h_e  = min(hx, hy)
    mesh = Mesh(Lx, Ly, numx, numy)

    # Three explicit stability limits (Sec. 7 of the .tex)
    C_max = float(np.max(np.linalg.eigvalsh(C_aniso(0.0))))
    c_d   = np.sqrt(C_max / rho)
    dt_u  = h_e / c_d

    ndim      = 2
    eta_d     = 2.0 * ndim * gc_d * ell_d / (h_e * c_d)
    eta_alpha = eta_d

    dt_d_crit     = eta_d     * h_e**2 / (2.0 * ndim * gc_d     * ell_d)
    dt_alpha_crit = eta_alpha * h_e**2 / (2.0 * ndim * gc_alpha * ell_alpha * (1.0 + xi))

    dtime = CFL * min(dt_u, dt_d_crit, dt_alpha_crit)

    return dict(
        material=material, C_aniso=C_aniso, M2=M2, omega_a=omega_a, T_sig=T_sig,
        mesh=mesh, numx=numx, numy=numy, h_e=h_e,
        C_max=C_max, c_d=c_d, dt_u=dt_u,
        eta_d=eta_d, eta_alpha=eta_alpha,
        dt_d_crit=dt_d_crit, dt_alpha_crit=dt_alpha_crit,
        dtime=dtime,
    )
