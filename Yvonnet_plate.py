# %%
# =============================================================================
#  Pre-notched plate  --  Explicit USL-MPM with the ANISOTROPIC TWO-FIELD
#  phase-field fracture model for 3D-printed layered materials.
#
#  Adaptation of `pre_noched_plate_PF_aniso_layer.py` to a smaller plate with
#  Yvonnet-style toughness parameters, following the theory of
#  `aniso_layer_PF_MPM_explicit.tex` (Li et al. 2021, recast as explicit USL-MPM).
#
#  Two damage fields are carried per material point:
#    * d(x)      -- BULK layer damage        (isotropic, AT2 crack density)
#    * alpha(x)  -- interfacial MICRO damage  (anisotropic, oriented by layers)
#
#  Constitutive law (Sec. 3-4 of the .tex):
#      sigma   = g(d) * C_a(alpha, theta) : eps ,      g(d) = (1 - d)^2
#      C_a(alpha,theta) = T_sig(theta) C'(alpha) T_sig(theta)^T          (eq. 33)
#      C'(alpha)_ij     = a1_ij + a2_ij (1 - alpha)^2                    (eq. 7.8)
#      dC_a/dalpha      = -2(1-alpha) T_sig A2 T_sig^T                   (eq. 34)
#      omega_a(theta)   = I + xi * P_a ,  P_a = e1' (x) e1'              (eq. 41)
#
#  Two dissipative micro-forces (eta_d dot d, eta_alpha dot alpha) turn the two
#  elliptic stationarity conditions into parabolic Allen-Cahn evolutions marched
#  by explicit forward Euler on lumped pseudo-masses (Sec. 6):
#      d_I^{t+dt}     = d_I    - dt/(eta_d     m_I) F_I^d
#      alpha_I^{t+dt} = alpha_I- dt/(eta_alpha m_I) F_I^alpha
#
#  eta_d, eta_alpha are chosen (eq. 8.x) so the two parabolic critical steps
#  coincide with the elastodynamic CFL, so all three fields march with one dt.
#
#  *** UNITS: SI throughout -> m, N, Pa, kg, s. ***
#      stress/stiffness [Pa], length [m], density [kg/m^3], toughness [N/m],
#      viscosity [Pa*s], traction [Pa], time [s].
#
#  Problem parameters (this study):
#      E = 10 GPa, nu = 0.25, rho = 2450 kg/m^3,  plate 0.01 x 0.01 m,
#      h = 0.25 mm,  notch x in [0, 0.005] m at y = 0.005 m,
#      applied sigma = 1 MPa.
#  Model parameters:
#      theta = 60 deg,  ell_d = ell_alpha = 2h,  xi = 30,
#      gc_d = 4e3 N/m,  gc_alpha = 1e3 N/m,  plane stress.
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.solver_utilities import NodeState, get_mpm2d_shape, build_particle_element_map
from src.vtk_export import write_pvd, write_particles_vtp_aniso


np.set_printoptions(precision=4, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# =============================================================================
#  Material properties  (SI units)
# =============================================================================
E   = 10.0e9      # Pa        (10 GPa)
nu  = 0.25        # -
rho = 2450.0      # kg/m^3

# Reference isotropic plane-stress material (used only for the intact check).
material = Material(E, nu, rho, stressState='PLANE_STRESS')
print_section("Material Properties (SI)")
print(f"E = {E:.4g} Pa,  nu = {nu},  rho = {rho:.4g} kg/m^3")
print("Isotropic plane-stress elasticity matrix [Pa]:")
print(material.elasticity_matrix)


# =============================================================================
#  Anisotropic layered constitutive model  (Sec. 3-4 of the .tex)
# =============================================================================
theta_deg = 60.0                       # layer angle theta (deg)
theta     = np.deg2rad(theta_deg)
xi        = 30.0                       # anisotropy penalty xi^alpha (>> 1)

c, s = np.cos(theta), np.sin(theta)
pref = E / (1.0 - nu**2)               # E/(1-nu^2) prefactor of the fit

# --- Fitted effective stiffness in the LAYER frame,  C'(alpha) = A1 + A2 (1-a)^2
#     (eq. 7.8).  Coefficients 0.8437, 0.1563 are the straight-interphase RVE fit;
#     A1 + A2 = pref * [[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]] identically
#     (0.8437 + 0.1563 = 1), so the intact limit C'(0) always recovers the
#     isotropic plane-stress matrix regardless of nu.
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

# --- Stress Voigt rotation matrix T_sig(theta)  (eq. 31), Voigt order {11,22,12}
T_sig = np.array([
    [c*c,  s*s, -2.0*c*s],
    [s*s,  c*c,  2.0*c*s],
    [c*s, -c*s,  c*c - s*s],
])

# --- Rotate once to the GLOBAL frame:  C_a(alpha) = CA1 + M2 (1-alpha)^2
#     dC_a/dalpha = -2 (1-alpha) M2.   (eqs. 33-34)
CA1 = T_sig @ A1 @ T_sig.T             # alpha-independent part
M2  = T_sig @ A2 @ T_sig.T             # multiplies (1-alpha)^2  (= T_sig A2 T_sig^T)


def C_aniso(alpha):
    """Global-frame damage-dependent effective stiffness C_a(alpha, theta) [Pa]."""
    return CA1 + M2 * (1.0 - alpha) ** 2


# --- Anisotropic gradient (structural) tensor omega_a(theta) = I + xi P_a  (eq. 41)
P_a     = np.array([[c*c, s*c], [s*c, s*s]])     # projector onto layer direction e1'
omega_a = np.eye(2) + xi * P_a                   # 2x2, constant for uniform theta

print_section("Anisotropic Layer Model")
print(f"Layer angle theta = {theta_deg} deg,  anisotropy penalty xi = {xi}")
print("C_a(alpha=0, theta) [Pa]  (should equal isotropic plane-stress matrix):")
print(C_aniso(0.0))
print(f"Max |C_a(0)-C_iso| = {np.max(np.abs(C_aniso(0.0)-material.elasticity_matrix)):.3e} Pa")
print("C_a(alpha=1, theta) [Pa]  (interphase fully debonded):")
print(C_aniso(1.0))
print("omega_a(theta):")
print(omega_a)
print(f"lambda_max(omega_a) = 1 + xi = {1.0 + xi:.1f}")


# =============================================================================
#  Fracture (phase-field) properties  (SI units)
# =============================================================================
# Length scales tied to the element size:  ell_d = ell_alpha = 2 h_e  (Sec. 4).
gc_d     = 4.0          # N/m   bulk layer fracture toughness
gc_alpha = 1.0          # N/m   interfacial toughness

print_section("Fracture Properties (SI)")
print(f"gc_d     = {gc_d:.4g} N/m   (bulk layer)")
print(f"gc_alpha = {gc_alpha:.4g} N/m   (interface)")


# =============================================================================
#  Background grid  (SI units)
# =============================================================================
_h_scale = float(os.environ.get("PF_H_SCALE", "1.0"))

Lx   = 0.01                      # m   plate width  (10 mm)
Ly   = 0.01                      # m   plate height (10 mm)
hx   = 0.25e-3 * _h_scale        # m   cell size x  (0.25 mm)
hy   = 0.25e-3 * _h_scale        # m   cell size y
numx = int(round(Lx / hx))       # 40
numy = int(round(Ly / hy))       # 40
h_e  = min(hx, hy)               # element size for the length scales

# Regularisation lengths ell_d = ell_alpha = 2 h_e
ell_d     = 2.0 * h_e
ell_alpha = 2.0 * h_e

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid (SI)")
print(f"Domain:   {Lx} x {Ly} m")
print(f"Cells:    {numx} x {numy},  h = {hx:.4g} m")
print(f"Nodes:    {mesh.nodeCount},  Elements: {mesh.elemCount}")
print(f"ell_d = ell_alpha = 2h = {ell_d:.4g} m")


# =============================================================================
#  Particle initialisation  (1 particle per background cell)
# =============================================================================
noX, noY, ngp = numx, numy, 1
W, Q  = gauss_2D(ngp)
pmesh = Mesh(Lx, Ly, noX, noY)
pCount_max = noX * noY * len(W)

all_pos  = np.zeros((pCount_max, 2))
all_vol  = np.zeros(pCount_max)
all_mass = np.zeros(pCount_max)

pid = 0
for e in range(pmesh.elemCount):
    sctr = pmesh.element[e, :]
    pts  = pmesh.node[sctr, :]
    for q in range(len(W)):
        N, dNdxi = lagrange_basis_Q4(Q[q, :])
        J0    = dNdxi.T @ pts
        detJ0 = np.linalg.det(J0)
        a     = W[q] * detJ0
        all_pos[pid, :] = N.T @ pts
        all_vol[pid]    = a
        all_mass[pid]   = a * rho
        pid += 1


# ----- Pre-notch: remove two grid rows straddling y = Ly/2, x in [0, notch] -----
notch_length = 0.005               # m   (5 mm from left)
notch_row_lo = numy // 2 - 1
notch_row_hi = numy // 2

def _in_notch(x, y):
    iy = int(np.floor(y / hy))
    return (x <= notch_length) and (iy == notch_row_lo or iy == notch_row_hi)

keep = np.array([not _in_notch(all_pos[p, 0], all_pos[p, 1])
                 for p in range(pCount_max)], dtype=bool)
n_removed = int(np.sum(~keep))

print_section("Notch")
print(f"Crack line:  y = {Ly/2:.4g} m  (rows iy in {{{notch_row_lo}, {notch_row_hi}}})")
print(f"Notch span:  x in [0, {notch_length}] m")
print(f"Removed:     {n_removed} particles")


# ----- Build ParticleSet -----
pCount = int(np.sum(keep))
particles = ParticleSet(pCount)
particles.positions[:]            = all_pos[keep]
particles.volume[:]               = all_vol[keep]
particles.mass[:]                 = all_mass[keep]
particles.deformation_gradient[:] = np.tile([1.0, 0.0, 0.0, 1.0], (pCount, 1))
particles.set_initial_state()

# Field state:  particles.phase  holds the BULK damage d,
#               particles.history holds H (bulk driving energy, max 1/2 eps:C_a:eps),
#               local array alpha_p holds the interfacial damage alpha.
particles.phase[:]   = 0.0
particles.history[:] = 0.0
alpha_p = np.zeros(pCount)        # interfacial micro damage alpha in [0,1]

print_section("Particle Initialization")
print(f"Active particles: {pCount}  (of {pCount_max} before notch)")
print(f"Total mass:       {np.sum(particles.mass):.6e} kg")


# =============================================================================
#  Neumann BCs -- symmetric tension sigma on top and bottom edges  (SI)
# =============================================================================
sigma = float(os.environ.get("PF_SIGMA", str(1.0e6)))   # Pa  (1 MPa)
neumann_traction_y = np.zeros(pCount)

for p in range(pCount):
    iy = int(np.floor(particles.positions[p, 1] / hy))
    iy = max(0, min(iy, numy - 1))
    if iy == numy - 1:          # top row -> +y traction
        particles.neumann_particles[p] = True
        neumann_traction_y[p]          =  sigma / hy
    elif iy == 0:               # bottom row -> -y traction
        particles.neumann_particles[p] = True
        neumann_traction_y[p]          = -sigma / hy

print_section("Boundary Conditions")
print(f"sigma = {sigma:.4g} Pa  -- no Dirichlet constraints")
print(f"Top particles (up): {int(np.sum(neumann_traction_y > 0))},  "
      f"bottom (dn): {int(np.sum(neumann_traction_y < 0))}")


# ----- Element-Particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)
node_state = NodeState(mesh.nodeCount)


# =============================================================================
#  Solver parameters -- three explicit stability limits (Sec. 7 of the .tex)
# =============================================================================
# (1) Elastodynamic CFL:  dt_u = h / c_d ,  c_d = sqrt(C_max / rho),
#     C_max = largest eigenvalue of the intact stiffness C_a(0, theta).
C_max  = float(np.max(np.linalg.eigvalsh(C_aniso(0.0))))
c_d    = np.sqrt(C_max / rho)
dt_u   = h_e / c_d

# Choose the viscosities (eq. 8.x) so the two parabolic critical steps EQUAL the
# elastodynamic CFL:   set dt_d_crit = dt_alpha_crit = dt_u.
#     dt_d_crit     = eta_d     h^2 / (2 n gc_d     ell_d)              (n = 2)
#     dt_alpha_crit = eta_alpha h^2 / (2 n gc_alpha ell_alpha (1+xi))
#  =>  eta_d     = 2 n gc_d     ell_d     / (h c_d)
#      eta_alpha = 2 n gc_alpha ell_alpha (1+xi) / (h c_d)
ndim      = 2
eta_d     = 2.0 * ndim * gc_d     * ell_d               / (h_e * c_d)
eta_alpha = 2.0 * ndim * gc_alpha * ell_alpha * (1.0 + xi) / (h_e * c_d)

# Resulting parabolic critical steps (equal to dt_u by construction)
dt_d_crit     = eta_d     * h_e**2 / (2.0 * ndim * gc_d     * ell_d)
dt_alpha_crit = eta_alpha * h_e**2 / (2.0 * ndim * gc_alpha * ell_alpha * (1.0 + xi))

CFL    = float(os.environ.get("PF_CFL", "0.5"))         # safety factor SF
dtime  = CFL * min(dt_u, dt_d_crit, dt_alpha_crit)
nsteps = int(os.environ.get("PF_NSTEPS", "100"))
tfinal = dtime * nsteps

print_section("Solver Parameters (three-way CFL)")
print(f"C_max (max eig C_a(0)):    {C_max:.4e} Pa")
print(f"Wave speed c_d:            {c_d:.4e} m/s")
print(f"(1) elastodynamic  dt_u        = {dt_u:.4e} s")
print(f"(2) bulk-damage    dt_d_crit   = {dt_d_crit:.4e} s")
print(f"(3) interfacial    dt_alpha_crit = {dt_alpha_crit:.4e} s")
print(f"eta_d     = {eta_d:.4e} Pa*s")
print(f"eta_alpha = {eta_alpha:.4e} Pa*s   (ratio eta_alpha/eta_d = {eta_alpha/eta_d:.3f})")
print(f"dtime (SF = {CFL}):         {dtime:.4e} s")
print(f"Total time:                {tfinal:.4e} s,  steps = {nsteps}")

shutil.rmtree('vtk_output_Yvonnet', ignore_errors=True)


# %%
# =============================================================================
#  Main explicit USL-MPM loop with two phase fields  (Algorithm 1 of the .tex)
# =============================================================================
tol         = 1e-24            # zero-mass guard [kg]
tol_pm      = 1e-30            # zero pseudo-mass guard [m^2]
vtk_dir     = 'vtk_output_Yvonnet'
vtk_interval = 50

ta, ka, sa, dda, aa = [], [], [], [], []   # time, kinetic, strain, max d, max alpha
vtk_entries = []
I_mat       = np.eye(2)
mpoints     = particles.mpoints
t           = 0.0

# Extra grid arrays for the interfacial field (bulk field reuses node_state.phase
# and node_state.damage_force; the shared pseudo-mass m_I = sum_p N_I V0).
alpha_grid = np.zeros(mesh.nodeCount)   # grid interfacial field alpha_I
alpha_num  = np.zeros(mesh.nodeCount)   # sum_p N_I V0 alpha_p
F_alpha    = np.zeros(mesh.nodeCount)   # interfacial nodal force F_I^alpha

for istep in range(nsteps):
    node_state.reset()
    alpha_grid.fill(0.0)
    alpha_num.fill(0.0)
    F_alpha.fill(0.0)

    # --- P2G: momentum + both damage projections -------------------------------
    for e in range(mesh.elemCount):
        esctr = mesh.element[e]
        for pid in mpoints[e]:
            stress = particles.stress[pid]          # degraded Cauchy stress g(d) sigma_0
            d_p    = particles.phase[pid]
            a_p    = alpha_p[pid]
            V0     = particles.initial_volume[pid]
            for idn in esctr:
                dx      = particles.positions[pid] - mesh.node[idn]
                N, dNdx = get_mpm2d_shape(dx, hx, hy)

                node_state.mass[idn]              += N * particles.mass[pid]
                node_state.momentum[idn]          += N * particles.mass[pid] * particles.velocities[pid]
                node_state.internal_force[idn, 0] -= particles.volume[pid] * (stress[0]*dNdx[0] + stress[2]*dNdx[1])
                node_state.internal_force[idn, 1] -= particles.volume[pid] * (stress[2]*dNdx[0] + stress[1]*dNdx[1])
                if particles.neumann_particles[pid]:
                    node_state.external_force[idn, 1] += neumann_traction_y[pid] * N * particles.volume[pid]

                # Shared pseudo-mass m_I = sum_p N_I V0  and mass-weighted fields
                node_state.pseudo_mass[idn] += N * V0
                node_state.phase_num[idn]   += N * V0 * d_p     # bulk d numerator
                alpha_num[idn]              += N * V0 * a_p      # interfacial alpha numerator

    # --- Grid damage fields:  field_I = (sum_p N_I V0 field_p) / m_I ------------
    has_pm = node_state.pseudo_mass > tol_pm
    node_state.phase[has_pm] = node_state.phase_num[has_pm] / node_state.pseudo_mass[has_pm]
    alpha_grid[has_pm]       = alpha_num[has_pm]      / node_state.pseudo_mass[has_pm]

    # --- Assemble both damage residuals F_I^d, F_I^alpha  (need grid gradients) -
    for e in range(mesh.elemCount):
        esctr = mesh.element[e]
        for pid in mpoints[e]:
            V0    = particles.initial_volume[pid]
            d_p   = particles.phase[pid]
            H_p   = particles.history[pid]
            eps_p = particles.strain[pid]           # Voigt [exx, eyy, gxy]

            # grid gradients of d and alpha at the particle
            grad_d = np.zeros(2)
            grad_a = np.zeros(2)
            shp = []
            for idn in esctr:
                dx      = particles.positions[pid] - mesh.node[idn]
                N, dNdx = get_mpm2d_shape(dx, hx, hy)
                grad_d += dNdx * node_state.phase[idn]
                grad_a += dNdx * alpha_grid[idn]
                shp.append((idn, N, dNdx))

            # Interfacial driving energy  G^alpha = (1-alpha) g(d) eps:M2:eps  (eq. 39)
            g_d      = (1.0 - d_p) ** 2                       # g(d)
            G_alpha  = (1.0 - alpha_p[pid]) * g_d * float(eps_p @ (M2 @ eps_p))

            # anisotropic flux omega_a . grad(alpha)
            omega_grad_a = omega_a @ grad_a

            for idn, N, dNdx in shp:
                # Bulk d residual (AT2):  (gc_d/ell_d) d N + gc_d ell_d (dN.grad d)
                #                          - 2(1-d) N H            (eqs. 24, 26)
                node_state.damage_force[idn] += (
                    (gc_d / ell_d) * d_p * N
                    + gc_d * ell_d * (dNdx @ grad_d)
                    - 2.0 * (1.0 - d_p) * N * H_p
                ) * V0

                # Interfacial alpha residual (anisotropic):
                #   (gc_a/ell_a) alpha N + gc_a ell_a (dN . omega_a grad alpha)
                #   - G^alpha N                                    (eqs. 25, 27)
                F_alpha[idn] += (
                    (gc_alpha / ell_alpha) * alpha_p[pid] * N
                    + gc_alpha * ell_alpha * (dNdx @ omega_grad_a)
                    - G_alpha * N
                ) * V0

    # --- Update momenta (no Dirichlet BCs) -------------------------------------
    node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime

    # --- Explicit grid damage updates (forward Euler on pseudo-mass) -----------
    d_grid_new = node_state.phase.copy()
    a_grid_new = alpha_grid.copy()
    d_grid_new[has_pm] -= (dtime / eta_d)     * node_state.damage_force[has_pm] / node_state.pseudo_mass[has_pm]
    a_grid_new[has_pm] -= (dtime / eta_alpha) * F_alpha[has_pm]                 / node_state.pseudo_mass[has_pm]

    # --- G2P: kinematics + map back both fields + stress/history (USL) ---------
    k = 0.0
    u = 0.0
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            Lp    = np.zeros((2, 2))
            d_new = 0.0
            a_new = 0.0
            for idn in mesh.element[e]:
                dx      = particles.positions[pid] - mesh.node[idn]
                N, dNdx = get_mpm2d_shape(dx, hx, hy)

                if node_state.mass[idn] > tol:
                    particles.velocities[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn]
                    particles.positions[pid]  += dtime * N * node_state.momentum[idn] / node_state.mass[idn]
                    vI = node_state.momentum[idn] / node_state.mass[idn]
                else:
                    vI = np.zeros(2)

                Lp    += np.outer(vI, dNdx)
                d_new += N * d_grid_new[idn]
                a_new += N * a_grid_new[idn]

            # Map back, clamp to [0,1], enforce irreversibility (pointwise max)
            d_new = min(max(d_new, 0.0), 1.0)
            a_new = min(max(a_new, 0.0), 1.0)
            d_new = max(particles.phase[pid], d_new)     # d_{t+dt} >= d_t
            a_new = max(alpha_p[pid],         a_new)     # alpha_{t+dt} >= alpha_t
            particles.phase[pid] = d_new
            alpha_p[pid]         = a_new

            # Kinematics
            F = (I_mat + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
            particles.deformation_gradient[pid] = F.reshape(4)
            particles.volume[pid]               = np.linalg.det(F) * particles.initial_volume[pid]
            dEps     = 0.5 * dtime * (Lp + Lp.T)
            dEps_vec = np.array([dEps[0, 0], dEps[1, 1], 2.0 * dEps[0, 1]])
            particles.strain[pid] += dEps_vec

            # Stress update (last): sigma_0 = C_a(alpha) : eps (total),  sigma = g(d) sigma_0
            eps_vec = particles.strain[pid]
            C_a     = C_aniso(a_new)
            sig0    = C_a @ eps_vec
            particles.eff_stress[pid] = sig0
            particles.stress[pid]     = ((1.0 - d_new) ** 2) * sig0

            # History field for bulk damage:  H = max 1/2 eps:C_a(alpha):eps
            psi_e0 = 0.5 * float(eps_vec @ sig0)
            if psi_e0 > particles.history[pid]:
                particles.history[pid] = psi_e0

            k += 0.5 * (particles.velocities[pid, 0]**2 + particles.velocities[pid, 1]**2) * particles.mass[pid]
            u += 0.5 * particles.volume[pid] * particles.stress[pid] @ particles.strain[pid]

    ta.append(t); ka.append(k); sa.append(u)
    dda.append(float(np.max(particles.phase)))
    aa.append(float(np.max(alpha_p)))

    if istep % vtk_interval == 0:
        # Fields written: velocity, von_mises, phase_d (=d), phase_a (=alpha), history.
        fname = write_particles_vtp_aniso(
            particles.positions, particles.velocities, particles.stress,
            particles.phase, alpha_p, particles.history,
            istep, t, vtk_dir,
        )
        vtk_entries.append((t, fname))

    _, mpoints = build_particle_element_map(particles.positions, mesh)
    t += dtime

print("\nSolver completed.")
print(f"Steps: {nsteps},  VTK frames: {len(vtk_entries)}")
print(f"Max bulk damage d reached:        {max(dda):.4f}")
print(f"Max interfacial damage alpha:     {max(aa):.4f}")


# ----- Checkpoint -----
_ckpt = os.path.join(vtk_dir, 'checkpoint.npz')
np.savez_compressed(
    _ckpt,
    pCount               = np.array(pCount),
    positions            = particles.positions,
    velocities           = particles.velocities,
    mass                 = particles.mass,
    volume               = particles.volume,
    deformation_gradient = particles.deformation_gradient,
    stress               = particles.stress,
    eff_stress           = particles.eff_stress,
    strain               = particles.strain,
    phase                = particles.phase,     # bulk damage d
    alpha                = alpha_p,             # interfacial damage alpha
    history              = particles.history,
    initial_positions    = particles.initial_positions,
    initial_volume       = particles.initial_volume,
    neumann_particles    = particles.neumann_particles,
    neumann_traction_y   = neumann_traction_y,
    theta_deg            = np.array(theta_deg),
    xi                   = np.array(xi),
    eta_d                = np.array(eta_d),
    eta_alpha            = np.array(eta_alpha),
    t                    = np.array(t),
    istep_start          = np.array(nsteps),
    ta = np.array(ta), ka = np.array(ka), sa = np.array(sa),
    dda = np.array(dda), aa = np.array(aa),
    vtk_times  = np.array([e[0] for e in vtk_entries]),
    vtk_fnames = np.array([e[1] for e in vtk_entries]),
)
print(f"Checkpoint saved  ->  {_ckpt}")

pvd_path = write_pvd(vtk_dir, vtk_entries)
print(f"VTK output: {pvd_path}")

solver_results = {
    'time': np.array(ta), 'kinetic': np.array(ka), 'strain': np.array(sa),
    'bulk_damage': np.array(dda), 'interf_damage': np.array(aa),
    'vtk_entries': vtk_entries,
}


# %%  ----- Post-processing (optional) -----
SHOW_PLOTS = False
if SHOW_PLOTS:
    t_us  = solver_results['time'] * 1e6
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(t_us, solver_results['kinetic']); axes[0].set_title('Kinetic energy')
    axes[1].plot(t_us, solver_results['strain']);  axes[1].set_title('Strain energy')
    axes[2].plot(t_us, solver_results['bulk_damage']);   axes[2].set_title('Max bulk d')
    axes[3].plot(t_us, solver_results['interf_damage']); axes[3].set_title('Max interfacial alpha')
    for ax in axes:
        ax.set_xlabel('Time [us]')
    plt.tight_layout(); plt.show()

    fig, (axd, axa) = plt.subplots(2, 1, figsize=(14, 10))
    scd = axd.scatter(particles.positions[:, 0], particles.positions[:, 1],
                      s=1.0, c=particles.phase, cmap='inferno', vmin=0, vmax=1, rasterized=True)
    plt.colorbar(scd, ax=axd, label='bulk damage d'); axd.set_aspect('equal')
    axd.set_title(f'Bulk damage d  (theta = {theta_deg} deg)')
    sca = axa.scatter(particles.positions[:, 0], particles.positions[:, 1],
                      s=1.0, c=alpha_p, cmap='viridis', vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sca, ax=axa, label='interfacial damage alpha'); axa.set_aspect('equal')
    axa.set_title(f'Interfacial damage alpha  (theta = {theta_deg} deg)')
    plt.tight_layout(); plt.show()
