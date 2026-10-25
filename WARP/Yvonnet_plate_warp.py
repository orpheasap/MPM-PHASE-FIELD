# %%
# =============================================================================
#  Pre-notched plate  --  Explicit USL-MPM with the ANISOTROPIC TWO-FIELD
#  phase-field fracture model for 3D-printed layered materials, on the GPU
#
#  Warp port of `serial_py/Yvonnet_plate.py`, with the cpGIMP basis and a
#  convected particle domain. Spec: .scratch/warp-yvonnet-plate/issues/
#  01-yvonnet-plate-warp.md.
#
#  Two damage fields are carried per material point:
#    * d(x)      -- BULK damage         (isotropic, AT2 crack density)
#    * alpha(x)  -- INTERFACIAL damage  (anisotropic, oriented by the layers)
#
#      sigma            = (1 - d)^2 C_a(alpha, theta) : eps        (total strain)
#      C_a(alpha,theta) = T_sig C'(alpha) T_sig^T,  C'(alpha) = A1 + A2 (1-alpha)^2
#      omega_a(theta)   = I + xi P_a,  P_a = e1' (x) e1'
#      d_I     -= dt/(eta_d m_I) F_I^d,   alpha_I -= dt/(eta_alpha m_I) F_I^alpha
#
#  Changes from the numpy reference:
#    * background grid gets a 2-cell pad ring on all four sides;
#    * the notch removes four particle rows (iy = 23..26), not two, so the
#      convected cpGIMP supports of the two notch faces do not bridge;
#    * the run is T = 20 us long (not 100 steps); no checkpoint, no resume;
#    * nominal traction t_p = (0, +-sigma) deposited as f_I += w_I t_p h.
#
#  *** UNITS: SI throughout -> m, N, Pa, kg, s.  Plane stress. ***
#  Parameters are those of serial_py/src/yvonnet_model.py DEFAULT_PARAMS:
#      E = 30 GPa, nu = 0.25, rho = 2450 kg/m^3,  plate 0.01 x 0.01 m,
#      h = 0.2 mm,  notch x in [0, 0.005] m at y = 0.005 m,  sigma = 2 MPa;
#      theta = 60 deg, xi = 30, gc_d = 10 N/m, gc_alpha = 1 N/m,
#      ell_d = ell_alpha = 0.2 mm.
#
#  ENERGY BUDGET (J/m, per unit thickness), as in the reference header:
#      K + E_el + E_d + E_a + E_visc  vs  W_ext,  a drift diagnostic.
#  E_d and E_a come from the start of the step, the other terms from the end.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import shutil
import warp as wp

from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.warp_usl_aniso import run_warp_aniso_solver
from src.vtk_export import write_pvd

np.set_printoptions(precision=4, suppress=True)
wp.init()
wp.set_device("cuda:0")


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material Properties (SI) -----
E   = 30.0e9     # Pa
nu  = 0.25
rho = 2450.0     # kg/m^3

C_iso = E / (1.0 - nu**2) * np.array([[1.0, nu,  0.0],
                                      [nu,  1.0, 0.0],
                                      [0.0, 0.0, (1.0 - nu) / 2.0]])

print_section("Material Properties (SI)")
print(f"E = {E:.4g} Pa,  nu = {nu},  rho = {rho:.4g} kg/m^3")
print("Isotropic plane-stress elasticity matrix [Pa]:")
print(C_iso)


# ----- Anisotropic Layer Model (as serial_py/src/yvonnet_model.build_model) -----
theta_deg = 60.0     # layer angle
xi        = 30.0     # anisotropy penalty

theta = np.deg2rad(theta_deg)
c, s  = np.cos(theta), np.sin(theta)
pref  = E / (1.0 - nu**2)

# Fitted effective stiffness in the LAYER frame, C'(alpha) = A1 + A2 (1-alpha)^2
A1 = pref * np.array([[0.8437, 0.0, 0.0],
                      [0.0,    0.0, 0.0],
                      [0.0,    0.0, 0.0]])
A2 = pref * np.array([[0.1563, nu,  0.0],
                      [nu,     1.0, 0.0],
                      [0.0,    0.0, (1.0 - nu) / 2.0]])

# Stress Voigt rotation matrix T_sig(theta), Voigt order {11, 22, 12}
T_sig = np.array([[c*c,  s*s, -2.0*c*s],
                  [s*s,  c*c,  2.0*c*s],
                  [c*s, -c*s,  c*c - s*s]])

CA1 = T_sig @ A1 @ T_sig.T     # alpha-independent part
M2  = T_sig @ A2 @ T_sig.T     # multiplies (1-alpha)^2


def C_aniso(alpha):
    """Global-frame damage-dependent effective stiffness C_a(alpha, theta) [Pa]."""
    return CA1 + M2 * (1.0 - alpha) ** 2


P_a     = np.array([[c*c, s*c], [s*c, s*s]])   # projector onto layer direction e1'
omega_a = np.eye(2) + xi * P_a                  # anisotropic gradient tensor

print_section("Anisotropic Layer Model")
print(f"Layer angle theta = {theta_deg} deg,  anisotropy penalty xi = {xi}")
print("C_a(alpha=0, theta) [Pa]  (should equal isotropic plane-stress matrix):")
print(C_aniso(0.0))
print(f"Max |C_a(0)-C_iso| = {np.max(np.abs(C_aniso(0.0) - C_iso)):.3e} Pa")
print("C_a(alpha=1, theta) [Pa]  (interphase fully debonded):")
print(C_aniso(1.0))
print("omega_a(theta):")
print(omega_a)
print(f"lambda_max(omega_a) = 1 + xi = {1.0 + xi:.1f}")


# ----- Fracture Properties (SI) -----
gc_d      = 10.0       # N/m   bulk
gc_alpha  = 1.0        # N/m   interface
ell_d     = 0.2e-3     # m
ell_alpha = 0.2e-3     # m

print_section("Fracture Properties (SI)")
print(f"gc_d     = {gc_d:.4g} N/m   (bulk layer)")
print(f"gc_alpha = {gc_alpha:.4g} N/m   (interface)")


# ----- Background Grid (SI) -----
# Square 0.2 mm cells; the 0.01 x 0.01 m plate sits inside a 2-cell pad ring.
Lx_mat = 0.01              # m  (plate width)
Ly_mat = 0.01              # m  (plate height)
h      = 0.05e-3            # m  (cell size)
pad    = 2                 # cells of pad ring on every side
numx_mat = int(round(Lx_mat / h))   # 50
numy_mat = int(round(Ly_mat / h))   # 50
numx, numy = numx_mat + 2 * pad, numy_mat + 2 * pad   # 54 x 54
Lx, Ly     = numx * h, numy * h
offset     = np.array([pad * h, pad * h])             # material origin on the grid

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid (SI)")
print(f"Domain:   {Lx:.4g} x {Ly:.4g} m  (plate {Lx_mat} x {Ly_mat} m at offset {offset} m)")
print(f"Cells:    {numx} x {numy},  h = {mesh.deltax:.4g} m,  pad ring {pad} cells")
print(f"Nodes:    {mesh.nodeCount},  Elements: {mesh.elemCount}")
print(f"ell_d = ell_alpha = {ell_d:.4g} m")


# ----- Particle initialisation (1 particle per material cell) -----
noX, noY, ngp = numx_mat, numy_mat, 1
W, Q  = gauss_2D(ngp)
pmesh = Mesh(Lx_mat, Ly_mat, noX, noY)
pCount_max = noX * noY * len(W)   # 2500

all_pos  = np.zeros((pCount_max, 2))     # material coordinates
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
        a     = W[q] * detJ0            # cell area = h * h
        all_pos[pid, :] = N.T @ pts
        all_vol[pid]    = a
        all_mass[pid]   = a * rho
        pid += 1


# ----- Pre-notch: remove four rows straddling y = Ly/2, x in [0, notch] -----
notch_length = 0.005                                         # m from left
notch_rows   = np.arange(numy_mat // 2 - 2, numy_mat // 2 + 2)   # iy = 23..26

iy_all = np.floor(all_pos[:, 1] / h).astype(int)
keep   = ~((all_pos[:, 0] <= notch_length) & np.isin(iy_all, notch_rows))
n_removed = int(np.sum(~keep))

print_section("Notch")
print(f"Crack line:  y = {Ly_mat/2:.4g} m  (rows iy in {set(notch_rows.tolist())})")
print(f"Notch span:  x in [0, {notch_length}] m")
print(f"Removed:     {n_removed} particles  "
      f"(expected {int(round(notch_length / h)) * len(notch_rows)})")


# ----- Build ParticleSet from surviving particles, shifted into the pad ring -----
pCount = int(np.sum(keep))
particles = ParticleSet(pCount)
particles.positions[:]            = all_pos[keep] + offset
particles.volume[:]               = all_vol[keep]
particles.mass[:]                 = all_mass[keep]
particles.deformation_gradient[:] = np.tile([1.0, 0.0, 0.0, 1.0], (pCount, 1))
particles.set_initial_state()
# bulk damage d, interfacial damage alpha and history H start at zero inside the solver

# cpGIMP particle domain: half the particle-mesh cell in each direction
lp0 = np.tile([pmesh.deltax / 2, pmesh.deltay / 2], (pCount, 1))

print_section("Particle Initialization")
print(f"Active particles: {pCount}  (of {pCount_max} before notch)")
print(f"Total mass:       {np.sum(particles.mass):.6e} kg")
print(f"Particle domain lp0: {lp0[0, 0]:.4g} x {lp0[0, 1]:.4g} m (half-widths)")


# ----- Neumann BCs -- symmetric tension sigma on top and bottom edges (SI) -----
# Nominal traction (GeoWarp style): t_p = (0, +-sigma) on the top / bottom material
# cell row, deposited in P2G as f_I += w_I t_p h.
sigma = 2.0e6    # Pa, constant from t = 0

iy_p     = np.floor(all_pos[keep, 1] / h).astype(int)
traction = np.zeros((pCount, 2))
traction[iy_p == numy_mat - 1, 1] =  sigma     # top row    -> upward (+y)
traction[iy_p == 0, 1]            = -sigma     # bottom row -> downward (-y)
particles.neumann_particles[:] = traction[:, 1] != 0.0

print_section("Boundary Conditions")
print(f"sigma = {sigma:.4g} Pa  -- no Dirichlet constraints")
print(f"Top particles (up): {int(np.sum(traction[:, 1] > 0))},  "
      f"bottom (dn): {int(np.sum(traction[:, 1] < 0))}")


# ----- Initial Configuration Plot -----
print_section("Visualisation — Initial Configuration")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
           s=4, c='steelblue', alpha=0.6, rasterized=True, label='Material points')
nm = particles.positions[particles.neumann_particles]
ax.scatter(nm[:, 0], nm[:, 1], s=8, c='green', alpha=0.9,
           rasterized=True, label='Neumann (traction)')
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('Yvonnet plate (Warp) -- initial particle layout', fontsize=12)
ax.legend(markerscale=2)
plt.tight_layout(); plt.show()


# ----- Solver parameters -- three explicit stability limits -----
CFL   = 0.5
ndim  = 2
C_max = float(np.max(np.linalg.eigvalsh(C_aniso(0.0))))
c_d   = np.sqrt(C_max / rho)
dt_u  = h / c_d

eta_d     = 2.0 * ndim * gc_d * ell_d / (h * c_d)
eta_alpha = eta_d                                    # shared viscosity

dt_d_crit     = eta_d     * h**2 / (2.0 * ndim * gc_d     * ell_d)
dt_alpha_crit = eta_alpha * h**2 / (2.0 * ndim * gc_alpha * ell_alpha * (1.0 + xi))

dtime  = CFL * min(dt_u, dt_d_crit, dt_alpha_crit)
time   = 30e-6                                       # s   (20 us)
nsteps = int(np.floor(time / dtime))
vtk_output_dir = 'vtk_output_Yvonnet_warp'
vtk_interval   = 50

print_section("Solver Parameters (three-way CFL)")
print(f"C_max (max eig C_a(0)):    {C_max:.4e} Pa")
print(f"Wave speed c_d:            {c_d:.4e} m/s")
print(f"(1) elastodynamic  dt_u        = {dt_u:.4e} s")
print(f"(2) bulk-damage    dt_d_crit   = {dt_d_crit:.4e} s")
print(f"(3) interfacial    dt_alpha_crit = {dt_alpha_crit:.4e} s")
print(f"eta_d     = {eta_d:.4e} Pa*s")
print(f"eta_alpha = {eta_alpha:.4e} Pa*s   (shared viscosity; ratio eta_alpha/eta_d = {eta_alpha/eta_d:.3f})")
print(f"Ordering: dt_alpha_crit < dt_d_crit = dt_u ?  {dt_alpha_crit < dt_d_crit and np.isclose(dt_d_crit, dt_u)}")
print(f"dtime (SF = {CFL}):         {dtime:.4e} s")
print(f"Total time:                {time:.4e} s,  steps = {nsteps}  (VTK every {vtk_interval})")

shutil.rmtree(vtk_output_dir, ignore_errors=True)


# %%  ----- Main explicit USL-MPM loop with two damage fields (on the device) -----
solver_results = run_warp_aniso_solver(
    mesh=mesh,
    particles=particles,
    CA1=CA1,
    M2=M2,
    omega_a=omega_a,
    gc_d=gc_d,
    ell_d=ell_d,
    gc_alpha=gc_alpha,
    ell_alpha=ell_alpha,
    eta_d=eta_d,
    eta_alpha=eta_alpha,
    dtime=dtime,
    time=time,
    traction=traction,
    lp0=lp0,
    vtk_output_dir=vtk_output_dir,
    vtk_interval=vtk_interval,
)

print("\nSolver completed.")
pvd_path = write_pvd(vtk_output_dir, solver_results['vtk_entries'])
print(f"Steps: {solver_results['nsteps']},  VTK frames: {len(solver_results['vtk_entries'])}  ->  {pvd_path}")
print(f"Time loop: {solver_results['wall_time']:.3f} s for {solver_results['nsteps']} steps "
      f"({solver_results['wall_time']/solver_results['nsteps']*1e3:.3f} ms/step, incl. output)")
print(f"Max bulk damage d reached:        {solver_results['d_max'].max():.4f}")
print(f"Max interfacial damage alpha:     {solver_results['alpha_max'].max():.4f}")


# ----- Save energy history -----
e_times = solver_results['energy_times']
K, U    = solver_results['kinetic'], solver_results['elastic']
G_d     = solver_results['crack_bulk']
G_a     = solver_results['crack_interf']
D_visc  = solver_results['visc']
W_ext   = solver_results['w_ext']
miss    = solver_results['miss']
with np.errstate(divide='ignore', invalid='ignore'):
    miss_pct = np.where(W_ext > 0, 100.0 * miss / W_ext, np.nan)
np.savetxt('energy_history_Yvonnet_warp.csv',
           np.column_stack([e_times * 1e6, K, U, G_d, G_a, D_visc, W_ext, miss, miss_pct]),
           header='time_us,K,U,Gamma_d,Gamma_alpha,D_visc,W_ext,miss,miss_pct_of_w_ext',
           delimiter=',', comments='')
print(f"Energy history saved -> energy_history_Yvonnet_warp.csv  ({len(e_times)} snapshots)")


# ----- Post-processing: energy budget -----
print_section("Energy Budget at t_final  (J/m, per unit thickness)")
print(f"Kinetic                K       = {K[-1]:.6e}")
print(f"Elastic (degraded)     E_el    = {U[-1]:.6e}")
print(f"Crack surface, bulk    E_d     = {G_d[-1]:.6e}")
print(f"Crack surface, interf. E_a     = {G_a[-1]:.6e}")
print(f"Crack surface, total   E_crack = {G_d[-1] + G_a[-1]:.6e}")
print(f"Viscous dissipation    E_visc  = {D_visc[-1]:.6e}")
print(f"External work          W_ext   = {W_ext[-1]:.6e}")
print(f"Sum - W_ext (drift)            = {miss[-1]:.6e}"
      + (f"   ({100.0*miss[-1]/W_ext[-1]:+.2f} % of W_ext)" if abs(W_ext[-1]) > 0.0 else ""))
print("NOTE: drift diagnostic only -- the history field, the irreversibility")
print("      clamp, the O(dt) work rule and the one-step lag on E_d/E_a make")
print("      exact closure impossible.")


# ----- Checks -----
print_section("Checks")

# Energy budget: |miss| <= 10 % of W_ext at the end of the run
miss_rel  = abs(miss[-1]) / W_ext[-1]
energy_ok = miss_rel <= 0.10
print(f"[{'PASS' if energy_ok else 'FAIL'}] Energy budget: K + U + Gamma_d + Gamma_a + D = "
      f"{K[-1] + U[-1] + G_d[-1] + G_a[-1] + D_visc[-1]:.5e} J/m vs W_ext = {W_ext[-1]:.5e} J/m; "
      f"|miss| = {miss_rel*100:.2f} % of W_ext  (limit 10 %)")
print(f"\nAll checks {'PASS' if energy_ok else 'did NOT pass'}.")


# %%
print_section("Visualisation — Final Configuration")
d_final = solver_results['bulk_damage']
a_final = solver_results['interf_damage']
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, field, label in ((axes[0], d_final, 'bulk damage d'),
                         (axes[1], a_final, 'interfacial damage alpha')):
    sc = ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
                    s=6, c=field, cmap='inferno', vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'Yvonnet plate (Warp) -- final {label}', fontsize=12)
plt.tight_layout(); plt.show()
