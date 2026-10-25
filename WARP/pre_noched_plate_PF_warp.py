# %%
# =============================================================================
#  Pre-notched plate  --  Explicit USL-MPM with AT2 phase-field fracture, on the GPU
#
#  Warp port of `serial_py/pre_noched_plate_PF.py` (Algorithm 1 of
#  `explicit_USL_MPM_PF.tex`), with the cpGIMP basis and a convected particle
#  domain. Spec: .scratch/warp-phase-field-plate/spec.md.
#
#  Changes from the numpy reference:
#    * background grid gets a 2-cell pad ring on all four sides;
#    * the notch removes four particle rows (iy = 78..81), not two, so the
#      convected cpGIMP supports of the two notch faces do not bridge;
#    * the run is T = 60 us long; no environment-variable overrides.
#
#  Phase-field parameters for this problem:
#      ell = 0.25 mm,  Gc = 3e-3 N/mm,  eta = 1e-9 MPa*s   (AT2 model)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import shutil
import warp as wp

from src.warp_small_strain import PlaneStrainMaterial
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.phasefield import PhaseFieldModel
from src.warp_usl_pf import CRACKED_PHI, run_warp_pf_solver
from src.vtk_export import write_pvd

np.set_printoptions(precision=3, suppress=True)
wp.init()
wp.set_device("cuda:0")


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material Properties -----
# Unit system: mm, N, s, tonne  ->  stress in N/mm^2 = MPa
E   = 32e3       # N/mm^2   (= 32 GPa)
nu  = 0.2
rho = 2.45e-9    # tonne/mm^3  (= 2450 kg/m^3)

material = PlaneStrainMaterial(E, nu, rho)
print_section("Material Properties")
print(f"E = {E:.4g} N/mm^2,  nu = {nu},  rho = {rho:.4g} tonne/mm^3  (plane strain, small strain)")
print("Elasticity matrix:")
print(material.elasticity_matrix)


# ----- Phase-field Properties -----
ell = 0.25       # mm        regularisation length scale
Gc  = 3e-3       # N/mm      critical energy release rate
eta = 1e-9       # MPa*s     phase-field viscosity (artificial)
pf  = PhaseFieldModel(Gc=Gc, ell=ell, eta=eta, model='AT2', kappa=1e-7)

print_section("Phase-field Properties")
print(f"Model:  {pf.model}   (c_w = {pf.c_w:.4f})")
print(f"ell = {ell} mm,  Gc = {Gc:.4g} N/mm,  eta = {eta:.4g} MPa*s,  kappa = {pf.kappa:.1e}")
print(f"Homogeneous critical stress sigma_c = {pf.sigma_c(E):.4g} N/mm^2")


# ----- Background Grid -----
# Square 0.25 mm cells; the 100 x 40 mm plate sits inside a 2-cell pad ring.
Lx_mat = 100.0             # mm  (plate width)
Ly_mat = 40.0              # mm  (plate height)
h      = 0.25              # mm  (cell size)
pad    = 2                 # cells of pad ring on every side
numx_mat = int(round(Lx_mat / h))   # 400
numy_mat = int(round(Ly_mat / h))   # 160
numx, numy = numx_mat + 2 * pad, numy_mat + 2 * pad   # 404 x 164
Lx, Ly     = numx * h, numy * h
offset     = np.array([pad * h, pad * h])             # material origin on the grid

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid")
print(f"Domain:   {Lx} x {Ly} mm  (plate {Lx_mat} x {Ly_mat} mm at offset {offset} mm)")
print(f"Cells:    {numx} x {numy},  h = {mesh.deltax} mm,  pad ring {pad} cells")
print(f"Nodes:    {mesh.nodeCount},  Elements: {mesh.elemCount}")


# ----- Particle Initialization -----
# One particle per material cell (ngp=1 -> particle at cell centre).
noX = numx_mat   # 400
noY = numy_mat   # 160
ngp = 1

W, Q = gauss_2D(ngp)
pmesh = Mesh(Lx_mat, Ly_mat, noX, noY)

pCount_max = noX * noY * len(W)   # 64 000

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
        all_pos[pid, :]  = N.T @ pts
        all_vol[pid]     = a
        all_mass[pid]    = a * rho
        pid += 1


# ----- Notch -----
# Geometric pre-notch: remove four particle rows straddling y = Ly/2,
# from the left edge (x = 0) to x = 50 mm (half the width).
notch_length = 50.0                                  # mm from left
notch_rows   = np.arange(numy_mat // 2 - 2, numy_mat // 2 + 2)   # iy = 78..81
y_notch      = Ly_mat / 2                            # crack line, material coordinates
crack_band   = 4 * ell                               # crack-path band half-width around it

iy_all = np.floor(all_pos[:, 1] / h).astype(int)
keep   = ~((all_pos[:, 0] <= notch_length) & np.isin(iy_all, notch_rows))

n_removed = int(np.sum(~keep))
print_section("Notch")
print(f"Crack line:  y = {y_notch:.1f} mm  (rows iy in {set(notch_rows.tolist())}, "
      f"y in [{notch_rows[0]*h}, {(notch_rows[-1]+1)*h}) mm)")
print(f"Notch span:  x in [0, {notch_length}] mm")
print(f"Removed:     {n_removed} particles  (expected {int(notch_length/h)*len(notch_rows)})")


# Build ParticleSet from surviving particles, shifted into the pad ring
pCount = int(np.sum(keep))
particles = ParticleSet(pCount)
particles.positions[:]            = all_pos[keep] + offset
particles.volume[:]               = all_vol[keep]
particles.mass[:]                 = all_mass[keep]
particles.deformation_gradient[:] = np.tile([1.0, 0.0, 0.0, 1.0], (pCount, 1))
particles.set_initial_state()
# bulk damage and history field start at zero (intact material) inside the solver

# cpGIMP particle domain: half the particle-mesh cell in each direction
lp0 = np.tile([pmesh.deltax / 2, pmesh.deltay / 2], (pCount, 1))

print_section("Particle Initialization")
print(f"Active particles: {pCount}  (of {pCount_max} before notch)")
print(f"Total mass:       {np.sum(particles.mass):.6e} tonne")
print(f"Particle domain lp0: {lp0[0, 0]:.3f} x {lp0[0, 1]:.3f} mm (half-widths)")


# ----- Neumann BCs -- symmetric tension sigma on top and bottom -----
# Nominal traction (GeoWarp style): t_p = (0, +-sigma) on the top / bottom material
# cell row, deposited in P2G as f_I += w_I t_p h.
sigma = 1.0      # N/mm^2  (= 1 MPa), constant from t = 0

iy_p     = np.floor(all_pos[keep, 1] / h).astype(int)
traction = np.zeros((pCount, 2))
traction[iy_p == numy_mat - 1, 1] =  sigma     # top row    -> upward (+y)
traction[iy_p == 0, 1]            = -sigma     # bottom row -> downward (-y)
particles.neumann_particles[:] = traction[:, 1] != 0.0

n_top    = int(np.sum(traction[:, 1] > 0))
n_bottom = int(np.sum(traction[:, 1] < 0))
print_section("Boundary Conditions")
print(f"sigma = {sigma} N/mm^2 (MPa) -- no Dirichlet constraints")
print(f"Top-surface particles (up):    {n_top}")
print(f"Bottom-surface particles (dn): {n_bottom}")


# ----- Initial Configuration Plot -----
print_section("Visualisation — Initial Configuration")
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
           s=0.5, c='steelblue', alpha=0.5, rasterized=True, label='Material points')
nm = particles.positions[particles.neumann_particles]
ax.scatter(nm[:, 0], nm[:, 1], s=2, c='green', alpha=0.9,
           rasterized=True, label='Neumann (traction)')
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('Pre-notched plate (Warp) -- initial particle layout', fontsize=12)
ax.legend(markerscale=5)
plt.tight_layout(); plt.show()


# ----- Solver Parameters -----
# Two explicit stability limits act simultaneously; take the smaller (Sec. 6).
c_wave   = np.sqrt(material.E / material.density)      # mm/s  (dilatational proxy)
dt_cfl   = h / c_wave                                  # s   elastodynamic CFL limit
dt_pf    = pf.dt_crit_phase(h, ndim=2)                 # s   parabolic diffusion limit
CFL      = 0.5                                         # safety factor
dtime    = CFL * min(dt_cfl, dt_pf)
time     = 60e-6                                       # s   (60 us)
nsteps   = int(np.floor(time / dtime))
vtk_output_dir = 'vtk_output_PF_warp'
vtk_interval   = 50

print_section("Solver Parameters")
print(f"Wave speed:            {c_wave:.4e} mm/s")
print(f"CFL (elastodynamic):   dt <= {dt_cfl:.4e} s")
print(f"Parabolic (phase):     dt <= {dt_pf:.4e} s")
print(f"Governing limit:       {'phase field' if dt_pf < dt_cfl else 'elastodynamic'}")
print(f"dtime (safety {CFL}):    {dtime:.4e} s")
print(f"Total time:            {time:.4e} s  ({time*1e6:.3f} us)")
print(f"Number of steps:       {nsteps}  (VTK every {vtk_interval})")

shutil.rmtree(vtk_output_dir, ignore_errors=True)


# %%  ----- Main USL-MPM-PF Loop (on the device) -----
solver_results = run_warp_pf_solver(
    mesh=mesh,
    particles=particles,
    material=material,
    pf=pf,
    dtime=dtime,
    time=time,
    traction=traction,
    lp0=lp0,
    vtk_output_dir=vtk_output_dir,
    vtk_interval=vtk_interval,
    crack_origin=offset,
    crack_line_y=y_notch,
    crack_band=crack_band,
)

print("\nSolver completed.")
pvd_path = write_pvd(vtk_output_dir, solver_results['vtk_entries'])
print(f"Steps: {solver_results['nsteps']},  VTK frames: {len(solver_results['vtk_entries'])}  ->  {pvd_path}")
print(f"Time loop: {solver_results['wall_time']:.3f} s for {solver_results['nsteps']} steps "
      f"({solver_results['wall_time']/solver_results['nsteps']*1e3:.3f} ms/step, incl. output)")
print(f"Max damage phi reached: {solver_results['phi_max'].max():.4f}")


# ----- Save energy and crack-tip histories -----
e_times = solver_results['energy_times']
K, U    = solver_results['kinetic'], solver_results['elastic']
G_bulk  = solver_results['crack_bulk']
D_visc  = solver_results['visc']
W_ext   = solver_results['w_ext']
miss    = solver_results['miss']
with np.errstate(divide='ignore', invalid='ignore'):
    miss_pct = np.where(W_ext > 0, 100.0 * miss / W_ext, np.nan)
np.savetxt('energy_history_PF_warp.csv',
           np.column_stack([e_times * 1e6, K, U, G_bulk, D_visc, W_ext, miss, miss_pct]),
           header='time_us,kinetic_Nmm,elastic_Nmm,crack_bulk_Nmm,visc_Nmm,w_ext_Nmm,miss_Nmm,miss_pct_of_w_ext',
           delimiter=',', comments='')
print(f"Energy history saved -> energy_history_PF_warp.csv  ({len(e_times)} snapshots)")

tip_x = solver_results['tip_x']
np.savetxt('crack_tip_PF_warp.csv',
           np.column_stack([solver_results['tip_times'] * 1e6, tip_x, solver_results['tip_phi_max']]),
           header='time_us,x_tip_mm,phi_max', delimiter=',', comments='')
print(f"Crack-tip history saved -> crack_tip_PF_warp.csv  ({len(tip_x)} snapshots)")


# ----- Post-processing: energy budget -----
print_section("Energy Budget (per unit thickness, N*mm)")
print(f"Kinetic K:              {K[-1]:.5e}")
print(f"Stored elastic U:       {U[-1]:.5e}")
print(f"Crack surface Gamma:    {G_bulk[-1]:.5e}")
print(f"Viscous dissipation D:  {D_visc[-1]:.5e}")
print(f"External work W_ext:    {W_ext[-1]:.5e}")
print(f"Miss (K+U+Gamma+D-W):   {miss[-1]:.5e}")
x_tip_final = tip_x[-1]
print(f"Final crack tip:        x = {x_tip_final:.3f} mm  (notch tip at {notch_length} mm)")


# ----- Checks -----
print_section("Checks")

# 1. Energy budget: |miss| <= 10 % of W_ext at the end of the run
miss_rel  = abs(miss[-1]) / W_ext[-1]
energy_ok = miss_rel <= 0.10
print(f"[{'PASS' if energy_ok else 'FAIL'}] Energy budget: K + U + Gamma + D = "
      f"{K[-1] + U[-1] + G_bulk[-1] + D_visc[-1]:.5e} N*mm vs W_ext = {W_ext[-1]:.5e} N*mm; "
      f"|miss| = {miss_rel*100:.2f} % of W_ext  (limit 10 %)")

# 2. Crack path: phi > 0.9 only near the notch line, and the crack grew past the notch tip
phi_final = solver_results['phase']
x_mat     = particles.positions - offset
cracked   = phi_final > CRACKED_PHI
dev_y     = np.abs(x_mat[cracked, 1] - y_notch)
max_dev   = dev_y.max() if cracked.any() else 0.0
n_off     = int(np.sum(dev_y >= crack_band))
path_ok   = cracked.any() and n_off == 0 and x_tip_final > notch_length
print(f"[{'PASS' if path_ok else 'FAIL'}] Crack path: {int(cracked.sum())} particles with phi > {CRACKED_PHI}, "
      f"max |y - y_notch| = {max_dev:.3f} mm ({n_off} at >= 4 ell = {crack_band} mm); "
      f"x_tip = {x_tip_final:.3f} mm  (limit > {notch_length} mm)")

all_ok = energy_ok and path_ok
print(f"\nAll checks {'PASS' if all_ok else 'did NOT pass'}.")


# %%
print_section("Visualisation — Final Configuration")
fig, ax = plt.subplots(figsize=(14, 6))
sc = ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
                s=1.0, c=phi_final, cmap='inferno', vmin=0, vmax=1, rasterized=True)
plt.colorbar(sc, ax=ax, label='bulk damage phi')
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.set_title('Pre-notched plate (Warp) -- final bulk damage', fontsize=12)
plt.tight_layout(); plt.show()
