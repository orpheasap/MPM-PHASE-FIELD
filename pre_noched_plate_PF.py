# %%
# =============================================================================
#  Pre-notched plate  --  Explicit USL-MPM with variational phase-field fracture
#
#  Adaptation of `pre_noched_plate_el.py` that incorporates the phase-field (PF)
#  additions of `explicit_USL_MPM_PF.tex` (Algorithm 1, "Solution procedure of
#  explicit USL-MPM with phase-field fracture").
#
#  New per-step ingredients relative to the elastic driver:
#    * P2G  : pseudo-mass  m_I^phi = sum_p N_I V_p^0  and the particle phase
#             field are mapped to the grid; internal force uses the *degraded*
#             stress  sigma = g(phi) sigma_0.
#    * Damage residual F_I is assembled from the grid phase field / its gradient.
#    * Explicit grid phase update:  phi_I^{t+dt} = phi_I^t - dt/(eta m_I^phi) F_I
#    * G2P  : particle phase interpolated back, irreversibility enforced,
#             history field H_p refreshed, effective + degraded stress updated.
#
#  Phase-field parameters for this problem:
#      ell = 0.25 mm,  Gc = 3e-3 N/mm,  eta = 1e-9 MPa*s   (AT2 model)
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
from src.vtk_export import write_pvd, write_particles_vtp
from src.phasefield import PhaseFieldModel, elastic_energy_density


np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material Properties -----
# Unit system: mm, N, s, tonne  ->  stress in N/mm^2 = MPa
E   = 32e3       # N/mm^2   (= 32 GPa)
nu  = 0.2
rho = 2.45e-9    # tonne/mm^3  (= 2450 kg/m^3)

material = Material(E, nu, rho, stressState='PLANE_STRAIN')
print_section("Material Properties")
print(f"E = {E:.4g} N/mm^2,  nu = {nu},  rho = {rho:.4g} tonne/mm^3")
print("Elasticity matrix:")
print(material.elasticity_matrix)


# ----- Phase-field Properties -----
ell = 0.25       # mm        regularisation length scale
Gc  = 3e-3       # N/mm      critical energy release rate
eta = 1e-9       # MPa*s     phase-field viscosity (artificial)
pf  = PhaseFieldModel(Gc=Gc, ell=ell, eta=eta, model='AT2', kappa=1e-7)

print_section("Phase-field Properties")
print(f"Model:  {pf.model}   (c_w = {pf.c_w:.4f})")
print(f"ell = {ell} mm,  Gc = {Gc:.4g} N/mm,  eta = {eta:.4g} MPa*s")
print(f"Homogeneous critical stress sigma_c = {pf.sigma_c(E):.4g} N/mm^2")


# ----- Background Grid -----
# Optional coarsening for quick smoke tests:  PF_H_SCALE multiplies the cell
# size, PF_NSTEPS overrides the step count (defaults reproduce the real problem).
_h_scale = float(os.environ.get("PF_H_SCALE", "1.0"))

Lx   = 100.0               # mm  (plate width)
Ly   = 40.0                # mm  (plate height)
hx   = 0.25 * _h_scale     # mm  (cell size x)
hy   = 0.25 * _h_scale     # mm  (cell size y)
numx = int(round(Lx / hx)) # 400
numy = int(round(Ly / hy)) # 160

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid")
print(f"Domain:   {Lx} x {Ly} mm")
print(f"Cells:    {numx} x {numy},  h = {hx} mm")
print(f"Nodes:    {mesh.nodeCount},  Elements: {mesh.elemCount}")


# ----- Particle Initialization -----
# One particle per background cell (ngp=1 -> particle at cell centre).
noX = numx   # 400
noY = numy   # 160
ngp = 1      # 1 Gauss point per element -> 1 particle per background cell

W, Q = gauss_2D(ngp)
pmesh = Mesh(Lx, Ly, noX, noY)

pCount_max = noX * noY * len(W)   # 64 000

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
        a     = W[q] * detJ0            # cell area = hx * hy
        all_pos[pid, :]  = N.T @ pts
        all_vol[pid]     = a
        all_mass[pid]    = a * rho
        pid += 1


# ----- Notch -----
# Geometric pre-notch: remove two background-grid rows straddling y = Ly/2,
# from the left edge (x = 0) to x = 50 mm (half the width).
notch_length  = 50.0               # mm from left
notch_row_lo  = numy // 2 - 1      # iy = 79  ->  y in [19.75, 20.00) mm
notch_row_hi  = numy // 2          # iy = 80  ->  y in [20.00, 20.25) mm

def _in_notch(x, y):
    iy = int(np.floor(y / hy))
    return (x <= notch_length) and (iy == notch_row_lo or iy == notch_row_hi)

keep = np.array([not _in_notch(all_pos[p, 0], all_pos[p, 1])
                 for p in range(pCount_max)], dtype=bool)

n_removed = int(np.sum(~keep))
print_section("Notch")
print(f"Crack line:  y = {Ly/2:.1f} mm  (rows iy in {{{notch_row_lo}, {notch_row_hi}}})")
print(f"Notch span:  x in [0, {notch_length}] mm")
print(f"Removed:     {n_removed} particles  (expected ~ {int(notch_length/hx)*2})")


# Build ParticleSet from surviving particles
pCount = int(np.sum(keep))
particles = ParticleSet(pCount)
particles.positions[:]            = all_pos[keep]
particles.volume[:]               = all_vol[keep]
particles.mass[:]                 = all_mass[keep]
particles.deformation_gradient[:] = np.tile([1.0, 0.0, 0.0, 1.0], (pCount, 1))
particles.set_initial_state()
# phase field and history start at zero (intact material)
particles.phase[:]   = 0.0
particles.history[:] = 0.0

print_section("Particle Initialization")
print(f"Active particles: {pCount}  (of {pCount_max} before notch)")
print(f"Total mass:       {np.sum(particles.mass):.6e} tonne")


# ----- Neumann BCs -- symmetric tension sigma on top and bottom -----
sigma = float(os.environ.get("PF_SIGMA", "1.0"))   # N/mm^2  (= 1 MPa)

neumann_traction_y = np.zeros(pCount)

for p in range(pCount):
    iy = int(np.floor(particles.positions[p, 1] / hy))
    iy = max(0, min(iy, numy - 1))
    if iy == numy - 1:      # top row  -> upward (+y) traction
        particles.neumann_particles[p] = True
        neumann_traction_y[p]          =  sigma / hy
    elif iy == 0:           # bottom row -> downward (-y) traction
        particles.neumann_particles[p] = True
        neumann_traction_y[p]          = -sigma / hy

n_top    = int(np.sum(neumann_traction_y > 0))
n_bottom = int(np.sum(neumann_traction_y < 0))
print_section("Boundary Conditions")
print(f"sigma = {sigma} N/mm^2 (MPa) -- no Dirichlet constraints")
print(f"Top-surface particles (up):    {n_top}")
print(f"Bottom-surface particles (dn): {n_bottom}")


# ----- Initial Configuration Plot -----
SHOW_PLOTS = False   # set True for interactive runs
if SHOW_PLOTS:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
               s=0.5, c='steelblue', alpha=0.5, rasterized=True, label='Material points')
    nm = particles.positions[particles.neumann_particles]
    ax.scatter(nm[:, 0], nm[:, 1], s=2, c='green', alpha=0.9,
               rasterized=True, label='Neumann (traction)')
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title('Pre-notched plate -- initial particle layout', fontsize=12)
    ax.legend(markerscale=5)
    plt.tight_layout(); plt.show()


# ----- Element-Particle Mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)
print_section("Element-Particle Mapping")
print(f"Particles located: {sum(len(m) for m in particles.mpoints)}")
print(f"Elements occupied: {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")


# ----- Node State -----
node_state = NodeState(mesh.nodeCount)


# ----- Solver Parameters -----
# Two explicit stability limits act simultaneously; take the smaller (Sec. 6).
c_wave   = np.sqrt(material.E / material.density)      # mm/s  (dilatational proxy)
dt_cfl   = hx / c_wave                                 # s   elastodynamic CFL limit
dt_pf    = pf.dt_crit_phase(min(hx, hy), ndim=2)       # s   parabolic diffusion limit
CFL      = 0.5                                         # safety factor
dtime    = CFL * min(dt_cfl, dt_pf)
nsteps   = int(os.environ.get("PF_NSTEPS", "4"))
time     = dtime * nsteps

print_section("Solver Parameters")
print(f"Wave speed:            {c_wave:.4e} mm/s")
print(f"CFL (elastodynamic):   dt <= {dt_cfl:.4e} s")
print(f"Parabolic (phase):     dt <= {dt_pf:.4e} s")
print(f"Governing limit:       {'phase field' if dt_pf < dt_cfl else 'elastodynamic'}")
print(f"dtime (safety {CFL}):    {dtime:.4e} s")
print(f"Total time:            {time:.4e} s  ({time*1e6:.3f} us)")
print(f"Number of steps:       {nsteps}")

shutil.rmtree('vtk_output', ignore_errors=True)


# %%  ----- Main USL-MPM-PF Loop -----
tol            = 1e-20           # zero-mass guard [tonne]
tol_phi        = 1e-30           # zero pseudo-mass guard [mm^2]
vtk_output_dir = 'vtk_output'
vtk_interval   = 1

ta, ka, sa, da = [], [], [], []   # time, kinetic, strain, max-damage histories
vtk_entries    = []
I_mat          = np.eye(2)
mpoints        = particles.mpoints
t              = 0.0

# Cached PF scalars
C_mat       = material.elasticity_matrix
crack_coeff = pf.crack_force_coeff          # Gc / (2 c_w ell)
l2          = pf.ell ** 2
kappa       = pf.kappa
inv_eta     = 1.0 / pf.eta

for istep in range(nsteps):
    node_state.reset()

    # --- P2G : mechanical + phase-field mapping ---
    for e in range(mesh.elemCount):
        esctr = mesh.element[e]
        for pid in mpoints[e]:
            stress = particles.stress[pid]          # degraded Cauchy stress  g(phi) sigma_0
            phi_p  = particles.phase[pid]
            V0     = particles.initial_volume[pid]
            for idn in esctr:
                dx       = particles.positions[pid] - mesh.node[idn]
                N, dNdx  = get_mpm2d_shape(dx, hx, hy)

                node_state.mass[idn]              += N * particles.mass[pid]
                node_state.momentum[idn]          += N * particles.mass[pid] * particles.velocities[pid]
                node_state.internal_force[idn, 0] -= particles.volume[pid] * (stress[0]*dNdx[0] + stress[2]*dNdx[1])
                node_state.internal_force[idn, 1] -= particles.volume[pid] * (stress[2]*dNdx[0] + stress[1]*dNdx[1])
                if particles.neumann_particles[pid]:
                    node_state.external_force[idn, 1] += neumann_traction_y[pid] * N * particles.volume[pid]

                # phase-field pseudo-mass and mass-weighted phase projection
                node_state.pseudo_mass[idn] += N * V0
                node_state.phase_num[idn]   += N * V0 * phi_p

    # --- Grid phase field  phi_I^t = (sum_p N_I V0 phi_p) / m_I^phi ---
    has_phi = node_state.pseudo_mass > tol_phi
    node_state.phase[has_phi] = node_state.phase_num[has_phi] / node_state.pseudo_mass[has_phi]

    # --- Assemble damage residual F_I  (needs grad phi from the grid) ---
    for e in range(mesh.elemCount):
        esctr = mesh.element[e]
        for pid in mpoints[e]:
            V0    = particles.initial_volume[pid]
            phi_p = particles.phase[pid]
            H_p   = particles.history[pid]
            gp    = pf.dg(phi_p)            # g'(phi)
            wp    = pf.dw(phi_p)            # w'(phi)

            # phase-field gradient at the particle from the grid nodes
            gradphi = np.zeros(2)
            shp = []
            for idn in esctr:
                dx      = particles.positions[pid] - mesh.node[idn]
                N, dNdx = get_mpm2d_shape(dx, hx, hy)
                gradphi += dNdx * node_state.phase[idn]
                shp.append((idn, N, dNdx))

            for idn, N, dNdx in shp:
                node_state.damage_force[idn] += (
                    gp * N * H_p
                    + crack_coeff * (0.5 * wp * N + l2 * (dNdx @ gradphi))
                ) * V0

    # --- Update momenta (no Dirichlet BCs) ---
    node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime

    # --- Explicit grid phase-field update:  phi_I <- phi_I - dt/(eta m_I^phi) F_I ---
    phi_grid_new = node_state.phase.copy()
    phi_grid_new[has_phi] -= (dtime * inv_eta) * node_state.damage_force[has_phi] / node_state.pseudo_mass[has_phi]

    # --- G2P : mechanical update + phase / history / stress (USL) ---
    k = 0.0
    u = 0.0
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            Lp      = np.zeros((2, 2))
            phi_new = 0.0
            for idn in mesh.element[e]:
                dx       = particles.positions[pid] - mesh.node[idn]
                N, dNdx  = get_mpm2d_shape(dx, hx, hy)

                vI = np.zeros(2)
                if node_state.mass[idn] > tol:
                    particles.velocities[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn]
                    particles.positions[pid]  += dtime * N * node_state.momentum[idn] / node_state.mass[idn]
                    vI = node_state.momentum[idn] / node_state.mass[idn]

                Lp      += np.outer(vI, dNdx)
                phi_new += N * phi_grid_new[idn]

            # --- Phase field: map back, clip, enforce irreversibility ---
            phi_new = min(max(phi_new, 0.0), 1.0)
            phi_new = max(particles.phase[pid], phi_new)     # phi_{t+dt} >= phi_t
            particles.phase[pid] = phi_new

            # --- Kinematics ---
            F = (I_mat + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
            particles.deformation_gradient[pid] = F.reshape(4)
            particles.volume[pid]               = np.linalg.det(F) * particles.initial_volume[pid]
            dEps     = 0.5 * dtime * (Lp + Lp.T)
            dEps_vec = np.array([dEps[0, 0], dEps[1, 1], 2.0*dEps[0, 1]])
            particles.strain[pid] += dEps_vec

            # --- Stress update (last): effective then degraded ---
            particles.eff_stress[pid] += C_mat @ dEps_vec          # sigma_0 = C : eps
            g_phi = pf.g(phi_new)
            particles.stress[pid] = (g_phi + kappa) * particles.eff_stress[pid]

            # --- History field (monotone crack driving force) ---
            psi_e0 = elastic_energy_density(particles.eff_stress[pid], particles.strain[pid])
            if psi_e0 > particles.history[pid]:
                particles.history[pid] = psi_e0

            k += 0.5 * (particles.velocities[pid, 0]**2 + particles.velocities[pid, 1]**2) * particles.mass[pid]
            u += 0.5 * particles.volume[pid] * particles.stress[pid] @ particles.strain[pid]

    ta.append(t)
    ka.append(k)
    sa.append(u)
    da.append(float(np.max(particles.phase)))

    if istep % vtk_interval == 0:
        fname = write_particles_vtp(
            particles.positions, particles.velocities,
            particles.stress, particles.strain,
            istep, t, vtk_output_dir,
            phase=particles.phase, history=particles.history,
        )
        vtk_entries.append((t, fname))

    _, mpoints = build_particle_element_map(particles.positions, mesh)
    t += dtime

print("\nSolver completed.")
print(f"Steps: {nsteps},  VTK frames: {len(vtk_entries)}")
print(f"Max damage phi reached: {max(da):.4f}")

# ----- Save checkpoint (resume file) -----
_ckpt_path = os.path.join(vtk_output_dir, 'checkpoint.npz')
np.savez_compressed(
    _ckpt_path,
    pCount               = np.array(pCount),
    positions            = particles.positions,
    velocities           = particles.velocities,
    mass                 = particles.mass,
    volume               = particles.volume,
    deformation_gradient = particles.deformation_gradient,
    stress               = particles.stress,
    eff_stress           = particles.eff_stress,
    strain               = particles.strain,
    phase                = particles.phase,
    history              = particles.history,
    initial_positions    = particles.initial_positions,
    initial_volume       = particles.initial_volume,
    neumann_particles    = particles.neumann_particles,
    neumann_traction_y   = neumann_traction_y,
    t                    = np.array(t),
    istep_start          = np.array(nsteps),
    ta                   = np.array(ta),
    ka                   = np.array(ka),
    sa                   = np.array(sa),
    da                   = np.array(da),
    vtk_times            = np.array([e[0] for e in vtk_entries]),
    vtk_fnames           = np.array([e[1] for e in vtk_entries]),
)
print(f"Checkpoint saved  ->  {_ckpt_path}")

pvd_path = write_pvd(vtk_output_dir, vtk_entries)
print(f"VTK output: {pvd_path}")

solver_results = {
    'time':    np.array(ta),
    'kinetic': np.array(ka),
    'strain':  np.array(sa),
    'damage':  np.array(da),
    'vtk_entries': vtk_entries,
}


# %%  ----- Post-Processing -----
if SHOW_PLOTS:
    t_us   = solver_results['time'] * 1e6
    kin    = solver_results['kinetic']
    strain = solver_results['strain']
    dmg    = solver_results['damage']
    total  = kin + strain

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(t_us, kin);   axes[0].set_title('Kinetic energy')
    axes[1].plot(t_us, strain);axes[1].set_title('Strain energy')
    axes[2].plot(t_us, total); axes[2].set_title('Total energy')
    axes[3].plot(t_us, dmg);   axes[3].set_title('Max damage phi')
    for ax in axes[:3]:
        ax.set_xlabel('Time [us]'); ax.set_ylabel('Energy [N*mm]')
    axes[3].set_xlabel('Time [us]'); axes[3].set_ylabel('phi_max')
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(figsize=(14, 6))
    sc = ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
                    s=1.0, c=particles.phase, cmap='inferno', vmin=0, vmax=1,
                    rasterized=True)
    plt.colorbar(sc, ax=ax, label='phase field phi')
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title('Pre-notched plate -- final damage field', fontsize=12)