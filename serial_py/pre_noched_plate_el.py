# %%
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


np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material Properties -----
# Unit system: mm, N, s, tonne  →  stress in N/mm² = MPa
E   = 32e3       # N/mm²   (= 32 GPa)
nu  = 0.2
rho = 2.45e-9    # tonne/mm³  (= 2450 kg/m³)

material = Material(E, nu, rho, stressState='PLANE_STRAIN')
print_section("Material Properties")
print(f"E = {E:.4g} N/mm²,  nu = {nu},  rho = {rho:.4g} tonne/mm³")
print("Elasticity matrix:")
print(material.elasticity_matrix)


# ----- Background Grid -----
Lx   = 100.0               # mm  (plate width)
Ly   = 40.0                # mm  (plate height)
hx   = 0.25                # mm  (cell size x)
hy   = 0.25                # mm  (cell size y)
numx = int(round(Lx / hx)) # 400
numy = int(round(Ly / hy)) # 160

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid")
print(f"Domain:   {Lx} × {Ly} mm")
print(f"Cells:    {numx} × {numy},  h = {hx} mm")
print(f"Nodes:    {mesh.nodeCount},  Elements: {mesh.elemCount}")


# ----- Particle Initialization -----
# One particle per background cell (ngp=1 → particle at cell centre).
# Volume per particle = hx * hy = 0.0625 mm² (plane-strain unit thickness 1 mm).
noX = numx   # 400
noY = numy   # 160
ngp = 1      # 1 Gauss point per element → 1 particle per background cell

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
# Two background-grid rows straddling the crack line y = Ly/2 = 20 mm,
# extending from the left edge (x = 0) to x = 50 mm (half the width).
notch_length  = 50.0               # mm from left
notch_row_lo  = numy // 2 - 1      # iy = 79  →  y ∈ [19.75, 20.00) mm
notch_row_hi  = numy // 2          # iy = 80  →  y ∈ [20.00, 20.25) mm

def _in_notch(x, y):
    iy = int(np.floor(y / hy))
    return (x <= notch_length) and (iy == notch_row_lo or iy == notch_row_hi)

keep = np.array([not _in_notch(all_pos[p, 0], all_pos[p, 1])
                 for p in range(pCount_max)], dtype=bool)

n_removed = int(np.sum(~keep))
print_section("Notch")
print(f"Crack line:  y = {Ly/2:.1f} mm  (rows iy ∈ {{{notch_row_lo}, {notch_row_hi}}})")
print(f"Notch span:  x ∈ [0, {notch_length}] mm")
print(f"y range:     [{notch_row_lo*hy:.3f}, {(notch_row_hi+1)*hy:.3f}] mm")
print(f"Removed:     {n_removed} particles  (expected ≈ {int(notch_length/hx)*2})")


# Build ParticleSet from surviving particles
pCount = int(np.sum(keep))
particles = ParticleSet(pCount)
particles.positions[:]            = all_pos[keep]
particles.volume[:]               = all_vol[keep]
particles.mass[:]                 = all_mass[keep]
particles.deformation_gradient[:] = np.tile([1.0, 0.0, 0.0, 1.0], (pCount, 1))
particles.set_initial_state()

print_section("Particle Initialization")
print(f"Active particles: {pCount}  (of {pCount_max} before notch)")
print(f"Total mass:       {np.sum(particles.mass):.6e} tonne")


# ----- Neumann BCs — symmetric tension σ on top and bottom -----
# No Dirichlet BCs are applied anywhere.
# Traction formula:  f_I += N_I(p) * σ * (V_p / hy)  [per-node, unit thickness]
# neumann_traction_y stores the signed prefactor σ/hy for each particle.
sigma = 1.0   # N/mm²  (= 1 MPa)

neumann_traction_y = np.zeros(pCount)

for p in range(pCount):
    iy = int(np.floor(particles.positions[p, 1] / hy))
    iy = max(0, min(iy, numy - 1))
    if iy == numy - 1:      # top row  → upward (+y) traction
        particles.neumann_particles[p] = True
        neumann_traction_y[p]          =  sigma / hy
    elif iy == 0:           # bottom row → downward (-y) traction
        particles.neumann_particles[p] = True
        neumann_traction_y[p]          = -sigma / hy

n_top    = int(np.sum(neumann_traction_y > 0))
n_bottom = int(np.sum(neumann_traction_y < 0))
expected_force = sigma * Lx   # total upward force on top surface [N]
print_section("Boundary Conditions")
print(f"σ = {sigma} N/mm² (MPa) — no Dirichlet constraints")
print(f"Top-surface particles (↑):    {n_top}")
print(f"Bottom-surface particles (↓): {n_bottom}")
print(f"Expected total tension force: {expected_force:.1f} N  (per unit thickness)")


# ----- Initial Configuration Plot -----
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
           s=0.5, c='steelblue', alpha=0.5, rasterized=True, label='Material points')
nm = particles.positions[particles.neumann_particles]
ax.scatter(nm[:, 0], nm[:, 1], s=2, c='green', alpha=0.9,
           rasterized=True, label='Neumann (traction)')
ax.set_aspect('equal')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('Pre-notched plate — initial particle layout', fontsize=12)
ax.legend(markerscale=5)
plt.tight_layout()
plt.show()


# ----- Element-Particle Mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)
print_section("Element-Particle Mapping")
print(f"Particles located: {sum(len(m) for m in particles.mpoints)}")
print(f"Elements occupied: {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")


# ----- Node State -----
node_state = NodeState(mesh.nodeCount)


# ----- Solver Parameters -----
c_wave = np.sqrt(material.E / material.density)   # mm/s
dtcrit = hx / c_wave                               # s  (CFL = 1 limit)
dtime  = 0.5 * dtcrit                              # s  (CFL = 0.5)
#time   = 2.0 * Lx / c_wave                        # s  (2 wave crossings across width)
#nsteps = int(np.floor(time / dtime))
nsteps = 100
time = dtime * nsteps

print_section("Solver Parameters")
print(f"Wave speed:          {c_wave:.4e} mm/s  ({c_wave*1e-3:.1f} m/s)")
print(f"Critical time step:  {dtcrit:.4e} s")
print(f"dtime (CFL 0.5):     {dtime:.4e} s")
print(f"Total time:          {time:.4e} s  ({time*1e6:.2f} μs)")
print(f"Number of steps:     {nsteps}")

shutil.rmtree('vtk_output', ignore_errors=True)


# %%  ----- Main MPM Loop -----
tol            = 1e-20           # zero-mass guard [tonne]
vtk_output_dir = 'vtk_output'
vtk_interval   = 2#max(1, nsteps // 100)   # ~100 output frames

ta, ka, sa  = [], [], []
vtk_entries = []
I_mat       = np.eye(2)
mpoints     = particles.mpoints
t           = 0.0

for istep in range(nsteps):
    node_state.reset()

    # --- P2G ---
    for e in range(mesh.elemCount):
        esctr = mesh.element[e]
        for pid in mpoints[e]:
            stress = particles.stress[pid]
            for idn in esctr:
                dx       = particles.positions[pid] - mesh.node[idn]
                N, dNdx  = get_mpm2d_shape(dx, hx, hy)

                node_state.mass[idn]              += N * particles.mass[pid]
                node_state.momentum[idn]          += N * particles.mass[pid] * particles.velocities[pid]
                node_state.internal_force[idn, 0] -= particles.volume[pid] * (stress[0]*dNdx[0] + stress[2]*dNdx[1])
                node_state.internal_force[idn, 1] -= particles.volume[pid] * (stress[2]*dNdx[0] + stress[1]*dNdx[1])
                if particles.neumann_particles[pid]:
                    node_state.external_force[idn, 1] += neumann_traction_y[pid] * N * particles.volume[pid]

    # --- Momentum update (no Dirichlet BCs) ---
    node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime

    # --- G2P ---
    k = 0.0
    u = 0.0
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            Lp = np.zeros((2, 2))
            for idn in mesh.element[e]:
                dx       = particles.positions[pid] - mesh.node[idn]
                N, dNdx  = get_mpm2d_shape(dx, hx, hy)

                vI = np.zeros(2)
                if node_state.mass[idn] > tol:
                    particles.velocities[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn]
                    particles.positions[pid]  += dtime * N * node_state.momentum[idn] / node_state.mass[idn]
                    vI = node_state.momentum[idn] / node_state.mass[idn]

                Lp += np.outer(vI, dNdx)

            F = (I_mat + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
            particles.deformation_gradient[pid] = F.reshape(4)
            particles.volume[pid]               = np.linalg.det(F) * particles.initial_volume[pid]
            dEps     = 0.5 * dtime * (Lp + Lp.T)
            dEps_vec = np.array([dEps[0, 0], dEps[1, 1], 2.0*dEps[0, 1]])
            particles.stress[pid] += material.elasticity_matrix @ dEps_vec
            particles.strain[pid] += dEps_vec

            k += 0.5 * (particles.velocities[pid, 0]**2 + particles.velocities[pid, 1]**2) * particles.mass[pid]
            u += 0.5 * particles.volume[pid] * particles.stress[pid] @ particles.strain[pid]

    ta.append(t)
    ka.append(k)
    sa.append(u)

    if istep % vtk_interval == 0:
        fname = write_particles_vtp(
            particles.positions, particles.velocities,
            particles.stress, particles.strain,
            istep, t, vtk_output_dir,
        )
        vtk_entries.append((t, fname))

    _, mpoints = build_particle_element_map(particles.positions, mesh)
    t += dtime

print("\nSolver completed.")
print(f"Steps: {nsteps},  VTK frames: {len(vtk_entries)}")

pvd_path = write_pvd(vtk_output_dir, vtk_entries)
print(f"VTK output: {pvd_path}")

solver_results = {
    'time':    np.array(ta),
    'kinetic': np.array(ka),
    'strain':  np.array(sa),
    'vtk_entries': vtk_entries,
}


# %%  ----- Post-Processing -----
t_us   = solver_results['time'] * 1e6
kin    = solver_results['kinetic']
strain = solver_results['strain']
total  = kin + strain

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(t_us, kin)
axes[0].set_xlabel('Time [μs]')
axes[0].set_ylabel('Energy [N·mm]')
axes[0].set_title('Kinetic energy')
axes[1].plot(t_us, strain)
axes[1].set_xlabel('Time [μs]')
axes[1].set_ylabel('Energy [N·mm]')
axes[1].set_title('Strain energy')
axes[2].plot(t_us, total)
axes[2].set_xlabel('Time [μs]')
axes[2].set_ylabel('Energy [N·mm]')
axes[2].set_title('Total energy (kinetic + strain)')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(particles.positions[:, 0], particles.positions[:, 1],
           s=0.5, c='steelblue', alpha=0.5, rasterized=True)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title('Pre-notched plate — final configuration', fontsize=12)
plt.tight_layout()
plt.show()
