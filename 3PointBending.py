# %%
import numpy as np
import matplotlib.pyplot as plt
import shutil

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.plot_utils import plot_mpm_domain, plot_particles
from src.solver2 import run_mpm_solver, NodeState, build_particle_element_map
from src.vtk_export import write_pvd


np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


def print_subsection(title):
    divider = "-" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material Properties — Aluminium 2024-T3 -----
rho  = 2700.0   # density           (kg/m³)
E    = 73.1e9   # Young's modulus   (Pa)
nu   = 0.33     # Poisson's ratio

# Johnson-Cook flow stress: sigma_f = [A + B*eps_p^n][1 + C*ln(eps_dot*)]
# Source: Lesuer (2000), Wierzbicki et al. (2005)
A         = 369e6  # initial yield stress    (Pa)
B         = 684e6  # hardening modulus       (Pa)
n         = 0.73   # hardening exponent
C         = 0.0083 # strain-rate sensitivity
eps_dot_0 = 1.0    # reference strain rate   (1/s)

# Johnson-Cook damage: eps_f = [D1 + D2*exp(D3*sigma*)][1 + D4*ln(eps_dot*)]
D1 =  0.13   # fracture strain offset
D2 =  0.13   # fracture strain scale
D3 = -1.5    # triaxiality sensitivity (negative: tension is more damaging)
D4 =  0.011  # strain-rate sensitivity of damage

material = Material(rho, E, nu, A, B, n, C, eps_dot_0, D1=D1, D2=D2, D3=D3, D4=D4)

print_section("Material Properties")
print(f"E = {E:.2e} Pa,  nu = {nu},  rho = {rho} kg/m³")
print(f"G = {material.G:.2e} Pa,  K = {material.K:.2e} Pa")
print(f"JC: A={A:.2e}, B={B:.2e}, n={n}, C={C}, eps_dot_0={eps_dot_0}")
print(f"P-wave speed: {material.wave_speed:.2f} m/s")

# ----- Computational grid -----
Lx   = 9
Ly   = 4
numx = 36
numy = 16

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Computational Grid")
print(f"Domain size: Lx = {Lx}, Ly = {Ly}")
print(f"Grid resolution: {numx} x {numy} elements")
print(f"Total nodes: {mesh.nodeCount},  Total elements: {mesh.elemCount}")
print(f"Cell size: dx = {mesh.deltax:.4f}, dy = {mesh.deltay:.4f}")

# ----- Particle distribution -----
Lxp = 9
Lyp = 1
noX = 32
noY = 4

pmesh = Mesh(Lxp, Lyp, noX, noY)
pmesh.node[:, 1] += 2.7 #3/2 - Lyp/2   # centre beam vertically in domain

print_section("Particle Mesh")
print(f"Domain size: Lxp = {Lxp}, Lyp = {Lyp}")
print(f"Grid resolution: {noX} x {noY} elements")

# ----- Particle initialisation -----
ngp    = 3
W, Q   = gauss_2D(ngp)
pCount = len(pmesh.element) * len(W)

particles = ParticleSet(pCount)

pid = 0
max_p_height = 0.0
for e in range(len(pmesh.element)):
    sctr = pmesh.element[e, :]
    pts  = pmesh.node[sctr, :]
    for q in range(len(W)):
        N, dNdxi = lagrange_basis_Q4(Q[q])
        J0    = dNdxi.T @ pts
        detJ0 = np.linalg.det(J0)
        a     = W[q] * detJ0
        particles.volume[pid]               = a
        particles.mass[pid]                 = a * rho
        particles.positions[pid]            = N.T @ pts
        particles.deformation_gradient[pid] = [1.0, 0.0, 0.0, 1.0]
        if particles.positions[pid, 1] > max_p_height:
            max_p_height = particles.positions[pid, 1]
        pid += 1

particles.set_initial_state()

# Neumann BC: downward traction applied at the top-centre of the beam
w  = 0.3   # semi-width of loading patch (m)
xc = 4.5   # centre of loading patch (m)
for p in range(particles.count):
    x = particles.positions[p, 0]
    if abs(x - xc) <= w and abs(particles.positions[p, 1] - max_p_height) < 1e-6:
        #particles.neumann_particles[p] = True
        particles.dirichlet_particles[p] = True

print_section("Particle Initialisation")
print(f"Number of particles: {particles.count}")
print(f"Total mass:   {np.sum(particles.mass):.4f} kg")
print(f"Total volume: {np.sum(particles.volume):.4f} m²")
print(f"Neumann BC particles: {np.sum(particles.neumann_particles)}")
print(f"Dirichlet BC particles: {np.sum(particles.dirichlet_particles)}")
# ----- Visualise initial configuration -----
print_section("Visualisation — Initial Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(12, 8), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
dirichlet_pos = particles.positions[particles.dirichlet_particles]
ax.scatter(dirichlet_pos[:, 0], dirichlet_pos[:, 1],
           s=32, c='green', marker='o', alpha=0.9, label='Dirichlet BC')
ax.set_title('Initial Particle Positions', fontsize=14, fontweight='bold')
plt.show()

# ----- Element-particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)

print_section("Element-Particle Mapping")
print(f"Particles located: {sum(len(m) for m in particles.mpoints)}")
print(f"Elements with particles: {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")

# ----- Solver setup -----
node_state  = NodeState(mesh.nodeCount)
v_dirichlet = [0.0, -30]   # indenter speed: downward at 1.0 m/s

dtcrit = mesh.deltax / material.wave_speed
dtime  = 0.5 * dtcrit      # safety factor of 0.5 on CFL
time   = 3500 * dtime       # total simulation time

total_displacement = v_dirichlet[1] * time

print_section("Solver")
print(f"CFL critical dt: {dtcrit:.2e} s")
print(f"Using dt:        {dtime:.2e} s  (safety factor 0.5)")
print(f"Total time:      {time:.2e} s  ({int(time/dtime)} steps)")
print(f"Indenter speed:  {v_dirichlet} m/s")
print(f"Total indenter displacement: {total_displacement:.4f} m")

shutil.rmtree('vtk_output', ignore_errors=True)

# %%
solver_results = run_mpm_solver(
    mesh=mesh,
    particles=particles,
    material=material,
    traction=0.0,
    g=9.81,
    dtime=dtime,
    time=time,
    alpha=0.99,
    node_state=node_state,
    vtk_output_dir='vtk_output',
    vtk_interval=10,
    v_dirichlet=v_dirichlet,
)

print_section("Solver Complete")
print(f"Steps run:       {len(solver_results['time'])}")
#print(f"Final KE:        {solver_results['kinetic'][-1]:.4e} J")

pvd_path = write_pvd('vtk_output', solver_results['vtk_entries'])
print(f"VTK steps saved: {len(solver_results['vtk_entries'])}  →  {pvd_path}")

# %%
# ----- Visualise final configuration -----
print_section("Visualisation — Final Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(12, 8), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title('Final Particle Positions', fontsize=14, fontweight='bold')
plt.show()
