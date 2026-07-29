import numpy as np
import matplotlib.pyplot as plt

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.plot_utils import plot_mpm_domain, plot_particles
from src.solver_utilities import NodeState, build_particle_element_map


np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


def print_subsection(title):
    divider = "-" * 70
    print(f"\n{divider}\n{title}\n{divider}")




#----- Material Properties ------
E = 110e9  # Young's modulus in Pascals
nu = 0.34   # Poisson's ratio
rho = 4430  # Density in kg/m^3

material = Material(E, nu, rho, stressState='PLANE_STRAIN')
print_section("Material Properties")
print(f"Stress state: {material.stressState}")
print_subsection("Elasticity Matrix (Plane Strain)")
print(material.elasticity_matrix)

#----- Computational grid ------
Lx = 9
Ly = 3

numx = 36  # number of elements along X direction
numy = 12  # number of elements along Y direction

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Computational Grid")
print(f"Domain size: Lx = {Lx}, Ly = {Ly}")
print(f"Grid resolution: {numx} elements in x, {numy} elements in y")
print(f"Total nodes: {mesh.nodeCount}")
print(f"Total elements: {mesh.elemCount}")
print(f"Cell size: dx = {mesh.deltax:.4f}, dy = {mesh.deltay:.4f}")

#----- Particle distribution ------
Lxp = 9
Lyp = 1

noX = 32
noY = 4

pmesh = Mesh(Lxp, Lyp, noX, noY)  # particle mesh
pmesh.node[:, 1] = pmesh.node[:, 1] + 3/2 - Lyp/2
print_section("Particle Mesh")
print(f"Domain size: Lxp = {Lxp}, Lyp = {Lyp}")
print(f"Grid resolution: {noX} elements in x, {noY} elements in y")
print(f"Total nodes: {pmesh.nodeCount}")
print(f"Total elements: {pmesh.elemCount}")
print(f"Particle mesh Y-offset applied: {3/2 - Lyp/2:.4f}")

#----- Particle initialization ------
ngp = 3  # Gauss points per direction
W, Q = gauss_2D(ngp)  # Gauss quadrature for 2D

pCount = len(pmesh.element) * len(W)

particles = ParticleSet(pCount)

# Initialize particle position, mass, volume, velocity
# Particles as Gauss points of the particle mesh
pid = 0
max_p_height = 0.0
for e in range(len(pmesh.element)):
    sctr = pmesh.element[e, :]
    pts = pmesh.node[sctr, :]
    for q in range(len(W)):
        pt = Q[q, :]
        wt = W[q]
        N, dNdxi = lagrange_basis_Q4(pt)  # element shape functions
        J0 = dNdxi.T @ pts
        detJ0 = np.linalg.det(J0)
        a = wt * detJ0
        particles.volume[pid] = a
        particles.mass[pid] = a * rho
        particles.positions[pid, :] = N.T @ pts
        particles.deformation_gradient[pid, :] = [1.0, 0.0, 0.0, 1.0]
        if particles.positions[pid, 1] > max_p_height:
            max_p_height = particles.positions[pid, 1]
        pid += 1

particles.set_initial_state()

w = 5e-2 # semi-width of force application region mm
xc = 4.5 # center of force application region mm
for p in range(particles.count): # loop over particles to identify those in the Neumann BC region
    x = particles.positions[p, 0]
    if abs(x - xc) <= w and particles.positions[p, 1] >= max_p_height:
        particles.neumann_particles[p] = True

print("id of particles with Neumann BC applied:", np.where(particles.neumann_particles)[0])


print_section("Particle Initialization")
print(f"Number of particles: {particles.count}")
print(f"Total particle mass: {np.sum(particles.mass):.6f}")
print(f"Total particle volume: {np.sum(particles.volume):.6f}")

#----- Visualization of initial configuration ------
print_section("Visualization of Initial Configuration")
fig, ax = plot_mpm_domain(
    mesh.node,
    mesh.element,
    particles.positions,
    figsize=(12, 8),
    particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
# plot neumann BC particles with a different color
neumann_particles = particles.positions[particles.neumann_particles]   
ax.scatter(neumann_particles[:, 0], neumann_particles[:, 1], s=32, c='green', marker='o', 
           alpha=0.9, label='Neumann BC Particles')
ax.set_title('Initial Particle Positions', fontsize=14, fontweight='bold')
plt.show()


