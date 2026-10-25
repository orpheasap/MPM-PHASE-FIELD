# %%
import numpy as np
import matplotlib.pyplot as plt
import shutil

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.plot_utils import plot_mpm_domain
from src.solver1 import run_mpm_solver, NodeState, build_particle_element_map
from src.vtk_export import write_pvd

np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material — OFHC Copper (Table 7.1) -----
rho       = 8940.0    # kg/m³
E         = 115e9     # Pa
nu        = 0.31

# Johnson-Cook flow stress
A         = 65e6      # Pa
B         = 356e6     # Pa
n         = 0.37
C         = 0.013
eps_dot_0 = 1.0       # reference strain rate (1/s)

# Mie-Grüneisen EOS (Table 7.1)
c0      = 3933.0      # bulk sound speed  (m/s)
S_alpha = 1.5         # Hugoniot slope
Gamma0  = 0.0         # Grüneisen Gamma — thermal pressure term vanishes
chi     = 0.9         # Taylor-Quinney coefficient

material = Material(rho, E, nu, A, B, n, C, eps_dot_0,
                    c0=c0, Gamma0=Gamma0, S_alpha=S_alpha, chi=chi)

print_section("Material Properties — OFHC Copper")
print(f"rho = {rho} kg/m³,  E = {E:.2e} Pa,  nu = {nu}")
print(f"G = {material.G:.2e} Pa,  K = {material.K:.2e} Pa")
print(f"JC:  A={A:.2e}, B={B:.2e}, n={n}, C={C}")
print(f"MG:  c0={c0} m/s,  S_alpha={S_alpha},  Gamma0={Gamma0}")
print(f"P-wave speed: {material.wave_speed:.2f} m/s")

# ----- Bar geometry & impact velocity -----
D_bar = 0.0076    # diameter/width (m) — 7.6 mm
L_bar = 0.0254    # length/height  (m) — 25.4 mm,  L/D ≈ 3.3
V0    = 190.0     # impact velocity (m/s), downward

# ----- Computational grid -----
Lx, Ly   = 0.025, 0.030    # 25 mm × 30 mm — room for mushrooming and free tip
numx, numy = 25,30#50, 60         # .5 mm cells

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Computational Grid")
print(f"Domain: {Lx*1e3:.0f} mm × {Ly*1e3:.0f} mm")
print(f"Grid:   {numx} × {numy}  ({mesh.deltax*1e3:.1f} mm cells)")

# ----- Particle mesh -----
# Bar bottom at y=0 — rigid wall contact begins at t=0
noX, noY = 15, 50                       # 10×40 elements × 16 GP = 6400 particles, ~5/grid cell
pmesh = Mesh(D_bar, L_bar, noX, noY)
pmesh.node[:, 0] += Lx/2 - D_bar/2     # centre bar horizontally
pmesh.node[:, 1] += .1*1e-3   # lift bar slightly above y=0 to avoid initial penetration with rigid wall

print_section("Bar Geometry")
print(f"D = {D_bar*1e3:.1f} mm,  L = {L_bar*1e3:.1f} mm  (L/D = {L_bar/D_bar:.2f})")
print(f"Particle mesh: {noX} x {noY} elements")

# ----- Particle initialisation -----
ngp    = 1
W, Q   = gauss_2D(ngp)
pCount = len(pmesh.element) * len(W)

particles = ParticleSet(pCount)

pid = 0
for e in range(len(pmesh.element)):
    sctr = pmesh.element[e, :]
    pts  = pmesh.node[sctr, :]
    for q in range(len(W)):
        N, dNdxi     = lagrange_basis_Q4(Q[q])
        J0           = dNdxi.T @ pts
        a            = W[q] * np.linalg.det(J0)
        particles.volume[pid]               = a
        particles.mass[pid]                 = a * rho
        particles.positions[pid]            = N.T @ pts
        particles.deformation_gradient[pid] = [1.0, 0.0, 0.0, 1.0]
        particles.velocities[pid]           = [0.0, -V0]
        pid += 1

particles.set_initial_state()
# No BC particles: rigid wall = bNodes (y=0), zeroed automatically every step.

# ----- Identify tracking particles (by initial corner proximity) -----
p0 = particles.initial_positions
pid_topleft = int(np.argmin(np.hypot(p0[:, 0] - p0[:, 0].min(), p0[:, 1] - p0[:, 1].max())))
pid_botleft = int(np.argmin(np.hypot(p0[:, 0] - p0[:, 0].min(), p0[:, 1] - p0[:, 1].min())))

print_section("Particle Initialisation")
print(f"Particles:    {particles.count}")
print(f"Total mass:   {np.sum(particles.mass)*1e3:.3f} g")
print(f"Total volume: {np.sum(particles.volume)*1e6:.3f} cm²")

# ----- Visualise initial configuration -----
print_section("Visualisation — Initial Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(6, 10), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title(f'OFHC Copper Taylor Bar — Initial  (V₀ = {V0} m/s)', fontsize=12, fontweight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.show()

# ----- Element-particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)

print_section("Element-Particle Mapping")
print(f"Particles located:        {sum(len(m) for m in particles.mpoints)}")
print(f"Elements with particles:  {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")

# ----- Solver setup -----
node_state = NodeState(mesh.nodeCount)

dtcrit = mesh.deltax / material.wave_speed
dtime  = 0.5 * dtcrit
nsteps = 1000             # ~120 μs — well past full mushrooming
time   = nsteps * dtime

print_section("Solver")
print(f"CFL dt:      {dtcrit:.3e} s")
print(f"Using dt:    {dtime:.3e} s  (CFL × 0.5)")
print(f"Total time:  {time*1e6:.1f} μs  ({nsteps} steps)")
print(f"Wave traversal time: {L_bar/material.wave_speed*1e6:.2f} μs")

shutil.rmtree('vtk_output', ignore_errors=True)

# %%
solver_results = run_mpm_solver(
    mesh=mesh,
    particles=particles,
    material=material,
    g=0,
    dtime=dtime,
    time=time,
    alpha=0.99,
    node_state=node_state,
    vtk_output_dir='vtk_output',
    vtk_interval=10,
    track_pids=[pid_topleft, pid_botleft],
)

print_section("Solver Complete")
pvd_path = write_pvd('vtk_output', solver_results['vtk_entries'])
print(f"VTK steps saved: {len(solver_results['vtk_entries'])}  →  {pvd_path}")

# ----- Save particle displacement histories -----
track_times = solver_results['track_times']
track_pos   = solver_results['track_positions']

tl_disp_y = track_pos[pid_topleft][:, 1] - p0[pid_topleft, 1]
bl_disp_x = track_pos[pid_botleft][:, 0] - p0[pid_botleft, 0]
bl_disp_y = track_pos[pid_botleft][:, 1] - p0[pid_botleft, 1]

np.savetxt('particle_history.csv',
           np.column_stack([track_times * 1e6, tl_disp_y * 1e3, bl_disp_x * 1e3, bl_disp_y * 1e3]),
           header='time_us,tl_disp_y_mm,bl_disp_x_mm,bl_disp_y_mm',
           delimiter=',', comments='')
print(f"Particle history saved → particle_history.csv  ({len(track_times)} snapshots)")
print(f"  top-left  pid={pid_topleft}  x0={p0[pid_topleft, 0]*1e3:.3f} mm  y0={p0[pid_topleft, 1]*1e3:.3f} mm")
print(f"  bot-left  pid={pid_botleft}  x0={p0[pid_botleft, 0]*1e3:.3f} mm  y0={p0[pid_botleft, 1]*1e3:.3f} mm")

rebound_velocity = particles.velocities[:, 1].mean()
print(f"Mean rebound velocity: {rebound_velocity:.2f} m/s")

# ----- Post-processing: benchmark metrics -----
print_section("Benchmark Metrics")
y_final = particles.positions[:, 1]
x_final = particles.positions[:, 0]

L_final = y_final.max() - y_final.min()
D_tip   = x_final.max() - x_final.min()

print(f"Initial length:    {L_bar*1e3:.2f} mm")
print(f"Final length:      {L_final*1e3:.2f} mm  (compression: {(1 - L_final/L_bar)*100:.1f}%)")
print(f"Mushroom diameter: {D_tip*1e3:.2f} mm  (mushrooming: {-(1 - D_tip/D_bar)*100:.1f}%)")

# %%
print_section("Visualisation — Final Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(6, 10), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title('OFHC Copper Taylor Bar — Final Deformed Shape', fontsize=12, fontweight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.show()
