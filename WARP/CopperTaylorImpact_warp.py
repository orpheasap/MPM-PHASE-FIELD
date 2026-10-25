# %%
import numpy as np
import matplotlib.pyplot as plt
import shutil
import warp as wp

from src.warp_hencky import ElasticMaterial
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.plot_utils import plot_mpm_domain
from src.warp_usl import run_warp_solver
from src.vtk_export import write_pvd

np.set_printoptions(precision=3, suppress=True)
wp.init()
wp.set_device("cuda:0")


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material — OFHC Copper, elastic only -----
rho = 8940.0    # kg/m³
E   = 115e9     # Pa
nu  = 0.31

material = ElasticMaterial(rho, E, nu)

print_section("Material Properties — OFHC Copper (Hencky elastic)")
print(f"rho = {rho} kg/m³,  E = {E:.2e} Pa,  nu = {nu}")
print(f"G = {material.G:.2e} Pa,  K = {material.K:.2e} Pa")
print(f"lambda = {material.lam:.2e} Pa,  mu = {material.mu:.2e} Pa")
print(f"P-wave speed: {material.wave_speed:.2f} m/s")

# ----- Bar geometry & impact velocity -----
D_bar = 0.0076    # diameter/width (m) — 7.6 mm
L_bar = 0.0254    # length/height  (m) — 25.4 mm,  L/D ≈ 3.3
V0    = 190.0     # impact velocity (m/s), downward

# ----- Background grid — cantilever-style layout -----
# Square 1 mm cells. Node row y = dx is the rigid wall, the row below it is a pad ring.
# 36 cells high (not 31): an elastic bar rebounds at ~V0, so over 40 µs its top rises ~5 mm
# above where it started and would leave a 31-cell grid.
dx         = 0.5e-3
numx, numy = 50, 72
Lx, Ly     = numx * dx, numy * dx

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid")
print(f"Domain: {Lx*1e3:.0f} mm × {Ly*1e3:.0f} mm")
print(f"Grid:   {numx} × {numy}  ({mesh.deltax*1e3:.1f} mm cells)")
print(f"Rigid wall: node row y = {dx*1e3:.1f} mm;  pad ring below it")

# ----- Particle mesh -----
noX, noY = 60, 200#15, 50                       # 750 elements × 1 GP = 750 particles
pmesh = Mesh(D_bar, L_bar, noX, noY)
pmesh.node[:, 0] += Lx/2 - D_bar/2      # centre bar horizontally
pmesh.node[:, 1] += dx + .1*1e-3        # bar bottom 0.1 mm above the rigid wall

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

# cpGIMP particle domain: half the particle-mesh element in each direction
lp0 = np.tile([pmesh.deltax / 2, pmesh.deltay / 2], (pCount, 1))

# ----- Identify tracking particles (by initial corner proximity) -----
p0 = particles.initial_positions
pid_topleft = int(np.argmin(np.hypot(p0[:, 0] - p0[:, 0].min(), p0[:, 1] - p0[:, 1].max())))
pid_botleft = int(np.argmin(np.hypot(p0[:, 0] - p0[:, 0].min(), p0[:, 1] - p0[:, 1].min())))

print_section("Particle Initialisation")
print(f"Particles:    {particles.count}")
print(f"Total mass:   {np.sum(particles.mass)*1e3:.3f} g")
print(f"Total volume: {np.sum(particles.volume)*1e6:.3f} mm²  (per unit thickness)")
print(f"Particle domain lp0: {lp0[0, 0]*1e3:.3f} × {lp0[0, 1]*1e3:.3f} mm (half-widths)")

# ----- Visualise initial configuration -----
print_section("Visualisation — Initial Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(6, 10), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title(f'Elastic Copper Taylor Bar — Initial  (V₀ = {V0} m/s)', fontsize=12, fontweight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.show()

# ----- Solver setup -----
dtcrit = mesh.deltax / material.wave_speed
dtime  = 0.5 * dtcrit
nsteps = 340              # ~40 μs — about 3 contact times
time   = nsteps * dtime

c_bar     = np.sqrt(E / (rho * (1 - nu**2)))     # plane-strain bar speed
t_contact = 2 * L_bar / c_bar

print_section("Solver")
print(f"CFL dt:      {dtcrit:.3e} s")
print(f"Using dt:    {dtime:.3e} s  (CFL × 0.5)")
print(f"Total time:  {time*1e6:.1f} μs  ({nsteps} steps)")
print(f"Wave traversal time: {L_bar/material.wave_speed*1e6:.2f} μs")
print(f"Expected contact time 2L/c_bar: {t_contact*1e6:.2f} μs  (c_bar = {c_bar:.0f} m/s)")

shutil.rmtree('vtk_output_warp', ignore_errors=True)

# %%
solver_results = run_warp_solver(
    mesh=mesh,
    particles=particles,
    material=material,
    dtime=dtime,
    time=time,
    alpha=0.99,
    vtk_output_dir='vtk_output_warp',
    vtk_interval=10,
    track_pids=[pid_topleft, pid_botleft],
    lp0=lp0,
)

print_section("Solver Complete")
pvd_path = write_pvd('vtk_output_warp', solver_results['vtk_entries'])
print(f"VTK steps saved: {len(solver_results['vtk_entries'])}  →  {pvd_path}")
print(f"Time loop: {solver_results['wall_time']:.3f} s for {solver_results['nsteps']} steps "
      f"({solver_results['wall_time']/solver_results['nsteps']*1e3:.2f} ms/step, incl. output)")

# ----- Save particle displacement and energy histories -----
track_times = solver_results['track_times']
track_pos   = solver_results['track_positions']

tl_disp_y = track_pos[pid_topleft][:, 1] - p0[pid_topleft, 1]
bl_disp_x = track_pos[pid_botleft][:, 0] - p0[pid_botleft, 0]
bl_disp_y = track_pos[pid_botleft][:, 1] - p0[pid_botleft, 1]

np.savetxt('particle_history_warp.csv',
           np.column_stack([track_times * 1e6, tl_disp_y * 1e3, bl_disp_x * 1e3, bl_disp_y * 1e3]),
           header='time_us,tl_disp_y_mm,bl_disp_x_mm,bl_disp_y_mm',
           delimiter=',', comments='')
print(f"Particle history saved → particle_history_warp.csv  ({len(track_times)} snapshots)")
print(f"  top-left  pid={pid_topleft}  x0={p0[pid_topleft, 0]*1e3:.3f} mm  y0={p0[pid_topleft, 1]*1e3:.3f} mm")
print(f"  bot-left  pid={pid_botleft}  x0={p0[pid_botleft, 0]*1e3:.3f} mm  y0={p0[pid_botleft, 1]*1e3:.3f} mm")

e_times = solver_results['energy_times']
E_kin, E_el, E_tot = solver_results['kinetic'], solver_results['elastic'], solver_results['total']
np.savetxt('energy_history_warp.csv',
           np.column_stack([e_times * 1e6, E_kin, E_el, E_tot]),
           header='time_us,kinetic_J_per_m,elastic_J_per_m,total_J_per_m',
           delimiter=',', comments='')
print(f"Energy history saved → energy_history_warp.csv  ({len(e_times)} snapshots)")

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
print(f"Final width:       {D_tip*1e3:.2f} mm  (change: {-(1 - D_tip/D_bar)*100:.1f}%)")
print(f"Final J range:     [{solver_results['J'].min():.4f}, {solver_results['J'].max():.4f}]")
print(f"Final bar top:     {y_final.max()*1e3:.2f} mm  (domain clamp at {(Ly - dx)*1e3:.0f} mm)")

# ----- Checks -----
print_section("Checks")

# 1. Energy: |E_total(end) - E_total(0)| <= 5 % of the initial kinetic energy
energy_err = abs(E_tot[-1] - E_tot[0]) / E_kin[0]
energy_ok  = energy_err <= 0.05
print(f"[{'PASS' if energy_ok else 'FAIL'}] Energy: E_total(0) = {E_tot[0]:.3f} J/m, "
      f"E_total(end) = {E_tot[-1]:.3f} J/m  (kinetic {E_kin[-1]:.3f}, elastic {E_el[-1]:.3f}); "
      f"drift = {energy_err*100:.2f} % of E_kin(0)  (limit 5 %)")

# 2. Contact time from the wall reaction, accumulated on the device per step
R      = solver_results['wall_reaction']
R_peak = R.max()
in_contact    = np.nonzero(R > 0.01 * R_peak)[0]
t_contact_mpm = (in_contact[-1] - in_contact[0] + 1) * dtime
contact_err   = abs(t_contact_mpm - t_contact) / t_contact
contact_ok    = contact_err <= 0.15
contact_ended = in_contact[-1] + 1 < len(R)
print(f"[{'PASS' if contact_ok else 'FAIL'}] Contact time: {t_contact_mpm*1e6:.2f} μs "
      f"(R > 1 % of peak {R_peak:.3e} N/m, steps {in_contact[0]}–{in_contact[-1]}) vs 2L/c_bar = "
      f"{t_contact*1e6:.2f} μs; error = {contact_err*100:.1f} %  (limit 15 %)")
print(f"       wall reaction nonzero over {np.count_nonzero(R)*dtime*1e6:.2f} μs; "
      f"contact ended before end of run: {contact_ended}")

# 3. Rebound: mean particle v_y after contact >= 0.8 V0 (upward)
rebound_ok = contact_ended and rebound_velocity >= 0.8 * V0
print(f"[{'PASS' if rebound_ok else 'FAIL'}] Rebound: mean v_y = {rebound_velocity:.2f} m/s "
      f"= {rebound_velocity/V0:.3f} V0  (limit ≥ 0.8 V0 = {0.8*V0:.0f} m/s)")

all_ok = energy_ok and contact_ok and rebound_ok
print(f"\nAll checks {'PASS' if all_ok else 'did NOT pass'}.")

# %%
print_section("Visualisation — Final Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(6, 10), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title('Elastic Copper Taylor Bar — Final Configuration', fontsize=12, fontweight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.show()
