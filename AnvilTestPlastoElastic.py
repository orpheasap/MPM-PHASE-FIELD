# %%
# =====================================================================
#  Copper anvil (Taylor) impact — 3D MPM, ELASTO-PLASTIC material
# =====================================================================
#  Full MUSL-EP formulation from README.md (steps 22-34), no damage:
#    - Johnson-Cook flow stress + radial-return plasticity
#    - Mie-Grüneisen equation of state for the pressure
#    - Polar decomposition (SVD) for objective stress integration
#  Background grid:  structured 8-node hexahedra (B8)
#  Particle init:    B8 element + Gauss quadrature loop (cylinder carve)
# ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import shutil

from src.material import Material
from src.mesh3D import Mesh3D
from src.particle import ParticleSet3D
from src.lagrange_basis import lagrange_basis_B8
from src.quadrature import gauss_3D
from src.solver3d_ep import run_mpm_solver_3d_ep, build_particle_element_map
from src.solver3d import NodeState3D
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

print_section("Material Properties — OFHC Copper (elasto-plastic)")
print(f"rho = {rho} kg/m³,  E = {E:.2e} Pa,  nu = {nu}")
print(f"G = {material.G:.2e} Pa,  K = {material.K:.2e} Pa")
print(f"JC:  A={A:.2e}, B={B:.2e}, n={n}, C={C}")
print(f"MG:  c0={c0} m/s,  S_alpha={S_alpha},  Gamma0={Gamma0}")
print(f"P-wave speed: {material.wave_speed:.2f} m/s")

# ----- Bar geometry & impact velocity -----
D_bar = 0.0076    # diameter (m) — 7.6 mm
L_bar = 0.0254    # length   (m) — 25.4 mm,  L/D ≈ 3.3
V0    = 190.0     # impact velocity (m/s), downward along -y

# ----- Computational grid (coarse, runnable demo) -----
Lx, Ly, Lz       = 0.012, 0.030, 0.012      # 12 × 30 × 12 mm box
numx, numy, numz = 8, 16, 8                  # ~1.5 mm cells

mesh = Mesh3D(Lx, Ly, Lz, numx, numy, numz)
print_section("Computational Grid (3D)")
print(f"Domain: {Lx*1e3:.0f} × {Ly*1e3:.0f} × {Lz*1e3:.0f} mm")
print(f"Grid:   {numx}×{numy}×{numz}  "
      f"({mesh.deltax*1e3:.2f}×{mesh.deltay*1e3:.2f}×{mesh.deltaz*1e3:.2f} mm cells)")
print(f"Nodes: {mesh.nodeCount},  Elements: {mesh.elemCount}")

# ----- Particle mesh (hexahedral block carved into a cylinder) -----
noX, noY, noZ = 8, 20, 8
pmesh = Mesh3D(D_bar, L_bar, D_bar, noX, noY, noZ)

print_section("Bar Geometry")
print(f"D = {D_bar*1e3:.1f} mm,  L = {L_bar*1e3:.1f} mm  (L/D = {L_bar/D_bar:.2f})")
print(f"Particle mesh: {noX}×{noY}×{noZ} elements")

# %%
# ----- Particle initialisation (B8 + quadrature loop, cylinder carve) -----
ngp  = 1
W, Q = gauss_3D(ngp)

radius    = D_bar / 2.0
center_xz = np.array([D_bar / 2.0, D_bar / 2.0])

volume_l, mass_l, coord_l = [], [], []
for e in range(pmesh.elemCount):
    sctr = pmesh.element[e]
    pts  = pmesh.node[sctr]                        # (8, 3)
    for q in range(len(W)):
        N, dNdxi = lagrange_basis_B8(Q[q])         # N:(8,) dNdxi:(8,3)
        J0 = pts.T @ dNdxi                          # (3, 3) Jacobian
        x  = N @ pts                                # (3,)  physical coord
        r  = np.linalg.norm(x[[0, 2]] - center_xz)  # radial dist in x-z plane
        if r - radius < 0:                          # inside the cylinder
            detJ = np.linalg.det(J0)
            volume_l.append(W[q] * detJ)
            mass_l.append(W[q] * detJ * rho)
            coord_l.append(x)

pCount    = len(coord_l)
particles = ParticleSet3D(pCount)
particles.volume[:]    = np.array(volume_l)
particles.mass[:]      = np.array(mass_l)
particles.positions[:] = np.array(coord_l)

# Centre the bar in the box (x, z) and lift it slightly above the floor (y=0)
particles.positions[:, 0] += Lx / 2.0 - D_bar / 2.0
particles.positions[:, 2] += Lz / 2.0 - D_bar / 2.0
particles.positions[:, 1] += 0.1e-3

# Initial state (README step 3-4): stress-free, moving downward,
# eps_p^0 = 0, D^0 = 0, sigma_dev^0 = 0, e^0 = 0 (reference temperature).
particles.velocities[:] = np.array([0.0, -V0, 0.0])
particles.set_initial_state()

# ----- Tracking particles -----
p0 = particles.initial_positions
axis_xz = np.hypot(p0[:, 0] - p0[:, 0].mean(), p0[:, 2] - p0[:, 2].mean())
pid_tip = int(np.lexsort((axis_xz, p0[:, 1]))[0])     # lowest, near-axis (impact tip)
pid_top = int(np.lexsort((axis_xz, -p0[:, 1]))[0])    # highest, near-axis (free end)

print_section("Particle Initialisation")
print(f"Candidate points: {pmesh.elemCount * len(W)}")
print(f"Particles kept:   {particles.count}  (cylinder carve)")
print(f"Total mass:       {np.sum(particles.mass)*1e3:.3f} g")
print(f"Analytic mass:    {np.pi*radius**2*L_bar*rho*1e3:.3f} g  (rho·πr²L)")

# %%
# ----- Visualise initial configuration (3D scatter) -----
def scatter3d(pos, scalar, title, fname, clabel):
    fig = plt.figure(figsize=(6, 8))
    ax  = fig.add_subplot(111, projection='3d')
    p = ax.scatter(pos[:, 0]*1e3, pos[:, 2]*1e3, pos[:, 1]*1e3,
                   s=6, c=scalar, cmap='inferno', alpha=0.8)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (mm)'); ax.set_zlabel('y (mm)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    fig.colorbar(p, ax=ax, shrink=0.6, label=clabel)
    try:
        ax.set_box_aspect((Lx, Lz, Ly))
    except Exception:
        pass
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close(fig)
    return fname

print_section("Visualisation — Initial Configuration")
f_init = scatter3d(particles.positions, particles.positions[:, 1],
                   f'Anvil Test — Initial (V₀={V0} m/s)', 'anvil_initial_3d.png', 'y (m)')
print(f"Saved {f_init}")

# ----- Element-particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)
print_section("Element-Particle Mapping")
print(f"Particles located:       {sum(len(m) for m in particles.mpoints)}")
print(f"Elements with particles: {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")

# ----- Solver setup -----
node_state = NodeState3D(mesh.nodeCount)

dtcrit = min(mesh.deltax, mesh.deltay, mesh.deltaz) / material.wave_speed
dtime  = 0.4 * dtcrit
nsteps = 1000
time   = nsteps * dtime

print_section("Solver")
print(f"CFL dt:     {dtcrit:.3e} s")
print(f"Using dt:   {dtime:.3e} s  (CFL × 0.4)")
print(f"Total time: {time*1e6:.2f} μs  ({nsteps} steps)")

shutil.rmtree('vtk_output_ep', ignore_errors=True)

# %%
solver_results = run_mpm_solver_3d_ep(
    mesh=mesh, particles=particles, material=material,
    g=0, dtime=dtime, time=time, alpha=0.99, g_dir=1,
    node_state=node_state,
    vtk_output_dir='vtk_output_ep', vtk_interval=10,
    track_pids=[pid_tip, pid_top],
)

print_section("Solver Complete")
pvd_path = write_pvd('vtk_output_ep', solver_results['vtk_entries'])
print(f"VTK steps saved: {len(solver_results['vtk_entries'])}  →  {pvd_path}")

# ----- Post-processing -----
print_section("Benchmark Metrics")
x_final, y_final, z_final = (particles.positions[:, 0],
                             particles.positions[:, 1],
                             particles.positions[:, 2])
L_final = y_final.max() - y_final.min()
D_tipx  = x_final.max() - x_final.min()
D_tipz  = z_final.max() - z_final.min()
print(f"Initial length:    {L_bar*1e3:.2f} mm")
print(f"Final length:      {L_final*1e3:.2f} mm  (axial compression: {(1 - L_final/L_bar)*100:.1f}%)")
print(f"Final width  (x):  {D_tipx*1e3:.2f} mm  (mushrooming: {(D_tipx/D_bar - 1)*100:.1f}%)")
print(f"Final width  (z):  {D_tipz*1e3:.2f} mm")
print(f"Max equiv. plastic strain: {particles.eps_p.max():.3f}")
print(f"Mean equiv. plastic strain: {particles.eps_p.mean():.3f}")
print(f"Max von-Mises-eq dev stress: {np.max([np.sqrt(1.5*(s[0]**2+s[1]**2+s[2]**2+2*(s[3]**2+s[4]**2+s[5]**2))) for s in particles.stress_dev])/1e6:.1f} MPa")
print(f"Mean v_y (rebound): {particles.velocities[:, 1].mean():.2f} m/s")

# %%
print_section("Visualisation — Final Configuration")
f_final = scatter3d(particles.positions, particles.eps_p,
                    'Anvil Test — Final (colour = equiv. plastic strain)',
                    'anvil_final_3d.png', 'eps_p')
print(f"Saved {f_final}")
