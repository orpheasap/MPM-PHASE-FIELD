# %%
# =====================================================================
#  Bone tensile fracture — 3D MPM, ELASTO-PLASTIC + DAMAGE
# =====================================================================
#  Pipeline
#    1. particles_from_stl.py carves the bone interior into particles and
#       writes their positions (mm) to bone_particles_mm.csv. The FIRST line of
#       that file is "# resolution_mm=<spacing>" so it is trivial to read here.
#    2. This script:
#         - reads those positions (mm -> m) and the grid resolution,
#         - builds a background hex grid big enough for the bone plus head-room
#           for the upward pull,
#         - initialises a ParticleSet3D and a brittle-bone material (D1..D5),
#         - fixes the bottom 10 % of the bone and pulls the top 10 % up at a
#           constant velocity (Dirichlet BCs, defined *here* because they are
#           problem specific),
#         - runs the MUSL elasto-plastic + damage time loop inline, using the
#           problem-independent routines in src/solver_utilities.py.
#
#  The constitutive formulation is the full README.md scheme (steps 22-34) with
#  the damage update (step 33) and D coupled into G, sigma_f and the EOS.
#
#  Output: vtk_output_bone/simulation.pvd (open in ParaView; colour by
#  "damage", "eps_p" or "von_mises") plus initial/final 3D scatter PNGs.
# ---------------------------------------------------------------------
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401  (registers 3d projection)

from src.material import Material
from src.mesh3D import Mesh3D
from src.particle import ParticleSet3D
from src.solver_utilities import (
    NodeState3D, get_mpm3d_shape, build_particle_element_map,
    write_particles_vtp_3d, write_pvd, _constitutive_update, _von_mises_full,
)

np.set_printoptions(precision=4, suppress=True)


def print_section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
PARTICLE_FILE = "bone_particles_mm.csv"     # produced by particles_from_stl.py
VTK_DIR       = "vtk_output_bone"

# Loading: pull the top up along the long (y) axis, fix the bottom.
PULL_AXIS     = 1                            # 0=x, 1=y, 2=z  (bone long axis = y)
GRIP_FRACTION = 0.10                         # top / bottom 10 % of the height
V_TOP         = float(os.environ.get("BONE_VTOP", 2.0))   # pull speed [m/s]

# Time stepping (env-overridable so a quick smoke test is easy)
N_STEPS       = int(os.environ.get("BONE_NSTEPS", 4000))
CFL           = 0.30
ALPHA         = 0.99                         # FLIP/PIC blend
VTK_INTERVAL  = 25
EOS           = "linear"                     # 'linear' (recommended for bone) or 'mg'
TSTAR         = 0.0                          # homologous temperature (room temp)

# ─────────────────────────────────────────────────────────────────────
# 1. MATERIAL — brittle cortical bone
# ─────────────────────────────────────────────────────────────────────
# Cortical bone is stiff, strong and fails at a very small strain (~0.5-0.8 %)
# with little plastic flow — i.e. brittle. The Johnson-Cook constants below are
# chosen to reproduce that behaviour rather than taken from a metals handbook.
rho       = 1900.0     # kg/m^3   (cortical bone ~1.8-2.0 g/cm^3)
E         = 12.0e9     # Pa       (cortical Young's modulus ~12-18 GPa)
nu        = 0.30

# Johnson-Cook flow stress  sigma_f = [A + B eps_p^n][1 + C ln(eps_dot*)]...
A         = 110.0e6    # Pa  initial yield ~ tensile strength of bone
B         = 80.0e6     # Pa  modest hardening
n         = 0.5
C         = 0.02       # mild strain-rate hardening (bone is rate sensitive)
eps_dot_0 = 1.0        # 1/s reference strain rate

# Johnson-Cook damage  eps_f = [D1 + D2 exp(D3 sigma*)][1 + D4 ln(eps_dot*)][1 + D5 T*]
# Note README sign convention: sigma* = -p_hat/sigma_eq, so a TENSILE state has
# sigma* < 0. With D3 > 0 the exponential then SHRINKS the failure strain under
# tension -> small, brittle tensile failure strain (~0.5-0.6 %).
D1        = 0.002      # baseline failure strain
D2        = 0.010      # triaxiality amplitude
D3        = 3.0        # triaxiality sensitivity (positive, see note above)
D4        = 0.02       # rate dependence of ductility
D5        = 0.0        # thermal term (inactive at body/room temperature, T*=0)

material = Material(
    rho, E, nu, A, B, n, C, eps_dot_0,
    D1=D1, D2=D2, D3=D3, D4=D4, D5=D5,
    # Mie-Grüneisen parameters (only used if EOS='mg'); harmless otherwise.
    c0=np.sqrt(E / rho), Gamma0=0.0, S_alpha=1.0, chi=0.9,
)

print_section("Material — brittle cortical bone")
print(f"rho = {rho} kg/m^3,  E = {E:.2e} Pa,  nu = {nu}")
print(f"G = {material.G:.3e} Pa,  K = {material.K:.3e} Pa")
print(f"JC flow:  A={A:.2e}  B={B:.2e}  n={n}  C={C}")
print(f"JC damage: D1={D1}  D2={D2}  D3={D3}  D4={D4}  D5={D5}  "
      f"(enabled={material.damage_enabled})")
print(f"P-wave speed: {material.wave_speed:.1f} m/s")
print(f"Yield strain ~ A/E = {A / E * 100:.3f} %")

# ─────────────────────────────────────────────────────────────────────
# 2. LOAD CARVED PARTICLES  (mm -> m); resolution is on the first line
# ─────────────────────────────────────────────────────────────────────
with open(PARTICLE_FILE) as fh:
    first = fh.readline()                    # "# resolution_mm=5"
spacing_mm = float(first.split("=")[1])
spacing    = spacing_mm * 1e-3               # m

pos_mm    = np.loadtxt(PARTICLE_FILE, delimiter=",", comments="#")
positions = pos_mm * 1e-3                     # mm -> m
pCount    = len(positions)
vol_p     = spacing ** 3                      # per-particle reference volume [m^3]

print_section("Particles loaded from STL carve")
print(f"File: {PARTICLE_FILE}")
print(f"Particles:    {pCount:,}")
print(f"Spacing:      {spacing_mm:.2f} mm  -> volume/particle = {vol_p:.3e} m^3")
print(f"Bone extent:  {(positions.max(0) - positions.min(0)) * 1e3} mm  (x, y, z)")

# ─────────────────────────────────────────────────────────────────────
# 3. BACKGROUND GRID — big enough for the bone + upward head-room
# ─────────────────────────────────────────────────────────────────────
h      = 1.2 * spacing                        # grid cell size (slightly > spacing)
pmin   = positions.min(0)
pmax   = positions.max(0)
extent = pmax - pmin

pad        = 3.0 * h                          # margin on every side
sim_time   = N_STEPS * CFL * h / material.wave_speed
head_room  = abs(V_TOP) * sim_time + 4.0 * h  # room above the top grip for the pull

L = extent + 2.0 * pad
L[PULL_AXIS] += head_room

numx = int(np.ceil(L[0] / h))
numy = int(np.ceil(L[1] / h))
numz = int(np.ceil(L[2] / h))
Lx, Ly, Lz = numx * h, numy * h, numz * h

mesh = Mesh3D(Lx, Ly, Lz, numx, numy, numz)

offset    = pad - pmin                         # pad on the low side of every axis
positions = positions + offset

print_section("Background grid (3D hex)")
print(f"Cell size:  {h * 1e3:.2f} mm")
print(f"Domain:     {Lx * 1e3:.1f} x {Ly * 1e3:.1f} x {Lz * 1e3:.1f} mm")
print(f"Grid:       {numx} x {numy} x {numz}  "
      f"(nodes={mesh.nodeCount:,}, elements={mesh.elemCount:,})")

# ─────────────────────────────────────────────────────────────────────
# 4. PARTICLE SET
# ─────────────────────────────────────────────────────────────────────
particles = ParticleSet3D(pCount)
particles.positions[:]  = positions
particles.volume[:]     = vol_p
particles.mass[:]       = vol_p * rho
particles.velocities[:] = 0.0
particles.set_initial_state()

# ─────────────────────────────────────────────────────────────────────
# 5. BOUNDARY CONDITIONS  (problem specific — defined in the main script)
#    Fix the bottom 10 %, pull the top 10 % upward at a constant velocity.
# ─────────────────────────────────────────────────────────────────────
yp     = particles.positions[:, PULL_AXIS]
y0, y1 = yp.min(), yp.max()
span   = y1 - y0
bottom_cut = y0 + GRIP_FRACTION * span
top_cut    = y1 - GRIP_FRACTION * span

fixed_mask  = yp <= bottom_cut                # Dirichlet: v = 0, frozen
driven_mask = yp >= top_cut                   # Dirichlet: v = v_drv (constant)

v_drv = np.zeros(3)
v_drv[PULL_AXIS] = V_TOP

# Enforce the prescribed grip velocities from the very first P2G mapping.
particles.velocities[fixed_mask]  = 0.0
particles.velocities[driven_mask] = v_drv

print_section("Boundary conditions")
print(f"Pull axis:        {'xyz'[PULL_AXIS]}  (V_top = {V_TOP} m/s)")
print(f"Fixed  (bottom):  {fixed_mask.sum():,} particles  (y <= {bottom_cut * 1e3:.1f} mm)")
print(f"Driven (top):     {driven_mask.sum():,} particles  (y >= {top_cut * 1e3:.1f} mm)")
print(f"Free   (gauge):   {pCount - fixed_mask.sum() - driven_mask.sum():,} particles")

# ─────────────────────────────────────────────────────────────────────
# 6. TIME STEP + INITIAL SNAPSHOT
# ─────────────────────────────────────────────────────────────────────
dtcrit = h / material.wave_speed
dtime  = CFL * dtcrit
time   = N_STEPS * dtime

print_section("Solver setup")
print(f"CFL dt:     {dtcrit:.3e} s")
print(f"Using dt:   {dtime:.3e} s  (CFL x {CFL})")
print(f"Steps:      {N_STEPS}   ->  total time {time * 1e3:.3f} ms")
print(f"EOS:        {EOS}")


def scatter3d(pos, scalar, title, fname, clabel, mask_colors=None):
    fig = plt.figure(figsize=(5, 9))
    ax  = fig.add_subplot(111, projection="3d")
    if mask_colors is not None:
        ax.scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, pos[:, 1] * 1e3,
                   s=5, c=mask_colors, alpha=0.8)
    else:
        p = ax.scatter(pos[:, 0] * 1e3, pos[:, 2] * 1e3, pos[:, 1] * 1e3,
                       s=5, c=scalar, cmap="inferno", alpha=0.85)
        fig.colorbar(p, ax=ax, shrink=0.5, label=clabel)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)"); ax.set_zlabel("y (mm)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    try:
        ax.set_box_aspect((Lx, Lz, Ly))
    except Exception:
        pass
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close(fig)
    return fname


grip_color = np.where(fixed_mask, "tab:blue",
                      np.where(driven_mask, "tab:red", "lightgray"))
scatter3d(particles.positions, None,
          "Bone — initial (red=pulled, blue=fixed)", "bone_initial_3d.png",
          "", mask_colors=grip_color)

pid_top   = int(np.argmax(yp))
pid_gauge = int(np.argmin(np.abs(yp - 0.5 * (y0 + y1))))

# ─────────────────────────────────────────────────────────────────────
# 7. MUSL ELASTO-PLASTIC + DAMAGE TIME LOOP  (README steps 6-37)
#    The loop lives here in the driver; the constitutive update and grid
#    helpers come from src/solver_utilities.py.
# ─────────────────────────────────────────────────────────────────────
shutil.rmtree(VTK_DIR, ignore_errors=True)

node_state = NodeState3D(mesh.nodeCount)
_, mpoints = build_particle_element_map(particles.positions, mesh)
dx, dy, dz = mesh.deltax, mesh.deltay, mesh.deltaz

vtk_entries = []
t = 0.0

print_section("Running")
for istep in range(N_STEPS):

    # ── 7: reset grid ──────────────────────────────────────────────────
    node_state.reset()

    # ── 8-12: particle-to-grid (P2G): mass, momentum, internal force ────
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            sigma = particles.stress[pid]
            mp    = particles.mass[pid]
            Vp    = particles.volume[pid]
            vp    = particles.velocities[pid]
            for idn in mesh.element[e]:
                x = particles.positions[pid] - mesh.node[idn]
                N, dNdx = get_mpm3d_shape(x, dx, dy, dz)
                if N == 0.0:
                    continue
                node_state.mass[idn]     += N * mp
                node_state.momentum[idn] += N * mp * vp
                node_state.internal_force[idn, 0] -= Vp * (
                    sigma[0]*dNdx[0] + sigma[3]*dNdx[1] + sigma[5]*dNdx[2])
                node_state.internal_force[idn, 1] -= Vp * (
                    sigma[3]*dNdx[0] + sigma[1]*dNdx[1] + sigma[4]*dNdx[2])
                node_state.internal_force[idn, 2] -= Vp * (
                    sigma[5]*dNdx[0] + sigma[4]*dNdx[1] + sigma[2]*dNdx[2])
            # (no body force / Neumann traction in this problem -> f_ext = 0)

    # ── 14: update nodal momenta ───────────────────────────────────────
    f_total  = node_state.internal_force + node_state.external_force
    mv_tilde = node_state.momentum + f_total * dtime

    # ── 16-18: MUSL double mapping — move particles, blend velocities ───
    m_safe  = np.where(node_state.mass > 0, node_state.mass, 1.0)
    v_old   = node_state.momentum / m_safe[:, None]
    v_tilde = mv_tilde            / m_safe[:, None]

    positions_old = particles.positions.copy()

    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            node_data = [
                (idn, *get_mpm3d_shape(positions_old[pid] - mesh.node[idn], dx, dy, dz))
                for idn in mesh.element[e]
            ]
            for idn, N, _ in node_data:
                if N != 0.0:
                    particles.positions[pid] += dtime * N * v_tilde[idn]
            particles.velocities[pid] *= ALPHA
            for idn, N, _ in node_data:
                if N != 0.0:
                    particles.velocities[pid] += N * (v_tilde[idn] - ALPHA * v_old[idn])

    # ── DIRICHLET BCs (problem specific): grips override the mapped motion
    #    Fixed bottom  -> velocity 0, position frozen.
    #    Driven top    -> constant velocity, position advanced at that velocity.
    if np.any(fixed_mask):
        particles.positions[fixed_mask]  = particles.initial_positions[fixed_mask]
        particles.velocities[fixed_mask] = 0.0
    if np.any(driven_mask):
        particles.positions[driven_mask]  = positions_old[driven_mask] + dtime * v_drv
        particles.velocities[driven_mask] = v_drv

    # ── 19-20: re-map the grip-corrected momentum back to the grid ─────
    mv_new = np.zeros_like(node_state.momentum)
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            for idn in mesh.element[e]:
                x = positions_old[pid] - mesh.node[idn]
                N, _ = get_mpm3d_shape(x, dx, dy, dz)
                if N != 0.0:
                    mv_new[idn] += N * particles.mass[pid] * particles.velocities[pid]

    # ── 22-33: grid-to-particle (G2P) velocity gradient + constitutive ─
    v_new = mv_new / m_safe[:, None]
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            Lp = np.zeros((3, 3))
            for idn in mesh.element[e]:
                x = positions_old[pid] - mesh.node[idn]
                _, dNdx = get_mpm3d_shape(x, dx, dy, dz)
                Lp += np.outer(v_new[idn], dNdx)
            _constitutive_update(material, pid, particles, dtime, Lp,
                                 eos=EOS, Tstar=TSTAR)

    # ── output ─────────────────────────────────────────────────────────
    if istep % VTK_INTERVAL == 0:
        fname = write_particles_vtp_3d(
            particles.positions, particles.velocities, particles.stress,
            istep, t, VTK_DIR, eps_p=particles.eps_p, damage=particles.D)
        vtk_entries.append((t, fname))
        print(f"  step {istep:5d}/{N_STEPS}  t={t*1e3:6.3f} ms  "
              f"Dmax={particles.D.max():.3f}  eps_p_max={particles.eps_p.max():.4f}")

    # ── 35: advance time + refresh the particle-element map ─────────────
    _, mpoints = build_particle_element_map(particles.positions, mesh)
    t += dtime

# final frame
fname = write_particles_vtp_3d(
    particles.positions, particles.velocities, particles.stress,
    N_STEPS, t, VTK_DIR, eps_p=particles.eps_p, damage=particles.D)
vtk_entries.append((t, fname))
pvd = write_pvd(VTK_DIR, vtk_entries)

# ─────────────────────────────────────────────────────────────────────
# 8. POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────
print_section("Results")
n_dam  = int(np.sum(particles.D > 0.0))
n_fail = int(np.sum(particles.D >= 0.999))
print(f"VTK frames written:        {len(vtk_entries)}  ->  {pvd}")
print(f"Max equiv. plastic strain: {particles.eps_p.max():.4f}")
print(f"Max damage D:              {particles.D.max():.3f}")
print(f"Damaged particles (D>0):   {n_dam:,} / {pCount:,}")
print(f"Fully failed   (D>=1):     {n_fail:,}")
print(f"Max von-Mises stress:      "
      f"{max(_von_mises_full(s) for s in particles.stress) / 1e6:.1f} MPa")

scatter3d(particles.positions, particles.D,
          "Bone — final damage D", "bone_damage_3d.png", "damage D")
scatter3d(particles.positions, particles.eps_p,
          "Bone — final equiv. plastic strain", "bone_epsp_3d.png", "eps_p")

print("\nDone. Open", pvd, "in ParaView and colour by 'damage'.")
