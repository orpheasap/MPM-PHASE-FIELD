# %%
# =============================================================================
#  resume_Yvonnet.py  --  Continue a two-field ANISOTROPIC USL-MPM-PF run from
#                         vtk_output_Yvonnet/checkpoint.npz
#
#  Companion to `Yvonnet_plate.py` (theory: aniso_layer_PF_MPM_explicit.tex).
#  Same role as `resume_aniso.py` but for the smaller 10mm x 10mm Yvonnet-style
#  plate (bulk-d + interfacial-alpha model, SI units).
#
#  Usage:
#      python resume_Yvonnet.py                 # runs PF_NSTEPS steps per batch (default 10)
#      PF_NSTEPS=500 python resume_Yvonnet.py   # override batch size
#
#  After each batch the script saves the checkpoint + PVD and asks whether to
#  continue.  Press 'y' (or wait 10 s) to run another batch; anything else stops.
# =============================================================================
import os
import msvcrt
import time
import numpy as np

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.solver_utilities import NodeState, get_mpm2d_shape, build_particle_element_map
from src.vtk_export import write_pvd, write_particles_vtp_aniso


np.set_printoptions(precision=4, suppress=True)

CHECKPOINT_FILE = os.path.join('vtk_output_Yvonnet', 'checkpoint.npz')

if not os.path.exists(CHECKPOINT_FILE):
    raise FileNotFoundError(
        f"No checkpoint found at '{CHECKPOINT_FILE}'.\n"
        "Run Yvonnet_plate.py first to generate one."
    )


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


def _input_with_timeout(prompt, timeout=10, default='y'):
    """Prompt with a live countdown; no background threads (msvcrt polling)."""
    print(prompt, flush=True)
    chars = []
    deadline = time.monotonic() + timeout
    last_remaining = -1
    while True:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            print('\r  [ auto-continuing ]              ', flush=True)
            return default
        if remaining != last_remaining:
            print(f'\r  {remaining:2d} s ', end='', flush=True)
            last_remaining = remaining
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\r', '\n'):
                print(flush=True)
                return ''.join(chars)
            elif ch == '\x03':      # Ctrl+C
                raise KeyboardInterrupt
            elif ch == '\x08':      # backspace
                if chars:
                    chars.pop()
                    print('\b \b', end='', flush=True)
            else:
                chars.append(ch)
                print(ch, end='', flush=True)
        time.sleep(0.05)


def _save_checkpoint(path, pCount, particles, alpha_p, neumann_traction_y,
                     theta_deg, xi, eta_d, eta_alpha,
                     t, istep_start, ta, ka, sa, dda, aa, vtk_entries):
    np.savez_compressed(
        path,
        pCount               = np.array(pCount),
        positions            = particles.positions,
        velocities           = particles.velocities,
        mass                 = particles.mass,
        volume               = particles.volume,
        deformation_gradient = particles.deformation_gradient,
        stress               = particles.stress,
        eff_stress           = particles.eff_stress,
        strain               = particles.strain,
        phase                = particles.phase,     # bulk damage d
        alpha                = alpha_p,             # interfacial damage alpha
        history              = particles.history,
        initial_positions    = particles.initial_positions,
        initial_volume       = particles.initial_volume,
        neumann_particles    = particles.neumann_particles,
        neumann_traction_y   = neumann_traction_y,
        theta_deg            = np.array(theta_deg),
        xi                   = np.array(xi),
        eta_d                = np.array(eta_d),
        eta_alpha            = np.array(eta_alpha),
        t                    = np.array(t),
        istep_start          = np.array(istep_start),
        ta = np.array(ta), ka = np.array(ka), sa = np.array(sa),
        dda = np.array(dda), aa = np.array(aa),
        vtk_times  = np.array([e[0] for e in vtk_entries]),
        vtk_fnames = np.array([e[1] for e in vtk_entries]),
    )


# =============================================================================
#  Material + anisotropic model  (must match the original run, SI units)
# =============================================================================
E   = 10.0e9      # Pa
nu  = 0.25
rho = 2450.0      # kg/m^3
material = Material(E, nu, rho, stressState='PLANE_STRESS')

theta_deg = 60.0
xi        = 30.0
theta     = np.deg2rad(theta_deg)
c, s      = np.cos(theta), np.sin(theta)
pref      = E / (1.0 - nu**2)

A1 = pref * np.array([[0.8437, 0.0, 0.0],
                      [0.0,    0.0, 0.0],
                      [0.0,    0.0, 0.0]])
A2 = pref * np.array([[0.1563, nu,  0.0],
                      [nu,     1.0, 0.0],
                      [0.0,    0.0, (1.0 - nu) / 2.0]])
T_sig = np.array([[c*c,  s*s, -2.0*c*s],
                  [s*s,  c*c,  2.0*c*s],
                  [c*s, -c*s,  c*c - s*s]])
CA1 = T_sig @ A1 @ T_sig.T
M2  = T_sig @ A2 @ T_sig.T

def C_aniso(alpha):
    return CA1 + M2 * (1.0 - alpha) ** 2

P_a     = np.array([[c*c, s*c], [s*c, s*s]])
omega_a = np.eye(2) + xi * P_a

# Fracture properties (SI)
gc_d     = 4.0e3
gc_alpha = 1.0e3


# =============================================================================
#  Background grid  (must match the original run, SI units)
# =============================================================================
_h_scale = float(os.environ.get("PF_H_SCALE", "1.0"))
Lx   = 0.01
Ly   = 0.01
hx   = 0.25e-3 * _h_scale
hy   = 0.25e-3 * _h_scale
numx = int(round(Lx / hx))
numy = int(round(Ly / hy))
h_e  = min(hx, hy)
ell_d     = 2.0 * h_e
ell_alpha = 2.0 * h_e

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid (SI)")
print(f"Cells: {numx} x {numy},  h = {hx:.4g} m,  nodes: {mesh.nodeCount}")
print(f"theta = {theta_deg} deg,  xi = {xi},  ell_d = ell_alpha = {ell_d:.4g} m")


# =============================================================================
#  Load checkpoint
# =============================================================================
print_section(f"Loading checkpoint:  {CHECKPOINT_FILE}")
ckpt = np.load(CHECKPOINT_FILE, allow_pickle=True)

pCount    = int(ckpt['pCount'])
particles = ParticleSet(pCount)
particles.positions[:]            = ckpt['positions']
particles.velocities[:]           = ckpt['velocities']
particles.mass[:]                 = ckpt['mass']
particles.volume[:]               = ckpt['volume']
particles.deformation_gradient[:] = ckpt['deformation_gradient']
particles.stress[:]               = ckpt['stress']
particles.eff_stress[:]           = ckpt['eff_stress']
particles.strain[:]               = ckpt['strain']
particles.phase[:]                = ckpt['phase']      # bulk damage d
particles.history[:]              = ckpt['history']
particles.initial_positions[:]    = ckpt['initial_positions']
particles.initial_volume[:]       = ckpt['initial_volume']
particles.neumann_particles[:]    = ckpt['neumann_particles']
alpha_p            = np.array(ckpt['alpha'])           # interfacial damage alpha
neumann_traction_y = np.array(ckpt['neumann_traction_y'])

t0          = float(ckpt['t'])
istep_start = int(ckpt['istep_start'])
ta  = list(ckpt['ta'])
ka  = list(ckpt['ka'])
sa  = list(ckpt['sa'])
dda = list(ckpt['dda'])
aa  = list(ckpt['aa'])

vtk_fnames  = [str(f) for f in ckpt['vtk_fnames']]
vtk_entries = list(zip(ckpt['vtk_times'].tolist(), vtk_fnames))

print(f"  Particles:            {pCount}")
print(f"  Resuming at step:     {istep_start},  t = {t0*1e6:.3f} us")
print(f"  Max bulk d so far:    {max(dda):.4f}" if dda else "  Max bulk d so far: 0.0000")
print(f"  Max alpha so far:     {max(aa):.4f}" if aa else "  Max alpha so far: 0.0000")
print(f"  Existing VTK frames:  {len(vtk_entries)}")


# ----- Element-particle mapping & node state -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)
node_state = NodeState(mesh.nodeCount)


# =============================================================================
#  Solver parameters -- three-way CFL (identical construction to the driver)
# =============================================================================
C_max  = float(np.max(np.linalg.eigvalsh(C_aniso(0.0))))
c_d    = np.sqrt(C_max / rho)
dt_u   = h_e / c_d
ndim   = 2
eta_d     = 2.0 * ndim * gc_d     * ell_d               / (h_e * c_d)
eta_alpha = 2.0 * ndim * gc_alpha * ell_alpha * (1.0 + xi) / (h_e * c_d)

CFL    = float(os.environ.get("PF_CFL", "0.5"))
dtime  = CFL * dt_u          # all three critical steps coincide with dt_u
nsteps = int(os.environ.get("PF_NSTEPS", "500"))   # steps per batch

print_section("Solver Parameters")
print(f"c_d = {c_d:.4e} m/s,  dt_u = dt_d = dt_alpha = {dt_u:.4e} s")
print(f"eta_d = {eta_d:.4e} Pa*s,  eta_alpha = {eta_alpha:.4e} Pa*s")
print(f"dtime (SF {CFL}): {dtime:.4e} s,  batch size: {nsteps} steps")


# %%  ----- Interactive batch loop -----
tol      = 1e-24
tol_pm   = 1e-30
vtk_dir  = 'vtk_output_Yvonnet'
vtk_interval = int(os.environ.get("PF_VTK_INTERVAL", "50"))

I_mat   = np.eye(2)
mpoints = particles.mpoints
t       = t0

alpha_grid = np.zeros(mesh.nodeCount)
alpha_num  = np.zeros(mesh.nodeCount)
F_alpha    = np.zeros(mesh.nodeCount)

while True:
    batch_start = istep_start
    batch_end   = istep_start + nsteps - 1
    print(f"\nRunning steps {batch_start} – {batch_end}  "
          f"(t = {t*1e6:.3f} → {(t + nsteps*dtime)*1e6:.3f} us) ...")

    for istep in range(istep_start, istep_start + nsteps):
        node_state.reset()
        alpha_grid.fill(0.0)
        alpha_num.fill(0.0)
        F_alpha.fill(0.0)

        # --- P2G: momentum + both damage projections ---
        for e in range(mesh.elemCount):
            esctr = mesh.element[e]
            for pid in mpoints[e]:
                stress = particles.stress[pid]
                d_p    = particles.phase[pid]
                a_p    = alpha_p[pid]
                V0     = particles.initial_volume[pid]
                for idn in esctr:
                    dx      = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(dx, hx, hy)

                    node_state.mass[idn]              += N * particles.mass[pid]
                    node_state.momentum[idn]          += N * particles.mass[pid] * particles.velocities[pid]
                    node_state.internal_force[idn, 0] -= particles.volume[pid] * (stress[0]*dNdx[0] + stress[2]*dNdx[1])
                    node_state.internal_force[idn, 1] -= particles.volume[pid] * (stress[2]*dNdx[0] + stress[1]*dNdx[1])
                    if particles.neumann_particles[pid]:
                        node_state.external_force[idn, 1] += neumann_traction_y[pid] * N * particles.volume[pid]
                    node_state.pseudo_mass[idn] += N * V0
                    node_state.phase_num[idn]   += N * V0 * d_p
                    alpha_num[idn]              += N * V0 * a_p

        # --- Grid damage fields ---
        has_pm = node_state.pseudo_mass > tol_pm
        node_state.phase[has_pm] = node_state.phase_num[has_pm] / node_state.pseudo_mass[has_pm]
        alpha_grid[has_pm]       = alpha_num[has_pm]      / node_state.pseudo_mass[has_pm]

        # --- Assemble both damage residuals ---
        for e in range(mesh.elemCount):
            esctr = mesh.element[e]
            for pid in mpoints[e]:
                V0    = particles.initial_volume[pid]
                d_p   = particles.phase[pid]
                H_p   = particles.history[pid]
                eps_p = particles.strain[pid]

                grad_d = np.zeros(2)
                grad_a = np.zeros(2)
                shp = []
                for idn in esctr:
                    dx      = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(dx, hx, hy)
                    grad_d += dNdx * node_state.phase[idn]
                    grad_a += dNdx * alpha_grid[idn]
                    shp.append((idn, N, dNdx))

                g_d     = (1.0 - d_p) ** 2
                G_alpha = (1.0 - alpha_p[pid]) * g_d * float(eps_p @ (M2 @ eps_p))
                omega_grad_a = omega_a @ grad_a

                for idn, N, dNdx in shp:
                    node_state.damage_force[idn] += (
                        (gc_d / ell_d) * d_p * N
                        + gc_d * ell_d * (dNdx @ grad_d)
                        - 2.0 * (1.0 - d_p) * N * H_p
                    ) * V0
                    F_alpha[idn] += (
                        (gc_alpha / ell_alpha) * alpha_p[pid] * N
                        + gc_alpha * ell_alpha * (dNdx @ omega_grad_a)
                        - G_alpha * N
                    ) * V0

        # --- Momentum update ---
        node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime

        # --- Explicit grid damage updates ---
        d_grid_new = node_state.phase.copy()
        a_grid_new = alpha_grid.copy()
        d_grid_new[has_pm] -= (dtime / eta_d)     * node_state.damage_force[has_pm] / node_state.pseudo_mass[has_pm]
        a_grid_new[has_pm] -= (dtime / eta_alpha) * F_alpha[has_pm]                 / node_state.pseudo_mass[has_pm]

        # --- G2P ---
        k = 0.0
        u = 0.0
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                Lp    = np.zeros((2, 2))
                d_new = 0.0
                a_new = 0.0
                for idn in mesh.element[e]:
                    dx      = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(dx, hx, hy)
                    if node_state.mass[idn] > tol:
                        particles.velocities[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn]
                        particles.positions[pid]  += dtime * N * node_state.momentum[idn] / node_state.mass[idn]
                        vI = node_state.momentum[idn] / node_state.mass[idn]
                    else:
                        vI = np.zeros(2)
                    Lp    += np.outer(vI, dNdx)
                    d_new += N * d_grid_new[idn]
                    a_new += N * a_grid_new[idn]

                d_new = min(max(d_new, 0.0), 1.0)
                a_new = min(max(a_new, 0.0), 1.0)
                d_new = max(particles.phase[pid], d_new)
                a_new = max(alpha_p[pid],         a_new)
                particles.phase[pid] = d_new
                alpha_p[pid]         = a_new

                F = (I_mat + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
                particles.deformation_gradient[pid] = F.reshape(4)
                particles.volume[pid]               = np.linalg.det(F) * particles.initial_volume[pid]
                dEps     = 0.5 * dtime * (Lp + Lp.T)
                dEps_vec = np.array([dEps[0, 0], dEps[1, 1], 2.0 * dEps[0, 1]])
                particles.strain[pid] += dEps_vec

                eps_vec = particles.strain[pid]
                sig0    = C_aniso(a_new) @ eps_vec
                particles.eff_stress[pid] = sig0
                particles.stress[pid]     = ((1.0 - d_new) ** 2) * sig0

                psi_e0 = 0.5 * float(eps_vec @ sig0)
                if psi_e0 > particles.history[pid]:
                    particles.history[pid] = psi_e0

                k += 0.5 * (particles.velocities[pid, 0]**2 + particles.velocities[pid, 1]**2) * particles.mass[pid]
                u += 0.5 * particles.volume[pid] * particles.stress[pid] @ particles.strain[pid]

        ta.append(t); ka.append(k); sa.append(u)
        dda.append(float(np.max(particles.phase)))
        aa.append(float(np.max(alpha_p)))

        if istep % vtk_interval == 0:
            fname = write_particles_vtp_aniso(
                particles.positions, particles.velocities, particles.stress,
                particles.phase, alpha_p, particles.history,
                istep, t, vtk_dir,
            )
            vtk_entries.append((t, fname))

        _, mpoints = build_particle_element_map(particles.positions, mesh)
        t += dtime

    # --- End of batch ---
    istep_start += nsteps
    print(f"Batch done.  Steps {batch_start}–{batch_end}  |  t = {t*1e6:.3f} us  "
          f"|  max d = {max(dda):.4f}  |  max alpha = {max(aa):.4f}")

    pvd_path = write_pvd(vtk_dir, vtk_entries)
    print(f"simulation.pvd updated  ({len(vtk_entries)} frames total)")

    _save_checkpoint(CHECKPOINT_FILE, pCount, particles, alpha_p, neumann_traction_y,
                     theta_deg, xi, eta_d, eta_alpha,
                     t, istep_start, ta, ka, sa, dda, aa, vtk_entries)
    print(f"Checkpoint saved  (next batch starts at step {istep_start})")

    ans = _input_with_timeout(
        f"\nContinue for {nsteps} more steps? [y/N]  (auto-continues in 10 s): ",
        timeout=10, default='y',
    ).strip().lower()
    if ans not in ('y', 'yes'):
        print("Stopping.")
        break
