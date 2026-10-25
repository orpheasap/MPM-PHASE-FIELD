# %%
# =============================================================================
#  resume.py  --  Continue a USL-MPM-PF run from vtk_output/checkpoint.npz
#
#  Usage:
#      python resume.py                     # runs PF_NSTEPS steps per batch (default 4)
#      PF_NSTEPS=200 python resume.py       # override batch size
#
#  After each batch the script saves the checkpoint + PVD and asks whether to
#  continue.  Answer 'y' to run another batch, anything else to stop.
# =============================================================================
import os
import msvcrt
import time
import numpy as np

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.solver_utilities import NodeState, get_mpm2d_shape, build_particle_element_map
from src.vtk_export import write_pvd, write_particles_vtp
from src.phasefield import PhaseFieldModel, elastic_energy_density


np.set_printoptions(precision=3, suppress=True)

CHECKPOINT_FILE = os.path.join('vtk_output', 'checkpoint.npz')

if not os.path.exists(CHECKPOINT_FILE):
    raise FileNotFoundError(
        f"No checkpoint found at '{CHECKPOINT_FILE}'.\n"
        "Run pre_noched_plate_PF.py first to generate one."
    )


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


def _input_with_timeout(prompt, timeout=10, default='y'):
    """Prompt with a live countdown; no background threads — uses msvcrt polling."""
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


def _save_checkpoint(path, pCount, particles, neumann_traction_y,
                     t, istep_start, ta, ka, sa, da, vtk_entries):
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
        phase                = particles.phase,
        history              = particles.history,
        initial_positions    = particles.initial_positions,
        initial_volume       = particles.initial_volume,
        neumann_particles    = particles.neumann_particles,
        neumann_traction_y   = neumann_traction_y,
        t                    = np.array(t),
        istep_start          = np.array(istep_start),
        ta                   = np.array(ta),
        ka                   = np.array(ka),
        sa                   = np.array(sa),
        da                   = np.array(da),
        vtk_times            = np.array([e[0] for e in vtk_entries]),
        vtk_fnames           = np.array([e[1] for e in vtk_entries]),
    )


# ----- Material & PF model (must match original run) -----
E   = 32e3
nu  = 0.2
rho = 2.45e-9
material = Material(E, nu, rho, stressState='PLANE_STRAIN')

ell = 0.25
Gc  = 3e-3
eta = 1e-9
pf  = PhaseFieldModel(Gc=Gc, ell=ell, eta=eta, model='AT2', kappa=1e-7)


# ----- Background grid (must match original run) -----
_h_scale = float(os.environ.get("PF_H_SCALE", "1.0"))
Lx   = 100.0
Ly   = 40.0
hx   = 0.25 * _h_scale
hy   = 0.25 * _h_scale
numx = int(round(Lx / hx))
numy = int(round(Ly / hy))

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Background Grid")
print(f"Cells: {numx} x {numy},  h = {hx} mm,  nodes: {mesh.nodeCount}")


# ----- Load checkpoint -----
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
particles.phase[:]                = ckpt['phase']
particles.history[:]              = ckpt['history']
particles.initial_positions[:]    = ckpt['initial_positions']
particles.initial_volume[:]       = ckpt['initial_volume']
particles.neumann_particles[:]    = ckpt['neumann_particles']
neumann_traction_y                = np.array(ckpt['neumann_traction_y'])

t0          = float(ckpt['t'])
istep_start = int(ckpt['istep_start'])
ta = list(ckpt['ta'])
ka = list(ckpt['ka'])
sa = list(ckpt['sa'])
da = list(ckpt['da'])

vtk_fnames  = [str(f) for f in ckpt['vtk_fnames']]
vtk_entries = list(zip(ckpt['vtk_times'].tolist(), vtk_fnames))

print(f"  Particles:             {pCount}")
print(f"  Resuming at step:      {istep_start},  t = {t0*1e6:.3f} us")
print(f"  Max damage so far:     {max(da):.4f}" if da else "  Max damage so far: 0.0000")
print(f"  Existing VTK frames:   {len(vtk_entries)}")


# ----- Element-particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)

# ----- Node state -----
node_state = NodeState(mesh.nodeCount)

# ----- Solver parameters -----
c_wave = np.sqrt(material.E / material.density)
dt_cfl = hx / c_wave
dt_pf  = pf.dt_crit_phase(min(hx, hy), ndim=2)
CFL    = 0.5
dtime  = CFL * min(dt_cfl, dt_pf)
nsteps = int(os.environ.get("PF_NSTEPS", "50"))   # steps per batch

print_section("Solver Parameters")
print(f"dtime (CFL {CFL}):    {dtime:.4e} s")
print(f"Batch size:        {nsteps} steps")


# %%  ----- Interactive batch loop -----
tol            = 1e-20
tol_phi        = 1e-30
vtk_output_dir = 'vtk_output'
vtk_interval   = 10

I_mat   = np.eye(2)
mpoints = particles.mpoints
t       = t0

C_mat       = material.elasticity_matrix
crack_coeff = pf.crack_force_coeff
l2          = pf.ell ** 2
kappa       = pf.kappa
inv_eta     = 1.0 / pf.eta

while True:
    batch_start = istep_start
    batch_end   = istep_start + nsteps - 1
    print(f"\nRunning steps {batch_start} – {batch_end}  (t = {t*1e6:.3f} → {(t + nsteps*dtime)*1e6:.3f} us) ...")

    for istep in range(istep_start, istep_start + nsteps):
        node_state.reset()

        # --- P2G ---
        for e in range(mesh.elemCount):
            esctr = mesh.element[e]
            for pid in mpoints[e]:
                stress = particles.stress[pid]
                phi_p  = particles.phase[pid]
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
                    node_state.phase_num[idn]   += N * V0 * phi_p

        # --- Grid phase field ---
        has_phi = node_state.pseudo_mass > tol_phi
        node_state.phase[has_phi] = node_state.phase_num[has_phi] / node_state.pseudo_mass[has_phi]

        # --- Damage residual F_I ---
        for e in range(mesh.elemCount):
            esctr = mesh.element[e]
            for pid in mpoints[e]:
                V0    = particles.initial_volume[pid]
                phi_p = particles.phase[pid]
                H_p   = particles.history[pid]
                gp    = pf.dg(phi_p)
                wp    = pf.dw(phi_p)
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

        # --- Momentum update ---
        node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime

        # --- Grid phase update ---
        phi_grid_new = node_state.phase.copy()
        phi_grid_new[has_phi] -= (dtime * inv_eta) * node_state.damage_force[has_phi] / node_state.pseudo_mass[has_phi]

        # --- G2P ---
        k = 0.0
        u = 0.0
        for e in range(mesh.elemCount):
            for pid in mpoints[e]:
                Lp      = np.zeros((2, 2))
                phi_new = 0.0
                for idn in mesh.element[e]:
                    dx      = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(dx, hx, hy)
                    vI = np.zeros(2)
                    if node_state.mass[idn] > tol:
                        particles.velocities[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn]
                        particles.positions[pid]  += dtime * N * node_state.momentum[idn] / node_state.mass[idn]
                        vI = node_state.momentum[idn] / node_state.mass[idn]
                    Lp      += np.outer(vI, dNdx)
                    phi_new += N * phi_grid_new[idn]

                phi_new = min(max(phi_new, 0.0), 1.0)
                phi_new = max(particles.phase[pid], phi_new)
                particles.phase[pid] = phi_new

                F = (I_mat + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
                particles.deformation_gradient[pid] = F.reshape(4)
                particles.volume[pid]               = np.linalg.det(F) * particles.initial_volume[pid]
                dEps     = 0.5 * dtime * (Lp + Lp.T)
                dEps_vec = np.array([dEps[0, 0], dEps[1, 1], 2.0*dEps[0, 1]])
                particles.strain[pid] += dEps_vec

                particles.eff_stress[pid] += C_mat @ dEps_vec
                g_phi = pf.g(phi_new)
                particles.stress[pid] = (g_phi + kappa) * particles.eff_stress[pid]

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

    # --- End of batch ---
    istep_start += nsteps

    print(f"Batch done.  Steps {batch_start}–{batch_end}  |  t = {t*1e6:.3f} us  |  max phi = {max(da):.4f}")

    pvd_path = write_pvd(vtk_output_dir, vtk_entries)
    print(f"simulation.pvd updated  ({len(vtk_entries)} frames total)")

    _save_checkpoint(CHECKPOINT_FILE, pCount, particles, neumann_traction_y,
                     t, istep_start, ta, ka, sa, da, vtk_entries)
    print(f"Checkpoint saved  (next batch starts at step {istep_start})")

    ans = _input_with_timeout(
        f"\nContinue for {nsteps} more steps? [y/N]  (auto-continues in 10 s): ",
        timeout=10, default='y',
    ).strip().lower()
    if ans not in ('y', 'yes'):
        print("Stopping.")
        break