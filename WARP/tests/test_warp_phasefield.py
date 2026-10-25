"""Bulk-damage kernels on a tiny grid: P2G pseudo-mass projection and the damage residual."""
import numpy as np
import warp as wp

from src.warp_phasefield import damage_residual, grid_phase_project
from src.warp_usl_pf import p2g_pf, reset_grid_pf

DX = 0.25              # square background cell (mm), as in the plate driver
N_GRID = (10, 8)       # cells in x, y
PHI_0 = 0.37
CRACK_COEFF = 0.012    # Gc / (2 c_w ell) for the plate's AT2 parameters
L2 = 0.25 ** 2


def _block_of_particles():
    """1 particle per cell at cell centres over cells 2..7 x 2..5, with lp = h/2."""
    xs = (np.arange(2, 8) + 0.5) * DX
    ys = (np.arange(2, 6) + 0.5) * DX
    x = np.array([(a, b) for b in ys for a in xs], np.float32)
    # perturb a few particles so some domains span a node
    x[3] += (0.07 * DX, -0.04 * DX)
    x[10] += (-0.11 * DX, 0.09 * DX)
    lp = np.tile([0.5 * DX, 0.5 * DX], (len(x), 1)).astype(np.float32)
    lp[7] = (0.47 * DX, 0.52 * DX)
    return x, lp


def _grid_arrays(n_nodes):
    return dict(
        m=wp.zeros(n_nodes, dtype=wp.float32),
        mv=wp.zeros(n_nodes, dtype=wp.vec2),
        f=wp.zeros(n_nodes, dtype=wp.vec2),
        fext=wp.zeros(n_nodes, dtype=wp.vec2),
        pseudo_mass=wp.zeros(n_nodes, dtype=wp.float32),
        phinum=wp.zeros(n_nodes, dtype=wp.float32),
        F=wp.zeros(n_nodes, dtype=wp.float32),
        phi=wp.zeros(n_nodes, dtype=wp.float32),
    )


def _project(x_h, lp_h, phi_h):
    n_p = len(x_h)
    n_nodes = (N_GRID[0] + 1) * (N_GRID[1] + 1)
    g = _grid_arrays(n_nodes)
    vol0 = np.full(n_p, DX * DX, np.float32)

    x = wp.array(x_h, dtype=wp.vec2)
    lp = wp.array(lp_h, dtype=wp.vec2)
    phi = wp.array(phi_h, dtype=wp.float32)
    wp.launch(reset_grid_pf, dim=n_nodes,
              inputs=[g['m'], g['mv'], g['f'], g['fext'], g['pseudo_mass'], g['phinum'], g['F']])
    wp.launch(p2g_pf, dim=n_p,
              inputs=[x, wp.zeros(n_p, dtype=wp.vec2), wp.array(vol0 * 2.45e-9, dtype=wp.float32),
                      wp.array(vol0, dtype=wp.float32), wp.ones(n_p, dtype=wp.float32),
                      wp.zeros(n_p, dtype=wp.vec3), wp.zeros(n_p, dtype=wp.vec2), phi, lp,
                      wp.float32(DX), N_GRID[0],
                      g['m'], g['mv'], g['f'], g['fext'], g['pseudo_mass'], g['phinum']])
    wp.launch(grid_phase_project, dim=n_nodes, inputs=[g['pseudo_mass'], g['phinum'], g['phi']])
    return x, lp, phi, wp.array(vol0, dtype=wp.float32), g


def test_uniform_phase_projects_to_itself():
    x_h, lp_h = _block_of_particles()
    phi_h = np.full(len(x_h), PHI_0, np.float32)
    *_, g = _project(x_h, lp_h, phi_h)

    pseudo_mass = g['pseudo_mass'].numpy()
    phi_I = g['phi'].numpy()
    has_mass = pseudo_mass > 0.0
    assert has_mass.sum() > 20
    np.testing.assert_allclose(phi_I[has_mass], PHI_0, rtol=1e-5)
    np.testing.assert_allclose(phi_I[~has_mass], 0.0, atol=1e-12)


def test_intact_unloaded_particles_have_zero_damage_residual():
    x_h, lp_h = _block_of_particles()
    n_p = len(x_h)
    x, lp, phi, vol0, g = _project(x_h, lp_h, np.zeros(n_p, np.float32))
    H = wp.zeros(n_p, dtype=wp.float32)
    crack_energy = wp.zeros(n_p, dtype=wp.float32)

    wp.launch(damage_residual, dim=n_p,
              inputs=[x, vol0, phi, H, lp, g['phi'], wp.float32(DX), N_GRID[0],
                      wp.float32(CRACK_COEFF), wp.float32(L2), g['F'], crack_energy])

    np.testing.assert_allclose(g['F'].numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(crack_energy.numpy(), 0.0, atol=1e-12)
