"""Two-field (bulk + interfacial damage) kernels on a tiny grid: P2G pseudo-mass projection, the
two-field damage residual and the anisotropic total-strain stress update."""
import numpy as np
import warp as wp

from src.warp_usl_aniso import (damage_residual_two_field, grid_two_field_project, p2g_aniso,
                                reset_grid_aniso, stress_update_aniso)

DX = 0.2e-3            # square background cell (m), as in the Yvonnet driver
N_GRID = (10, 8)       # cells in x, y
D_0, ALPHA_0 = 0.37, 0.21
E, NU, RHO = 30.0e9, 0.25, 2450.0
THETA, XI = np.deg2rad(60.0), 30.0
GC_D, GC_A, ELL = 10.0, 1.0, 0.2e-3


def _layer_model():
    """CA1, M2 and omega_a exactly as the driver builds them."""
    c, s = np.cos(THETA), np.sin(THETA)
    pref = E / (1.0 - NU ** 2)
    A1 = pref * np.array([[0.8437, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    A2 = pref * np.array([[0.1563, NU, 0.0], [NU, 1.0, 0.0], [0.0, 0.0, (1.0 - NU) / 2.0]])
    T = np.array([[c * c, s * s, -2.0 * c * s],
                  [s * s, c * c, 2.0 * c * s],
                  [c * s, -c * s, c * c - s * s]])
    omega_a = np.eye(2) + XI * np.array([[c * c, s * c], [s * c, s * s]])
    return T @ A1 @ T.T, T @ A2 @ T.T, omega_a


def _mat33(a):
    return wp.mat33(*np.asarray(a, np.float32).ravel())


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


def _project(x_h, lp_h, d_h, alpha_h):
    n_p = len(x_h)
    n_nodes = (N_GRID[0] + 1) * (N_GRID[1] + 1)
    g = {k: wp.zeros(n_nodes, dtype=wp.float32)
         for k in ('m', 'pseudo_mass', 'dnum', 'anum', 'Fd', 'Fa', 'd', 'alpha')}
    for k in ('mv', 'f', 'fext'):
        g[k] = wp.zeros(n_nodes, dtype=wp.vec2)
    vol0 = np.full(n_p, DX * DX, np.float32)

    x = wp.array(x_h, dtype=wp.vec2)
    lp = wp.array(lp_h, dtype=wp.vec2)
    d = wp.array(d_h, dtype=wp.float32)
    alpha = wp.array(alpha_h, dtype=wp.float32)
    wp.launch(reset_grid_aniso, dim=n_nodes,
              inputs=[g['m'], g['mv'], g['f'], g['fext'], g['pseudo_mass'], g['dnum'], g['anum'],
                      g['Fd'], g['Fa']])
    wp.launch(p2g_aniso, dim=n_p,
              inputs=[x, wp.zeros(n_p, dtype=wp.vec2), wp.array(vol0 * RHO, dtype=wp.float32),
                      wp.array(vol0, dtype=wp.float32), wp.ones(n_p, dtype=wp.float32),
                      wp.zeros(n_p, dtype=wp.vec3), wp.zeros(n_p, dtype=wp.vec2), d, alpha, lp,
                      wp.float32(DX), N_GRID[0],
                      g['m'], g['mv'], g['f'], g['fext'], g['pseudo_mass'], g['dnum'], g['anum']])
    wp.launch(grid_two_field_project, dim=n_nodes,
              inputs=[g['pseudo_mass'], g['dnum'], g['anum'], g['d'], g['alpha']])
    return x, lp, d, alpha, wp.array(vol0, dtype=wp.float32), g


def test_uniform_fields_project_to_themselves():
    x_h, lp_h = _block_of_particles()
    n_p = len(x_h)
    *_, g = _project(x_h, lp_h, np.full(n_p, D_0, np.float32), np.full(n_p, ALPHA_0, np.float32))

    has_mass = g['pseudo_mass'].numpy() > 0.0
    assert has_mass.sum() > 20
    np.testing.assert_allclose(g['d'].numpy()[has_mass], D_0, rtol=1e-5)
    np.testing.assert_allclose(g['alpha'].numpy()[has_mass], ALPHA_0, rtol=1e-5)
    np.testing.assert_allclose(g['d'].numpy()[~has_mass], 0.0, atol=1e-12)
    np.testing.assert_allclose(g['alpha'].numpy()[~has_mass], 0.0, atol=1e-12)


def test_intact_unstrained_particles_have_zero_two_field_residual():
    x_h, lp_h = _block_of_particles()
    n_p = len(x_h)
    zeros = np.zeros(n_p, np.float32)
    x, lp, d, alpha, vol0, g = _project(x_h, lp_h, zeros, zeros)
    _, M2, omega_a = _layer_model()
    energy_d = wp.zeros(n_p, dtype=wp.float32)
    energy_a = wp.zeros(n_p, dtype=wp.float32)

    wp.launch(damage_residual_two_field, dim=n_p,
              inputs=[x, vol0, d, alpha, wp.zeros(n_p, dtype=wp.float32),
                      wp.zeros(n_p, dtype=wp.vec3), lp, g['d'], g['alpha'],
                      wp.float32(DX), N_GRID[0], wp.float32(GC_D), wp.float32(ELL),
                      wp.float32(GC_A), wp.float32(ELL),
                      wp.mat22(*np.asarray(omega_a, np.float32).ravel()), _mat33(M2),
                      g['Fd'], g['Fa'], energy_d, energy_a])

    np.testing.assert_allclose(g['Fd'].numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(g['Fa'].numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(energy_d.numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(energy_a.numpy(), 0.0, atol=1e-12)


def test_undamaged_stress_is_isotropic_plane_stress():
    rng = np.random.default_rng(3)
    n_p = 16
    dt = 8.0e-9
    eps0 = rng.uniform(-1e-4, 1e-4, (n_p, 3)).astype(np.float32)
    L_h = rng.uniform(-1e3, 1e3, (n_p, 2, 2)).astype(np.float32)
    CA1, M2, _ = _layer_model()

    F = wp.array(np.tile(np.eye(3, dtype=np.float32), (n_p, 1, 1)), dtype=wp.mat33)
    eps = wp.array(eps0, dtype=wp.vec3)
    sigma0 = wp.zeros(n_p, dtype=wp.vec3)
    sigma = wp.zeros(n_p, dtype=wp.vec3)
    H = wp.zeros(n_p, dtype=wp.float32)
    J = wp.zeros(n_p, dtype=wp.float32)
    wp.launch(stress_update_aniso, dim=n_p,
              inputs=[F, wp.array(L_h, dtype=wp.mat22), wp.zeros(n_p, dtype=wp.float32),
                      wp.zeros(n_p, dtype=wp.float32), _mat33(CA1), _mat33(M2), wp.float32(dt),
                      eps, sigma0, sigma, H, J])

    eps_h = eps0.astype(np.float64) + dt * np.column_stack(
        [L_h[:, 0, 0], L_h[:, 1, 1], L_h[:, 0, 1] + L_h[:, 1, 0]])
    C_iso = E / (1.0 - NU ** 2) * np.array([[1.0, NU, 0.0], [NU, 1.0, 0.0],
                                            [0.0, 0.0, (1.0 - NU) / 2.0]])
    sig_ref = eps_h @ C_iso.T
    scale = np.abs(sig_ref).max()
    np.testing.assert_allclose(eps.numpy(), eps_h, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(sigma.numpy(), sig_ref, atol=1e-5 * scale)
    np.testing.assert_allclose(sigma0.numpy(), sig_ref, atol=1e-5 * scale)
    np.testing.assert_allclose(H.numpy(), np.maximum(0.5 * np.sum(eps_h * sig_ref, axis=1), 0.0),
                               rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(J.numpy(), 1.0, rtol=1e-6)
