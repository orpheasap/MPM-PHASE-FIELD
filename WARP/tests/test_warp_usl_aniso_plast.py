"""Elastoplastic stress update of the Yvonnet plate: the plane-stress closest-point return on the
effective stress with the full C_a(alpha), checked kernel by kernel on a handful of particles."""
import numpy as np
import warp as wp

from src.warp_usl_aniso import stress_update_aniso
from src.warp_usl_aniso_plast import stress_update_aniso_plast

E, NU = 30.0e9, 0.25
THETA = np.deg2rad(60.0)
SIGMA_Y = 10.0e6       # yield stress (Pa), as in the driver
K_H = 1.5e9            # hardening modulus (Pa), as in the driver
DT = 8.0e-9
VOL0 = (0.05e-3) ** 2
P_VM = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]]) / 3.0


def _layer_model():
    """CA1 and M2 exactly as the driver builds them."""
    c, s = np.cos(THETA), np.sin(THETA)
    pref = E / (1.0 - NU ** 2)
    A1 = pref * np.array([[0.8437, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    A2 = pref * np.array([[0.1563, NU, 0.0], [NU, 1.0, 0.0], [0.0, 0.0, (1.0 - NU) / 2.0]])
    T = np.array([[c * c, s * s, -2.0 * c * s],
                  [s * s, c * c, 2.0 * c * s],
                  [c * s, -c * s, c * c - s * s]])
    return T @ A1 @ T.T, T @ A2 @ T.T


def _c_iso():
    return E / (1.0 - NU ** 2) * np.array([[1.0, NU, 0.0], [NU, 1.0, 0.0],
                                           [0.0, 0.0, (1.0 - NU) / 2.0]])


def _mat33(a):
    return wp.mat33(*np.asarray(a, np.float32).ravel())


def _q(s):
    """Plane-stress von Mises equivalent stress of Voigt rows (11, 22, 12)."""
    return np.sqrt(1.5 * np.einsum('ni,ij,nj->n', s, P_VM, s))


def _run_plast(L_h, d_h, alpha_h, eps0, eps_p0, p0, H0, CA1, M2, sigma_y=SIGMA_Y, k_h=K_H,
               step=0, nsteps=1):
    """One launch of stress_update_aniso_plast; returns every output as a float64 numpy array."""
    n_p = len(L_h)
    f32 = np.float32
    out = {
        'eps': wp.array(np.asarray(eps0, f32), dtype=wp.vec3),
        'eps_p': wp.array(np.asarray(eps_p0, f32), dtype=wp.vec3),
        'p': wp.array(np.asarray(p0, f32), dtype=wp.float32),
        'eps_e': wp.zeros(n_p, dtype=wp.vec3),
        'sigma0': wp.zeros(n_p, dtype=wp.vec3),
        'sigma': wp.zeros(n_p, dtype=wp.vec3),
        'H': wp.array(np.asarray(H0, f32), dtype=wp.float32),
        'J': wp.zeros(n_p, dtype=wp.float32),
        'w_p': wp.zeros(nsteps, dtype=wp.float32),
        'p_max': wp.zeros(nsteps, dtype=wp.float32),
    }
    F = wp.array(np.tile(np.eye(3, dtype=f32), (n_p, 1, 1)), dtype=wp.mat33)
    wp.launch(stress_update_aniso_plast, dim=n_p,
              inputs=[F, wp.array(np.asarray(L_h, f32), dtype=wp.mat22),
                      wp.array(np.asarray(d_h, f32), dtype=wp.float32),
                      wp.array(np.asarray(alpha_h, f32), dtype=wp.float32),
                      wp.array(np.full(n_p, VOL0, f32), dtype=wp.float32),
                      _mat33(CA1), _mat33(M2), f32(sigma_y), f32(k_h), f32(DT), step,
                      out['eps'], out['eps_p'], out['p'], out['eps_e'], out['sigma0'],
                      out['sigma'], out['H'], out['J'], out['w_p'], out['p_max']])
    return {k: a.numpy().astype(np.float64) for k, a in out.items()}


# Layer direction e1' as a Voigt strain (engineering shear): stretching along it loads the
# stiffness that survives full interfacial damage, C_a(1) = CA1.
LAYER_STRAIN = np.array([np.cos(THETA) ** 2, np.sin(THETA) ** 2,
                         2.0 * np.cos(THETA) * np.sin(THETA)])


def _layer_L(n_p, rate):
    """Velocity gradients that stretch along the layer direction e1' at `rate` (1/s), plus noise."""
    rng = np.random.default_rng(11)
    e1 = np.array([np.cos(THETA), np.sin(THETA)])
    return rate * (np.outer(e1, e1)[None] + rng.uniform(-0.1, 0.1, (n_p, 2, 2)))


def test_no_yield_reproduces_elastic_stress_update():
    rng = np.random.default_rng(3)
    n_p = 16
    eps0 = rng.uniform(-1e-4, 1e-4, (n_p, 3)).astype(np.float32)
    L_h = rng.uniform(-1e3, 1e3, (n_p, 2, 2)).astype(np.float32)
    d_h = rng.uniform(0.0, 0.9, n_p).astype(np.float32)
    a_h = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
    H0 = rng.uniform(0.0, 50.0, n_p).astype(np.float32)
    CA1, M2 = _layer_model()

    ref = {'eps': wp.array(eps0, dtype=wp.vec3), 'sigma0': wp.zeros(n_p, dtype=wp.vec3),
           'sigma': wp.zeros(n_p, dtype=wp.vec3), 'H': wp.array(H0, dtype=wp.float32),
           'J': wp.zeros(n_p, dtype=wp.float32)}
    F = wp.array(np.tile(np.eye(3, dtype=np.float32), (n_p, 1, 1)), dtype=wp.mat33)
    wp.launch(stress_update_aniso, dim=n_p,
              inputs=[F, wp.array(L_h, dtype=wp.mat22), wp.array(d_h, dtype=wp.float32),
                      wp.array(a_h, dtype=wp.float32), _mat33(CA1), _mat33(M2),
                      wp.float32(DT), ref['eps'], ref['sigma0'], ref['sigma'], ref['H'],
                      ref['J']])
    ref = {k: a.numpy().astype(np.float64) for k, a in ref.items()}

    got = _run_plast(L_h, d_h, a_h, eps0, np.zeros((n_p, 3)), np.zeros(n_p), H0, CA1, M2,
                     sigma_y=1e30)

    scale = np.abs(ref['sigma0']).max()
    np.testing.assert_allclose(got['eps'], ref['eps'], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got['eps_e'], ref['eps'], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(got['sigma0'], ref['sigma0'], atol=1e-6 * scale)
    np.testing.assert_allclose(got['sigma'], ref['sigma'], atol=1e-6 * scale)
    np.testing.assert_allclose(got['H'], ref['H'], rtol=1e-5)
    np.testing.assert_allclose(got['J'], ref['J'], rtol=1e-6)
    np.testing.assert_allclose(got['eps_p'], 0.0, atol=1e-12)
    np.testing.assert_allclose(got['p'], 0.0, atol=1e-12)
    np.testing.assert_allclose(got['w_p'][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(got['p_max'][0], 0.0, atol=1e-12)


def test_yielding_states_return_onto_the_yield_surface():
    CA1, M2 = _layer_model()
    n_p = 12
    rng = np.random.default_rng(5)
    alpha_h = np.repeat([0.0, 0.5, 1.0], n_p // 3)
    # Total strain well past first yield (sigma_y / E ~ 3.3e-4) along e1', a prior plastic state
    eps0 = 2e-3 * LAYER_STRAIN + rng.uniform(-3e-4, 3e-4, (n_p, 3))
    eps_p0 = rng.uniform(-1e-4, 1e-4, (n_p, 3))
    p0 = rng.uniform(0.0, 2e-4, n_p)
    L_h = _layer_L(n_p, 1e3)
    d_h = rng.uniform(0.0, 0.5, n_p)

    got = _run_plast(L_h, d_h, alpha_h, eps0, eps_p0, p0, np.zeros(n_p), CA1, M2)

    C = CA1[None] + M2[None] * ((1.0 - alpha_h) ** 2)[:, None, None]
    s_tr = np.einsum('nij,nj->ni', C, got['eps'] - eps_p0)
    assert np.all(_q(s_tr) > SIGMA_Y + K_H * p0), "every state must yield"
    assert np.all(got['p'] > p0)

    s0 = got['sigma0']
    np.testing.assert_allclose(_q(s0), SIGMA_Y + K_H * got['p'], rtol=1e-4)
    s_el = np.einsum('nij,nj->ni', C, got['eps'] - got['eps_p'])
    assert np.all(np.linalg.norm(s_el - s0, axis=1) <= 1e-4 * np.linalg.norm(s0, axis=1))
    np.testing.assert_allclose(got['eps_e'], got['eps'] - got['eps_p'], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(got['sigma'], ((1.0 - d_h) ** 2)[:, None] * s0,
                               atol=1e-5 * np.abs(s0).max())
    np.testing.assert_allclose(got['H'], 0.5 * np.sum(s0 * got['eps_e'], axis=1), rtol=1e-4)
    np.testing.assert_allclose(got['p_max'][0], got['p'].max(), rtol=1e-6)


def test_isotropic_pure_shear_matches_closed_form():
    n_p = 4
    mu = E / (2.0 * (1.0 + NU))
    gamma = np.array([1.0e-3, 2.0e-3, 4.0e-3, -3.0e-3])    # engineering shear strain
    eps0 = np.column_stack([np.zeros(n_p), np.zeros(n_p), gamma])
    p0 = np.array([0.0, 1e-4, 3e-4, 0.0])
    zeros = np.zeros(n_p)

    got = _run_plast(np.zeros((n_p, 2, 2)), zeros, zeros, eps0, np.zeros((n_p, 3)), p0,
                     zeros, _c_iso(), np.zeros((3, 3)))

    q_tr = np.sqrt(3.0) * mu * np.abs(gamma)
    dp_ref = (q_tr - SIGMA_Y - K_H * p0) / (3.0 * mu + K_H)
    assert np.all(dp_ref > 0.0)
    np.testing.assert_allclose(got['p'] - p0, dp_ref, rtol=1e-4)
    np.testing.assert_allclose(got['sigma0'][:, :2], 0.0, atol=1e-4 * SIGMA_Y)
    np.testing.assert_allclose(got['eps_p'][:, :2], 0.0, atol=1e-10)


def test_elastic_step_below_yield_leaves_plastic_state_and_work_unchanged():
    CA1, M2 = _layer_model()
    n_p = 6
    rng = np.random.default_rng(8)
    eps_p0 = rng.uniform(-1e-4, 1e-4, (n_p, 3))
    eps0 = eps_p0 + rng.uniform(-5e-5, 5e-5, (n_p, 3))      # ~1.5 MPa, well below yield
    p0 = rng.uniform(0.0, 2e-4, n_p)
    alpha_h = rng.uniform(0.0, 1.0, n_p)

    got = _run_plast(rng.uniform(-1e2, 1e2, (n_p, 2, 2)), np.zeros(n_p), alpha_h, eps0,
                     eps_p0, p0, np.zeros(n_p), CA1, M2)

    np.testing.assert_allclose(got['eps_p'], eps_p0, rtol=1e-6)
    np.testing.assert_allclose(got['p'], p0, rtol=1e-6)
    np.testing.assert_allclose(got['w_p'][0], 0.0, atol=1e-12)
    # p_max is written by every particle, yielded or not
    np.testing.assert_allclose(got['p_max'][0], p0.max(), rtol=1e-6)


def test_yielding_step_does_positive_plastic_work():
    CA1, M2 = _layer_model()
    n_p = 8
    eps0 = np.tile(1.0e-3 * LAYER_STRAIN, (n_p, 1))
    L_h = _layer_L(n_p, 1e4)
    nsteps, step = 3, 1

    got = _run_plast(L_h, np.full(n_p, 0.2), np.linspace(0.0, 1.0, n_p), eps0,
                     np.zeros((n_p, 3)), np.zeros(n_p), np.zeros(n_p), CA1, M2,
                     step=step, nsteps=nsteps)

    assert np.all(got['p'] > 0.0)
    w_ref = VOL0 * np.sum(got['sigma'] * got['eps_p'])        # eps_p0 = 0, so d eps_p = eps_p
    assert got['w_p'][step] > 0.0
    np.testing.assert_allclose(got['w_p'][step], w_ref, rtol=1e-4)
    np.testing.assert_allclose(got['w_p'][[0, 2]], 0.0, atol=1e-12)
