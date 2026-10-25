"""cpGIMP partition of unity and zero gradient sum, evaluated on the device by a tiny kernel."""
import numpy as np
import pytest
import warp as wp

from src.warp_gimp import gimp_stencil_origin, gimp_weights_gradients_2d, update_gimp_lp_2d

DX = 1.0e-3            # square background cell, as in the Taylor bar driver
N_GRID = (25, 31)      # cells in x, y; particle domains clamped to [dx, L - dx]


@wp.kernel
def _stencil_sums(x: wp.array(dtype=wp.vec2),
                  lp: wp.array(dtype=wp.vec2),
                  dx: wp.float32,
                  w_sum: wp.array(dtype=wp.float32),
                  w_min: wp.array(dtype=wp.float32),
                  grad_sum: wp.array(dtype=wp.vec2),
                  origin: wp.array(dtype=wp.vec2i)):
    p = wp.tid()
    inv_dx = 1.0 / dx
    s = gimp_weights_gradients_2d(x[p], dx, inv_dx, lp[p][0], lp[p][1])
    ws = wp.float32(0.0)
    wm = wp.float32(1.0e9)
    gs = wp.vec2(0.0, 0.0)
    for i in range(3):
        for j in range(3):
            w = s[2 * i] * s[2 * j + 1]
            ws += w
            wm = wp.min(wm, w)
            gs += wp.vec2(s[6 + 2 * i] * s[2 * j + 1], s[2 * i] * s[6 + 2 * j + 1])
    w_sum[p] = ws
    w_min[p] = wm
    grad_sum[p] = gs
    origin[p] = gimp_stencil_origin(x[p], inv_dx, lp[p])


def _clamped_lp(x, lp0):
    """Run the convected/clamped domain update with F = I."""
    n = len(x)
    F = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33)
    xa = wp.array(np.asarray(x, np.float32), dtype=wp.vec2)
    lp = wp.zeros(n, dtype=wp.vec2)
    wp.launch(update_gimp_lp_2d, dim=n,
              inputs=[F, lp, wp.array(np.asarray(lp0, np.float32), dtype=wp.vec2), xa,
                      DX, (N_GRID[0] - 1) * DX, DX, (N_GRID[1] - 1) * DX])
    return lp.numpy()


# (label, position / dx, half-width / dx)
CASES = [
    ("inside one cell",         (5.50, 7.50), (0.25, 0.25)),
    ("spans boundary in x, y",  (5.10, 7.90), (0.25, 0.30)),
    ("spans boundary in x only", (9.95, 3.50), (0.20, 0.40)),
    ("cell-sized domain",       (12.30, 20.70), (0.50, 0.50)),
    ("anisotropic half-widths", (6.02, 11.45), (0.48, 0.05)),
    ("clamped at lower-left edge", (1.10, 1.15), (0.25, 0.25)),
    ("clamped at upper-right edge", (23.90, 29.85), (0.25, 0.25)),
]


@pytest.fixture(scope="module")
def results():
    wp.init()
    x = np.array([c[1] for c in CASES], dtype=np.float32) * DX
    lp0 = np.array([c[2] for c in CASES], dtype=np.float32) * DX
    lp = _clamped_lp(x, lp0)
    n = len(CASES)
    w_sum = wp.zeros(n, dtype=wp.float32)
    w_min = wp.zeros(n, dtype=wp.float32)
    grad_sum = wp.zeros(n, dtype=wp.vec2)
    origin = wp.zeros(n, dtype=wp.vec2i)
    wp.launch(_stencil_sums, dim=n,
              inputs=[wp.array(x, dtype=wp.vec2), wp.array(lp, dtype=wp.vec2), DX,
                      w_sum, w_min, grad_sum, origin])
    return dict(x=x, lp0=lp0, lp=lp, w_sum=w_sum.numpy(), w_min=w_min.numpy(),
                grad_sum=grad_sum.numpy(), origin=origin.numpy())


@pytest.mark.parametrize("k", range(len(CASES)), ids=[c[0] for c in CASES])
def test_partition_of_unity(results, k):
    assert results["w_sum"][k] == pytest.approx(1.0, abs=1e-5)
    assert results["w_min"][k] >= -1e-6


@pytest.mark.parametrize("k", range(len(CASES)), ids=[c[0] for c in CASES])
def test_gradients_sum_to_zero(results, k):
    assert np.abs(results["grad_sum"][k]).max() <= 1e-5 / DX


@pytest.mark.parametrize("k", range(len(CASES)), ids=[c[0] for c in CASES])
def test_stencil_stays_on_grid(results, k):
    ix, iy = results["origin"][k]
    assert 0 <= ix and ix + 2 <= N_GRID[0]
    assert 0 <= iy and iy + 2 <= N_GRID[1]


def test_clamping_shrinks_domain_inside_grid(results):
    lo, hi = DX, (np.array(N_GRID) - 1) * DX
    for k in (5, 6):
        x, lp, lp0 = results["x"][k], results["lp"][k], results["lp0"][k]
        assert np.all(lp < lp0)
        assert np.all(x - lp >= lo - 1e-9) and np.all(x + lp <= hi + 1e-9)
    # unclamped domains keep their seeded half-widths (F = I)
    np.testing.assert_allclose(results["lp"][:5], results["lp0"][:5], rtol=1e-6)
