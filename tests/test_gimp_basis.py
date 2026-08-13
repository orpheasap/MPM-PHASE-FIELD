"""Tests for the uGIMP shape function and the nine-cell connectivity builder.

These tests assert; they do not print results for a human to read.  Run with

    py tests/test_gimp_basis.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.mesh2D import Mesh
from src.solver_utilities import (
    get_mpm2d_shape,
    get_gimp2d_shape,
    build_gimp_element_map,
    _gimp_neighbour_cells,
)


# =============================================================================
#  Shape function -- the four branches
# =============================================================================
#  Every branch is exercised in the x direction with the y offset held at zero,
#  where the y factor is the constant  Sy = 1 - lpy/(2 hy)  with zero gradient,
#  so the x branch under test is the only thing that moves.

HX, HY = 2.0, 3.0
LPX, LPY = 0.4, 0.5
SY = 1.0 - LPY / (2.0 * HY)          # y factor at zero y offset


def test_branch_inner():
    """|x| < lpx:  the quadratic cap straddling the node."""
    N, dNdx = get_gimp2d_shape(np.array([0.2, 0.0]), HX, HY, LPX, LPY)

    Sx = 1.0 - (0.2 ** 2 + LPX ** 2) / (2.0 * HX * LPX)
    dSx = -0.2 / (HX * LPX)
    assert np.isclose(N, Sx * SY)
    assert np.isclose(dNdx[0], dSx * SY)
    assert np.isclose(dNdx[1], 0.0)


def test_branch_linear():
    """lpx <= |x| < hx - lpx:  the branch that coincides with the hat function."""
    N, dNdx = get_gimp2d_shape(np.array([1.0, 0.0]), HX, HY, LPX, LPY)

    Sx = 1.0 - 1.0 / HX
    dSx = -1.0 / HX
    assert np.isclose(N, Sx * SY)
    assert np.isclose(dNdx[0], dSx * SY)

    # ... and it is genuinely linear: mirrored offset, mirrored gradient.
    Nm, dNdxm = get_gimp2d_shape(np.array([-1.0, 0.0]), HX, HY, LPX, LPY)
    assert np.isclose(Nm, N)
    assert np.isclose(dNdxm[0], -dNdx[0])


def test_branch_outer():
    """hx - lpx <= |x| < hx + lpx:  the quadratic tail reaching past one cell."""
    N, dNdx = get_gimp2d_shape(np.array([2.0, 0.0]), HX, HY, LPX, LPY)

    Sx = (HX + LPX - 2.0) ** 2 / (4.0 * HX * LPX)
    dSx = -(HX + LPX - 2.0) / (2.0 * HX * LPX)
    assert np.isclose(N, Sx * SY)
    assert np.isclose(dNdx[0], dSx * SY)

    # The hat function is already dead this far out; uGIMP is not.
    assert N > 0.0
    assert get_mpm2d_shape(np.array([2.0, 0.0]), HX, HY)[0] == 0.0


def test_zero_outside_support():
    """Outside the support both the weight and the gradient are exactly zero."""
    outside = [
        [HX + LPX, 0.0],             # exactly at the x support edge
        [HX + LPX + 1e-9, 0.0],      # just past it
        [-(HX + LPX), 0.0],
        [0.0, HY + LPY],             # y support edge
        [0.0, -(HY + LPY + 1.0)],
        [HX + LPX, HY + LPY],        # past both at once
    ]
    for x in outside:
        N, dNdx = get_gimp2d_shape(np.array(x), HX, HY, LPX, LPY)
        assert N == 0.0, f"nonzero weight at offset {x}"
        assert np.all(dNdx == 0.0), f"nonzero gradient at offset {x}"


def test_matches_hat_at_cell_centre():
    """With one particle per cell the two bases agree in the undeformed state.

    A particle at a cell centre sits at offset (+-hx/2, +-hy/2) from each of its
    four cell nodes, and the particle domain half-width is half a cell.  That
    offset lands in the outer branch, where the uGIMP weight collapses onto the
    hat weight.  The bases only separate once the particle convects off centre.
    """
    lpx, lpy = HX / 2.0, HY / 2.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = np.array([sx * HX / 2.0, sy * HY / 2.0])
            N_gimp, dN_gimp = get_gimp2d_shape(x, HX, HY, lpx, lpy)
            N_hat, dN_hat = get_mpm2d_shape(x, HX, HY)
            assert np.isclose(N_gimp, N_hat)
            assert np.allclose(dN_gimp, dN_hat)

    # Off centre the two must disagree, or the test above proves nothing.
    x = np.array([0.1 * HX, 0.1 * HY])
    assert not np.isclose(get_gimp2d_shape(x, HX, HY, lpx, lpy)[0],
                          get_mpm2d_shape(x, HX, HY)[0])


# =============================================================================
#  Neighbour lookup and connectivity
# =============================================================================

def _mesh(numx=5, numy=4):
    return Mesh(numx * 1.0, numy * 2.0, numx, numy)


def test_neighbour_cell_counts():
    """4 cells at a corner, 6 along an edge, 9 in the interior."""
    mesh = _mesh()
    nx, ny = mesh.numx, mesh.numy

    corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
    for ix, iy in corners:
        assert len(_gimp_neighbour_cells(ix, iy, nx, ny)) == 4, f"corner {(ix, iy)}"

    edges = [(2, 0), (2, ny - 1), (0, 2), (nx - 1, 2)]
    for ix, iy in edges:
        assert len(_gimp_neighbour_cells(ix, iy, nx, ny)) == 6, f"edge {(ix, iy)}"

    assert len(_gimp_neighbour_cells(2, 2, nx, ny)) == 9

    # Every returned cell is a real cell, and the centre cell is among them.
    cells = _gimp_neighbour_cells(2, 2, nx, ny)
    assert len(set(cells)) == len(cells)
    assert all(0 <= e < mesh.elemCount for e in cells)
    assert (2 + 2 * nx) in cells


def test_element_map_shape_and_indices():
    """One entry per cell, sorted, no repeats, no index off the end of the grid."""
    mesh = _mesh()
    emap = build_gimp_element_map(mesh)

    assert len(emap) == mesh.elemCount
    for e, nodes in enumerate(emap):
        assert nodes.ndim == 1
        assert len(set(nodes.tolist())) == len(nodes), f"repeated node in cell {e}"
        assert np.all(np.diff(nodes) > 0), f"cell {e} not sorted"
        assert nodes.min() >= 0 and nodes.max() < mesh.nodeCount, f"cell {e} out of range"
        # Every node of the cell itself is reachable from it.
        assert set(mesh.element[e].tolist()) <= set(nodes.tolist())


def test_element_map_counts_at_border_and_interior():
    """A 3x3 block of cells spans 4x4 nodes; a 2x2 block at a corner spans 3x3."""
    mesh = _mesh()
    emap = build_gimp_element_map(mesh)
    nx = mesh.numx

    assert len(emap[2 + 2 * nx]) == 16        # interior: 9 cells
    assert len(emap[0]) == 9                  # corner:   4 cells
    assert len(emap[2]) == 12                 # bottom edge: 6 cells
    assert len(emap[mesh.elemCount - 1]) == 9  # far corner


def test_partition_of_unity_in_interior():
    """Weights sum to one and gradients to zero over an interior particle's nodes."""
    mesh = _mesh()
    emap = build_gimp_element_map(mesh)
    hx, hy = mesh.deltax, mesh.deltay
    lpx, lpy = hx / 2.0, hy / 2.0

    # Cell centre, and two off-centre positions inside the same interior cell.
    ix, iy = 2, 2
    base = np.array([ix * hx, iy * hy])
    for shift in ([0.5, 0.5], [0.23, 0.5], [0.77, 0.31]):
        xp = base + np.array([shift[0] * hx, shift[1] * hy])
        nodes = emap[ix + iy * mesh.numx]

        total_N = 0.0
        total_dN = np.zeros(2)
        for idn in nodes:
            N, dNdx = get_gimp2d_shape(xp - mesh.node[idn, :], hx, hy, lpx, lpy)
            total_N += N
            total_dN += dNdx

        assert np.isclose(total_N, 1.0), f"weights sum to {total_N} at {xp}"
        assert np.allclose(total_dN, 0.0, atol=1e-12), f"gradients sum to {total_dN} at {xp}"


if __name__ == "__main__":
    test_branch_inner()
    test_branch_linear()
    test_branch_outer()
    test_zero_outside_support()
    test_matches_hat_at_cell_centre()
    test_neighbour_cell_counts()
    test_element_map_shape_and_indices()
    test_element_map_counts_at_border_and_interior()
    test_partition_of_unity_in_interior()
    print("All uGIMP basis tests passed.")
