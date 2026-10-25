"""Integration test: uGIMP kernel + nine-cell connectivity + padded grid, composed
over the real plate layout the uGIMP driver uses.

Each of the three pieces can be correct on its own while the composition still
leaks weight off the end of the grid at the loaded surfaces, or lets a particle
support reach across the pre-notch.  This reconstructs the driver's setup --
padded model, one particle per cell at cell centres, translated by the plate
origin, four-row notch removed -- and asserts the two properties the design
turns on:

  1. partition of unity, which fails if the pad is too narrow, the connectivity
     is wrong or the kernel is wrong;
  2. notch separation, which fails if the notch is too narrow for the support.

These are the two runtime diagnostics deliberately kept out of the driver.  Run with

    py tests/test_gimp_integration.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.yvonnet_model import DEFAULT_PARAMS, build_padded_model
from src.solver_utilities import (
    get_gimp2d_shape,
    build_gimp_element_map,
    build_particle_element_map,
)


PAD_CELLS = 2        # pinned here and in the driver
NOTCH_ROWS = 4       # plate-local cell rows removed, straddling the midline

# Convection the layout must tolerate, in cells.  Both the pad width and the
# notch width exist to buy this margin, and at zero drift neither can be shown
# to be doing anything -- see the two "fails" tests below.
DRIFT = 0.45


def build_layout(pad_cells=PAD_CELLS, notch_rows=NOTCH_ROWS, shift=(0.0, 0.0)):
    """Reconstruct the driver's particle layout on the padded grid.

    Mirrors what the uGIMP driver does: seed one particle per plate cell at the
    cell centre, remove the notch rows, translate into grid coordinates.

    `shift` displaces every particle by that many cells, standing in for the
    convection of a running simulation.  It stays below half a cell, so each
    particle remains in the cell it was seeded in and the connectivity entry
    still covers its support.
    """
    params = dict(DEFAULT_PARAMS)
    params['pad_cells'] = pad_cells
    model = build_padded_model(params)

    mesh = model['mesh']
    hx, hy = mesh.deltax, mesh.deltay
    numx, numy = model['plate_numx'], model['plate_numy']
    ox, oy = model['plate_origin']
    notch_length = params['notch_length']

    # Notch rows straddle the plate midline, so the void is centred exactly on
    # the crack line whatever the row count.
    row_lo = numy // 2 - notch_rows // 2
    row_hi = row_lo + notch_rows - 1
    y_crack = 0.5 * numy * hy

    positions = []
    plate_y = []
    for iy in range(numy):
        for ix in range(numx):
            xp = (ix + 0.5) * hx
            yp = (iy + 0.5) * hy
            if xp <= notch_length and row_lo <= iy <= row_hi:
                continue                       # inside the notch: no material
            positions.append([xp + ox + shift[0] * hx,     # grid coordinates
                              yp + oy + shift[1] * hy])
            plate_y.append(yp)

    positions = np.array(positions)
    plate_y = np.array(plate_y)
    plate_x = positions[:, 0] - ox

    # "Either side of the notch" only means anything within the notch span; past
    # the tip the material is continuous and shares nodes by design.
    in_span = plate_x <= notch_length

    return dict(
        mesh=mesh,
        positions=positions,
        lpx=hx / 2.0,                     # one particle per cell
        lpy=hy / 2.0,
        emap=build_gimp_element_map(mesh),
        pElems=build_particle_element_map(positions, mesh)[0],
        below=in_span & (plate_y < y_crack),
        above=in_span & (plate_y > y_crack),
        y_crack=y_crack + oy,             # in grid coordinates
        numy=numy,
    )


def partition_residuals(layout):
    """Return (max |sum N - 1|, max |sum grad N| * h) over every particle."""
    mesh = layout['mesh']
    hx, hy = mesh.deltax, mesh.deltay
    lpx, lpy = layout['lpx'], layout['lpy']

    worst_N = 0.0
    worst_dN = 0.0
    for pid in range(len(layout['positions'])):
        total_N = 0.0
        total_dN = np.zeros(2)
        for idn in layout['emap'][layout['pElems'][pid]]:
            N, dNdx = get_gimp2d_shape(layout['positions'][pid] - mesh.node[idn],
                                       hx, hy, lpx, lpy)
            if N == 0.0:
                continue
            total_N += N
            total_dN += dNdx
        worst_N = max(worst_N, abs(total_N - 1.0))
        worst_dN = max(worst_dN, np.max(np.abs(total_dN)) * min(hx, hy))

    return worst_N, worst_dN


def reached_nodes(layout, mask, drift=0.0):
    """Node indices reached by the masked particles, optionally after drifting
    each particle up to `drift` cells in any direction."""
    mesh = layout['mesh']
    hx, hy = mesh.deltax, mesh.deltay
    reach_x = hx + layout['lpx'] + drift * hx
    reach_y = hy + layout['lpy'] + drift * hy
    nnx = mesh.numx + 1

    nodes = set()
    for pid in np.flatnonzero(mask):
        xp, yp = layout['positions'][pid]
        if drift == 0.0:
            # Exactly what the driver would deposit: the connectivity entry,
            # filtered by a nonzero weight.
            for idn in layout['emap'][layout['pElems'][pid]]:
                N, _ = get_gimp2d_shape(layout['positions'][pid] - mesh.node[idn],
                                        hx, hy, layout['lpx'], layout['lpy'])
                if N != 0.0:
                    nodes.add(int(idn))
        else:
            ix0 = int(np.floor((xp - reach_x) / hx))
            iy0 = int(np.floor((yp - reach_y) / hy))
            for iy in range(iy0, iy0 + 2 * int(np.ceil(reach_y / hy)) + 2):
                if iy < 0 or iy > mesh.numy:
                    continue
                if abs(yp - iy * hy) >= reach_y:
                    continue
                for ix in range(ix0, ix0 + 2 * int(np.ceil(reach_x / hx)) + 2):
                    if ix < 0 or ix > mesh.numx:
                        continue
                    if abs(xp - ix * hx) < reach_x:
                        nodes.add(ix + iy * nnx)

    return nodes


# =============================================================================
#  The two properties
# =============================================================================

def test_partition_of_unity_over_the_whole_plate():
    """Every particle's weights sum to one -- including the loaded surfaces.

    The top and bottom rows are where the traction is applied, so weight lost
    off the end of the grid there is load that never reaches the solid.

    Checked at the seeded positions and after convecting the whole plate by
    just under half a cell, each way.  The drifted cases are the ones with
    teeth: at exact cell centres uGIMP collapses onto the hat function, whose
    support is a single cell, so a seeded layout sums to one whether or not the
    pad ring is there.
    """
    for shift in ([0.0, 0.0], [DRIFT, DRIFT], [-DRIFT, -DRIFT]):
        layout = build_layout(shift=shift)
        worst_N, worst_dN = partition_residuals(layout)

        assert worst_N < 1e-12, f"shift {shift}: weights sum to 1 +- {worst_N}"
        assert worst_dN < 1e-12, f"shift {shift}: gradients sum to {worst_dN}"


def test_partition_of_unity_fails_without_the_pad():
    """The check above genuinely detects leakage rather than passing vacuously.

    With no pad ring, a surface particle that has convected outward reaches past
    the end of the grid and its weight is silently dropped.  This must show up
    as a failed sum, not as an index error -- the connectivity is ragged, so
    nothing crashes; the mass just quietly goes missing.
    """
    worst = 0.0
    for shift in ([DRIFT, DRIFT], [-DRIFT, -DRIFT]):
        layout = build_layout(pad_cells=0, shift=shift)
        worst = max(worst, partition_residuals(layout)[0])

    assert worst > 1e-3, ("an unpadded grid leaked no weight, so the padded "
                          f"result proves nothing (worst deviation {worst})")


def test_notch_separation():
    """No node receives weight from particles on both faces of the notch.

    A shared node would let the crack faces exchange momentum and diffuse damage
    across a gap that is supposed to be a free surface.
    """
    layout = build_layout()
    shared = reached_nodes(layout, layout['below']) & reached_nodes(layout, layout['above'])

    assert not shared, f"{len(shared)} nodes bridge the notch: {sorted(shared)[:8]}"


def test_notch_separation_has_a_margin():
    """The four-row notch keeps the faces apart under half a cell of convection.

    At zero drift a two-row notch also passes: its nearest particles sit one and
    a half cells from the crack line, exactly the support half-width, so the two
    faces touch with a weight of exactly zero.  That is a knife edge -- the
    moment particles convect toward each other the supports overlap and the
    notch heals.  The four-row notch is what buys real clearance, and this is
    the check that tells the two apart.
    """
    layout = build_layout()
    shared = (reached_nodes(layout, layout['below'], drift=DRIFT)
              & reached_nodes(layout, layout['above'], drift=DRIFT))

    assert not shared, f"{len(shared)} nodes bridge the notch under drift"


def test_notch_separation_fails_with_two_rows():
    """The margin check above genuinely detects bridging."""
    layout = build_layout(notch_rows=2)
    shared = (reached_nodes(layout, layout['below'], drift=DRIFT)
              & reached_nodes(layout, layout['above'], drift=DRIFT))

    assert shared, ("a two-row notch bridged nothing, so the four-row result "
                    "proves nothing")


if __name__ == "__main__":
    test_partition_of_unity_over_the_whole_plate()
    test_partition_of_unity_fails_without_the_pad()
    test_notch_separation()
    test_notch_separation_has_a_margin()
    test_notch_separation_fails_with_two_rows()
    print("All uGIMP integration tests passed.")
