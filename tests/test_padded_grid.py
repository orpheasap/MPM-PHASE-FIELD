"""Tests for the padded background grid builder.

These tests assert; they do not print results for a human to read.  Run with

    py tests/test_padded_grid.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.yvonnet_model import DEFAULT_PARAMS, build_model, build_padded_model


PAD = 2


def _params(pad_cells=PAD):
    params = dict(DEFAULT_PARAMS)
    params['pad_cells'] = pad_cells
    return params


def test_returns_everything_the_plain_builder_returns():
    """The padded dict is a superset: plate offset and plate counts on top."""
    plain = build_model(DEFAULT_PARAMS)
    padded = build_padded_model(_params())

    assert set(plain).issubset(set(padded))
    for key in ('plate_numx', 'plate_numy', 'plate_origin', 'pad_cells'):
        assert key in padded, f"missing {key}"


def test_solver_constants_are_identical():
    """Padding changes the grid extent and nothing else the solver depends on.

    A padded run and an unpadded run must march in lockstep, or a comparison
    between them is measuring the pad as well as the basis.
    """
    plain = build_model(DEFAULT_PARAMS)
    padded = build_padded_model(_params())

    for key in ('h_e', 'C_max', 'c_d', 'dt_u', 'eta_d', 'eta_alpha',
                'dt_d_crit', 'dt_alpha_crit', 'dtime'):
        assert padded[key] == plain[key], f"{key} moved: {padded[key]} vs {plain[key]}"

    # The time step is the one that actually has to be bit-identical.
    assert padded['dtime'] == plain['dtime']

    # The material model is unaffected too.
    assert np.array_equal(padded['M2'], plain['M2'])
    assert np.array_equal(padded['C_aniso'](0.3), plain['C_aniso'](0.3))


def test_grid_counts_are_plate_counts_plus_two_pads():
    params = _params()
    padded = build_padded_model(params)
    plain = build_model(DEFAULT_PARAMS)

    assert padded['plate_numx'] == plain['numx']
    assert padded['plate_numy'] == plain['numy']
    assert padded['numx'] == padded['plate_numx'] + 2 * PAD
    assert padded['numy'] == padded['plate_numy'] + 2 * PAD

    # The mesh inside the dict carries the padded counts, not the plate's.
    assert padded['mesh'].numx == padded['numx']
    assert padded['mesh'].numy == padded['numy']


def test_plate_origin_is_pad_times_cell_size():
    params = _params()
    padded = build_padded_model(params)

    assert np.allclose(padded['plate_origin'],
                       [PAD * params['hx'], PAD * params['hy']])

    # Cell size itself is untouched by padding.
    assert np.isclose(padded['mesh'].deltax, params['hx'])
    assert np.isclose(padded['mesh'].deltay, params['hy'])


def test_zero_pad_reproduces_the_unpadded_grid():
    plain = build_model(DEFAULT_PARAMS)
    padded = build_padded_model(_params(pad_cells=0))

    assert padded['numx'] == plain['numx']
    assert padded['numy'] == plain['numy']
    assert padded['mesh'].nodeCount == plain['mesh'].nodeCount
    assert padded['mesh'].elemCount == plain['mesh'].elemCount
    assert np.allclose(padded['mesh'].node, plain['mesh'].node)
    assert np.array_equal(padded['mesh'].element, plain['mesh'].element)
    assert np.allclose(padded['plate_origin'], [0.0, 0.0])


def test_pad_ring_is_outside_the_plate_on_all_four_sides():
    """The plate, placed at the offset, leaves a pad's worth of grid all round."""
    params = _params()
    padded = build_padded_model(params)
    ox, oy = padded['plate_origin']
    node = padded['mesh'].node

    assert np.isclose(ox, PAD * params['hx'])
    assert np.isclose(node[:, 0].max(), ox + params['Lx'] + PAD * params['hx'])
    assert np.isclose(node[:, 1].max(), oy + params['Ly'] + PAD * params['hy'])
    assert np.isclose(node[:, 0].min(), 0.0)
    assert np.isclose(node[:, 1].min(), 0.0)


def test_existing_builder_is_untouched_by_the_padded_one():
    """build_padded_model must not mutate the caller's params dict."""
    params = _params()
    before = dict(params)
    build_padded_model(params)

    assert params == before


if __name__ == "__main__":
    test_returns_everything_the_plain_builder_returns()
    test_solver_constants_are_identical()
    test_grid_counts_are_plate_counts_plus_two_pads()
    test_plate_origin_is_pad_times_cell_size()
    test_zero_pad_reproduces_the_unpadded_grid()
    test_pad_ring_is_outside_the_plate_on_all_four_sides()
    test_existing_builder_is_untouched_by_the_padded_one()
    print("All padded grid tests passed.")
