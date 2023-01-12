""" test_directionality.py

Tests the directionality functions

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, \
    assert_array_equal

import pyspike as spk
from pyspike import SpikeTrain, DiscreteFunc

def test_spike_directionality():
    
    st1 = SpikeTrain([100, 200, 300], [0, 1000])
    st2 = SpikeTrain([105, 205, 300], [0, 1000])
    assert_almost_equal(spk.spike_directionality(st1, st2), 2.0/3.0)
    assert_almost_equal(spk.spike_directionality(st1, st2, normalize=False),
                        2.0)

    # exchange order of spike trains should give exact negative profile
    assert_almost_equal(spk.spike_directionality(st2, st1), -2.0/3.0)
    assert_almost_equal(spk.spike_directionality(st2, st1, normalize=False),
                        -2.0)

    st3 = SpikeTrain([105, 195, 500], [0, 1000])
    assert_almost_equal(spk.spike_directionality(st1, st3), 0.0)
    assert_almost_equal(spk.spike_directionality(st1, st3, normalize=False),
                        0.0)
    assert_almost_equal(spk.spike_directionality(st3, st1), 0.0)

    D = spk.spike_directionality_matrix([st1, st2, st3], normalize=False)
    D_expected = np.array([[0, 2.0, 0.0], [-2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert_array_equal(D, D_expected)

    dir_profs = spk.spike_directionality_values([st1, st2, st3])
    assert_array_equal(dir_profs[0], [1.0, 0.0, 0.0])
    assert_array_equal(dir_profs[1], [-0.5, -1.0, 0.0])


def test_spike_train_order():
    st1 = SpikeTrain([100, 200, 300], [0, 1000])
    st2 = SpikeTrain([105, 205, 300], [0, 1000])
    st3 = SpikeTrain([105, 195, 500], [0, 1000])

    expected_x12 = np.array([0, 100, 105, 200, 205, 300, 1000])
    expected_y12 = np.array([1, 1, 1, 1, 1, 0, 0])
    expected_mp12 = np.array([1, 1, 1, 1, 1, 2, 2])

    f = spk.spike_train_order_profile(st1, st2)

    assert f.almost_equal(DiscreteFunc(expected_x12, expected_y12,
                                       expected_mp12))
    assert_almost_equal(f.avrg(), 2.0/3.0)
    assert_almost_equal(f.avrg(normalize=False), 4.0)
    assert_almost_equal(spk.spike_train_order(st1, st2), 2.0/3.0)
    assert_almost_equal(spk.spike_train_order(st1, st2, normalize=False), 4.0)

    expected_x23 = np.array([0, 105, 195, 205, 300, 500, 1000])
    expected_y23 = np.array([0, 0, -1, -1, 0, 0, 0])
    expected_mp23 = np.array([2, 2, 1, 1, 1, 1, 1])

    f = spk.spike_train_order_profile(st2, st3)

    assert_array_equal(f.x, expected_x23)
    assert_array_equal(f.y, expected_y23)
    assert_array_equal(f.mp, expected_mp23)
    assert f.almost_equal(DiscreteFunc(expected_x23, expected_y23,
                                       expected_mp23))
    assert_almost_equal(f.avrg(), -1.0/3.0)
    assert_almost_equal(f.avrg(normalize=False), -2.0)
    assert_almost_equal(spk.spike_train_order(st2, st3), -1.0/3.0)
    assert_almost_equal(spk.spike_train_order(st2, st3, normalize=False), -2.0)

    f = spk.spike_train_order_profile_multi([st1, st2, st3])

    expected_x = np.array([0, 100, 105, 195, 200, 205, 300, 500, 1000])
    expected_y = np.array([2, 2, 2, -2, 0, 0, 0, 0, 0])
    expected_mp = np.array([2, 2, 4, 2, 2, 2, 4, 2, 2])

    assert_array_equal(f.x, expected_x)
    assert_array_equal(f.y, expected_y)
    assert_array_equal(f.mp, expected_mp)

    # Averaging the profile should be the same as computing the synfire indicator directly.
    assert_almost_equal(f.avrg(), spk.spike_train_order([st1, st2, st3]))

    # We can also compute the synfire indicator from the Directionality Matrix:
    D_matrix = spk.spike_directionality_matrix([st1, st2, st3], normalize=False)
    num_spikes = sum(len(st) for st in [st1, st2, st3])
    syn_fire = np.sum(np.triu(D_matrix)) / num_spikes
    assert_almost_equal(f.avrg(), syn_fire)
