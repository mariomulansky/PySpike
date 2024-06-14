""" test_order.py

Tests the order functions

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal

import pyspike as spk
from pyspike import SpikeTrain, DiscreteFunc
from pyspike.spike_order import spike_order_matrix
from pyspike.plotting import Multi_Profile

def test_spike_order():

    st1 = SpikeTrain([100, 200, 300], [0, 1000])
    st2 = SpikeTrain([105, 205, 300], [0, 1000])

    result = spk.spike_order_values(st1, st2)
    result_expected = [np.array([1, 1, 0]), np.array([-1, -1, 0])]
    assert_array_equal(result, result_expected)

    st3 = SpikeTrain([105, 195, 500], [0, 1000])

    result = spk.spike_order_values(st1, st3)
    result_expected = [np.array([1, -1, 0]), np.array([-1, 1, 0])]
    assert_array_equal(result, result_expected)

    result = spk.spike_order_values(st1, st2, st3)
    result_expected = [np.array([1, 0, 0]), np.array([-0.5, -1, 0]), np.array([-0.5, 1, 0])]
    assert_array_equal(result, result_expected)

    result = spike_order_matrix(st1, st2, st3)
    result_expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert_array_equal(result, result_expected)

    result = spike_order_matrix(st1, st2, st3, verification=True)
    result_expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert_array_equal(result, result_expected)

def test_spike_train_order():

    st1 = SpikeTrain([100, 200, 300], [0, 1000])
    st2 = SpikeTrain([105, 205, 300], [0, 1000])
    st3 = SpikeTrain([105, 195, 500], [0, 1000])

    result = spk.spike_train_order(st1, st2)
    result_expected = 0.6666
    assert_almost_equal(result, result_expected, decimal=4)
    result = spk.spike_train_order(st1, st2, normalize=False)
    result_expected = 2.0
    assert_almost_equal(result, result_expected, decimal=4)

    result = spk.spike_train_order(st1, st3)
    result_expected = 0.0
    assert_almost_equal(result, result_expected, decimal=4)
    result = spk.spike_train_order(st1, st3, normalize=False)
    result_expected = 0.0
    assert_almost_equal(result, result_expected, decimal=4)

    result = spk.spike_train_order_value(st1, st2, st3)
    result_expected_synf = 0.1111
    assert_almost_equal(result, result_expected_synf, decimal=4)
    result = spk.spike_train_order_value([st1, st2, st3], normalize=False) ##### how can we check if it's correct ? 
                                                                           ##### I changed the initial function ...
    result_expected = 2
    assert_almost_equal(result, result_expected, decimal=4)

    result_prof = Multi_Profile([st1, st2, st3], 3)
    result_expected = np.array([[100, 105, 105, 195, 200, 205, 300, 300, 500], [1, 0.5, 0.5, -1, 0, 0, 0, 0, 0]])
    assert_array_equal(result_prof, result_expected)

    result = spk.spike_train_order_matrix([st1, st2, st3])
    result_expected = np.array([[0, 0.6667, 0], [-0.6667, 0, -0.3333], [0, 0.3333, 0]])
    assert_array_almost_equal(result, result_expected, decimal=4)
    result = spk.spike_train_order_matrix([st1, st2, st3], normalize=False)
    result_expected = np.array([[0, 2, 0], [-2, 0, -1], [0, 1, 0]])
    assert_array_almost_equal(result, result_expected, decimal=4)

    # Averaging the profile should be the same as computing the synfire indicator directly.
    # We can also compute the synfire indicator from the Order Matrix:
    num_spikes = sum(len(st) for st in [st1, st2, st3])
    syn_fire = np.sum(np.triu(result)) / num_spikes
    avg_prof = np.sum(result_prof[1])/len(result_prof[1])
    assert_almost_equal(avg_prof, syn_fire, decimal=4)
    assert_almost_equal(result_expected_synf, syn_fire, decimal=4)
