""" test_spike_delay_asymmetry.py

Tests the asymmetry functions

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, \
    assert_array_equal

import pyspike as spk
from pyspike import SpikeTrain, DiscreteFunc


def test_profile():
    st1 = SpikeTrain([100, 200, 300], [0, 1000])
    st2 = SpikeTrain([105, 205, 300], [0, 1000])
    expected_x = np.array([0, 100, 105, 200, 205, 300, 1000])
    expected_y = np.array([1, 1, 1, 1, 1, 0, 0])
    expected_mp = np.array([1, 1, 1, 1, 1, 2, 2])

    f = spk.drct.spike_train_order_profile(st1, st2)

    assert f.almost_equal(DiscreteFunc(expected_x, expected_y, expected_mp))
    assert_almost_equal(f.avrg(), 2.0/3.0)
    assert_almost_equal(spk.drct.spike_train_order(st1, st2), 2.0/3.0)
    assert_almost_equal(spk.drct.spike_train_order(st1, st2, normalize=False),
                        4.0)

    st3 = SpikeTrain([105, 195, 500], [0, 1000])
    expected_x = np.array([0, 100, 105, 195, 200, 300, 500, 1000])
    expected_y = np.array([1, 1, 1, -1, -1, 0, 0, 0])
    expected_mp = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    f = spk.drct.spike_train_order_profile(st1, st3)
    assert f.almost_equal(DiscreteFunc(expected_x, expected_y, expected_mp))
