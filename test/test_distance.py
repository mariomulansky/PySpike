""" test_distance.py

Tests the isi- and spike-distance computation

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
import numpy as np
from copy import copy
from numpy.testing import assert_equal, assert_array_almost_equal

import pyspike as spk


def test_isi():
    # generate two spike trains:
    t1 = np.array([0.2, 0.4, 0.6, 0.7])
    t2 = np.array([0.3, 0.45, 0.8, 0.9, 0.95])

    # pen&paper calculation of the isi distance
    expected_times = [0.0, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    expected_isi = [-0.1/0.3, -0.1/0.3, 0.05/0.2, 0.05/0.2, -0.15/0.35,
                    -0.25/0.35, -0.05/0.35, 0.2/0.3, 0.25/0.3, 0.25/0.3]
    
    t1 = spk.add_auxiliary_spikes(t1, 1.0)
    t2 = spk.add_auxiliary_spikes(t2, 1.0)
    f = spk.isi_distance(t1, t2)

    # print("ISI: ", f.y)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y, expected_isi, decimal=14)

    # check with some equal spike times
    t1 = np.array([0.2, 0.4, 0.6])
    t2 = np.array([0.1, 0.4, 0.5, 0.6])

    expected_times = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 1.0]
    expected_isi = [0.1/0.2, -0.1/0.3, -0.1/0.3, 0.1/0.2, 0.1/0.2, -0.0/0.5]

    t1 = spk.add_auxiliary_spikes(t1, 1.0)
    t2 = spk.add_auxiliary_spikes(t2, 1.0)
    f = spk.isi_distance(t1, t2)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y, expected_isi, decimal=14)


def test_spike():
    # generate two spike trains:
    t1 = np.array([0.2, 0.4, 0.6, 0.7])
    t2 = np.array([0.3, 0.45, 0.8, 0.9, 0.95])

    # pen&paper calculation of the spike distance
    expected_times = [0.0, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    s1 = np.array([0.1, 0.1, (0.1*0.1+0.05*0.1)/0.2, 0.05, (0.05*0.15 * 2)/0.2,
                   0.15, 0.1, 0.1*0.2/0.3, 0.1**2/0.3, 0.1*0.05/0.3, 0.1])
    s2 = np.array([0.1, 0.1*0.2/0.3, 0.1, (0.1*0.05 * 2)/.15, 0.05,
                   (0.05*0.2+0.1*0.15)/0.35, (0.05*0.1+0.1*0.25)/0.35,
                   0.1, 0.1, 0.05, 0.05])
    isi1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.3, 0.3, 0.3, 0.3])
    isi2 = np.array([0.3, 0.3, 0.15, 0.15, 0.35, 0.35, 0.35, 0.1, 0.05, 0.05])
    expected_y1 = (s1[:-1]*isi2+s2[:-1]*isi1) / (0.5*(isi1+isi2)**2)
    expected_y2 = (s1[1:]*isi2+s2[1:]*isi1) / (0.5*(isi1+isi2)**2)

    t1 = spk.add_auxiliary_spikes(t1, 1.0)
    t2 = spk.add_auxiliary_spikes(t2, 1.0)
    f = spk.spike_distance(t1, t2)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y1, expected_y1, decimal=14)
    assert_array_almost_equal(f.y2, expected_y2, decimal=14)

    # check with some equal spike times
    t1 = np.array([0.2, 0.4, 0.6])
    t2 = np.array([0.1, 0.4, 0.5, 0.6])

    expected_times = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 1.0]
    s1 = np.array([0.1, 0.1*0.1/0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    s2 = np.array([0.1*0.1/0.3, 0.1, 0.1*0.2/0.3, 0.0, 0.1, 0.0, 0.0])
    isi1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.4])
    isi2 = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.4])
    expected_y1 = (s1[:-1]*isi2+s2[:-1]*isi1) / (0.5*(isi1+isi2)**2)
    expected_y2 = (s1[1:]*isi2+s2[1:]*isi1) / (0.5*(isi1+isi2)**2)

    t1 = spk.add_auxiliary_spikes(t1, 1.0)
    t2 = spk.add_auxiliary_spikes(t2, 1.0)
    f = spk.spike_distance(t1, t2)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y1, expected_y1, decimal=14)
    assert_array_almost_equal(f.y2, expected_y2, decimal=14)


def check_multi_distance(dist_func, dist_func_multi):
    # generate spike trains:
    t1 = spk.add_auxiliary_spikes(np.array([0.2, 0.4, 0.6, 0.7]), 1.0)
    t2 = spk.add_auxiliary_spikes(np.array([0.3, 0.45, 0.8, 0.9, 0.95]), 1.0)
    t3 = spk.add_auxiliary_spikes(np.array([0.2, 0.4, 0.6]), 1.0)
    t4 = spk.add_auxiliary_spikes(np.array([0.1, 0.4, 0.5, 0.6]), 1.0)
    spike_trains = [t1, t2, t3, t4]

    f12 = dist_func(t1, t2)
    f13 = dist_func(t1, t3)
    f14 = dist_func(t1, t4)
    f23 = dist_func(t2, t3)
    f24 = dist_func(t2, t4)
    f34 = dist_func(t3, t4)

    f_multi = dist_func_multi(spike_trains, [0, 1])
    assert f_multi.almost_equal(f12, decimal=14)

    f = copy(f12)
    f.add(f13)
    f.add(f23)
    f.mul_scalar(1.0/3)
    f_multi = dist_func_multi(spike_trains, [0, 1, 2])
    assert f_multi.almost_equal(f, decimal=14)

    f.mul_scalar(3)  # revert above normalization
    f.add(f14)
    f.add(f24)
    f.add(f34)
    f.mul_scalar(1.0/6)
    f_multi = dist_func_multi(spike_trains)
    assert f_multi.almost_equal(f, decimal=14)


def test_multi_isi():
    check_multi_distance(spk.isi_distance, spk.isi_distance_multi)


def test_multi_spike():
    check_multi_distance(spk.spike_distance, spk.spike_distance_multi)


if __name__ == "__main__":
    test_isi()
    test_spike()
    test_multi_isi()
    test_multi_spike()
