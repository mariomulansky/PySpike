""" test_distance.py

Tests the isi- and spike-distance computation

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

import pyspike as spk

def test_isi():
    # generate two spike trains:
    t1 = np.array([0.2, 0.4, 0.6, 0.7])
    t2 = np.array([0.3, 0.45, 0.8, 0.9, 0.95])

    # pen&paper calculation of the isi distance
    expected_times = [0.0,0.2,0.3,0.4,0.45,0.6,0.7,0.8,0.9,0.95,1.0]
    expected_isi = [-0.1/0.3, -0.1/0.3, 0.05/0.2, 0.05/0.2, -0.15/0.35, 
                    -0.25/0.35, -0.05/0.35, 0.2/0.3, 0.25/0.3, 0.25/0.3]
    
    f = spk.isi_distance(t1, t2, 1.0)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y, expected_isi, decimal=14)

    # check with some equal spike times
    t1 = np.array([0.2,0.4,0.6])
    t2 = np.array([0.1,0.4,0.5,0.6])

    expected_times = [0.0,0.1,0.2,0.4,0.5,0.6,1.0]
    expected_isi = [0.1/0.2, -0.1/0.3, -0.1/0.3, 0.1/0.2, 0.1/0.2, -0.0/0.5]

    f = spk.isi_distance(t1, t2, 1.0)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y, expected_isi, decimal=14)


def test_spike():
    # generate two spike trains:
    t1 = np.array([0.2, 0.4, 0.6, 0.7])
    t2 = np.array([0.3, 0.45, 0.8, 0.9, 0.95])

    # pen&paper calculation of the spike distance
    expected_times = [0.0,0.2,0.3,0.4,0.45,0.6,0.7,0.8,0.9,0.95,1.0]
    s1 = np.array([0.0, 0.1, (0.1*0.1+0.05*0.1)/0.2, 0.05, (0.05*0.15 * 2)/0.2,
          0.15, 0.1, 0.1*0.2/0.3, 0.1**2/0.3, 0.1*0.05/0.3, 0.0])
    s2 = np.array([0.0, 0.1*0.2/0.3, 0.1, (0.1*0.05 * 2)/.15, 0.05, 
          (0.05*0.2+0.1*0.15)/0.35, (0.05*0.1+0.1*0.25)/0.35, 0.1,0.1,0.05,0.0])
    isi1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.3, 0.3, 0.3, 0.3])
    isi2 = np.array([0.3, 0.3, 0.15, 0.15, 0.35, 0.35, 0.35, 0.1, 0.05, 0.05])
    expected_y1 = (s1[:-1]*isi2+s2[:-1]*isi1) / (0.5*(isi1+isi2)**2)
    expected_y2 = (s1[1:]*isi2+s2[1:]*isi1) / (0.5*(isi1+isi2)**2)

    f = spk.spike_distance(t1, t2, 1.0)

    assert_equal(f.x, expected_times)
    assert_array_almost_equal(f.y1, expected_y1, decimal=14)
    assert_array_almost_equal(f.y2, expected_y2, decimal=14)

if __name__ == "main":
    test_isi()
    test_spike()
