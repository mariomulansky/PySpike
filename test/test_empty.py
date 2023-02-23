""" test_empty.py

Tests the distance measure for empty spike trains

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, \
    assert_array_equal, assert_array_almost_equal

import pyspike as spk
from pyspike import SpikeTrain


def test_get_non_empty():
    st = SpikeTrain([], edges=(0.0, 1.0))
    spikes = st.get_spikes_non_empty()
    assert_array_equal(spikes, [0.0, 1.0])

    st = SpikeTrain([0.5, ], edges=(0.0, 1.0))
    spikes = st.get_spikes_non_empty()
    # assert_array_equal(spikes, [0.0, 0.5, 1.0])
    # spike trains with one spike don't get edge spikes anymore
    assert_array_equal(spikes, [0.5, ])


def test_isi_empty():
    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([], edges=(0.0, 1.0))
    d = spk.isi_distance(st1, st2)
    assert_allclose(d, 0.0)
    prof = spk.isi_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 1.0])
    assert_array_equal(prof.y, [0.0, ])

    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.isi_distance(st1, st2)
    assert_allclose(d, 0.6*0.4+0.4*0.6)
    prof = spk.isi_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 0.4, 1.0])
    assert_array_equal(prof.y, [0.6, 0.4])

    st1 = SpikeTrain([0.6, ], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.isi_distance(st1, st2)
    assert_almost_equal(d, 0.2/0.6*0.4 + 0.0 + 0.2/0.6*0.4, decimal=15)
    prof = spk.isi_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_almost_equal(prof.x, [0.0, 0.4, 0.6, 1.0], decimal=15)
    assert_array_almost_equal(prof.y, [0.2/0.6, 0.0, 0.2/0.6], decimal=15)


def test_spike_empty():
    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([], edges=(0.0, 1.0))
    d = spk.spike_distance(st1, st2)
    assert_allclose(d, 0.0)
    prof = spk.spike_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 1.0])
    assert_array_equal(prof.y1, [0.0, ])
    assert_array_equal(prof.y2, [0.0, ])

    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.spike_distance(st1, st2)
    d_expect = 2*0.4*0.4*1.0/(0.4+1.0)**2 + 2*0.6*0.4*1.0/(0.6+1.0)**2
    assert_almost_equal(d, d_expect, decimal=15)
    prof = spk.spike_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 0.4, 1.0])
    assert_array_almost_equal(prof.y1, [2*0.4*1.0/(0.4+1.0)**2,
                                        2*0.4*1.0/(0.6+1.0)**2],
                              decimal=15)
    assert_array_almost_equal(prof.y2, [2*0.4*1.0/(0.4+1.0)**2,
                                        2*0.4*1.0/(0.6+1.0)**2],
                              decimal=15)

    st1 = SpikeTrain([0.6, ], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.spike_distance(st1, st2)
    s1 = np.array([0.2, 0.2, 0.2, 0.2])
    s2 = np.array([0.2, 0.2, 0.2, 0.2])
    isi1 = np.array([0.6, 0.6, 0.4])
    isi2 = np.array([0.4, 0.6, 0.6])
    expected_y1 = (s1[:-1]*isi2+s2[:-1]*isi1) / (0.5*(isi1+isi2)**2)
    expected_y2 = (s1[1:]*isi2+s2[1:]*isi1) / (0.5*(isi1+isi2)**2)
    expected_times = np.array([0.0, 0.4, 0.6, 1.0])
    expected_spike_val = sum((expected_times[1:] - expected_times[:-1]) *
                             (expected_y1+expected_y2)/2)
    expected_spike_val /= (expected_times[-1]-expected_times[0])

    assert_almost_equal(d, expected_spike_val, decimal=15)
    prof = spk.spike_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_almost_equal(prof.x, [0.0, 0.4, 0.6, 1.0], decimal=15)
    assert_array_almost_equal(prof.y1, expected_y1, decimal=15)
    assert_array_almost_equal(prof.y2, expected_y2, decimal=15)


def test_spike_sync_empty():
    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([], edges=(0.0, 1.0))
    d = spk.spike_sync(st1, st2)
    assert_allclose(d, 1.0)
    prof = spk.spike_sync_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 1.0])
    assert_array_equal(prof.y, [1.0, 1.0])

    st1 = SpikeTrain([], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.spike_sync(st1, st2)
    assert_allclose(d, 0.0)
    prof = spk.spike_sync_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_equal(prof.x, [0.0, 0.4, 1.0])
    assert_array_equal(prof.y, [0.0, 0.0, 0.0])

    st1 = SpikeTrain([0.6, ], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.4, ], edges=(0.0, 1.0))
    d = spk.spike_sync(st1, st2)
    assert_almost_equal(d, 1.0, decimal=15)
    prof = spk.spike_sync_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_almost_equal(prof.x, [0.0, 0.4, 0.6, 1.0], decimal=15)
    assert_array_almost_equal(prof.y, [1.0, 1.0, 1.0, 1.0], decimal=15)

    st1 = SpikeTrain([0.2, ], edges=(0.0, 1.0))
    st2 = SpikeTrain([0.8, ], edges=(0.0, 1.0))
    d = spk.spike_sync(st1, st2)
    assert_almost_equal(d, 0.0, decimal=15)
    prof = spk.spike_sync_profile(st1, st2)
    assert_allclose(d, prof.avrg())
    assert_array_almost_equal(prof.x, [0.0, 0.2, 0.8, 1.0], decimal=15)
    assert_array_almost_equal(prof.y, [0.0, 0.0, 0.0, 0.0], decimal=15)

    # test with empty intervals
    st1 = SpikeTrain([2.0, 5.0], [0, 10.0])
    st2 = SpikeTrain([2.1, 7.0], [0, 10.0])
    st3 = SpikeTrain([5.1, 6.0], [0, 10.0])
    res = spk.spike_sync_profile(st1, st2).avrg(interval=[3.0, 4.0])
    assert_allclose(res, 1.0)
    res = spk.spike_sync(st1, st2, interval=[3.0, 4.0])
    assert_allclose(res, 1.0)

    sync_matrix = spk.spike_sync_matrix([st1, st2, st3], interval=[3.0, 4.0])
    assert_array_equal(sync_matrix, np.ones((3, 3)))


if __name__ == "__main__":
    test_get_non_empty()
    test_isi_empty()
    test_spike_empty()
    test_spike_sync_empty()
