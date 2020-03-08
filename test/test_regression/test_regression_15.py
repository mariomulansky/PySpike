""" test_regression_15.py

Regression test for Issue #15

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, \
    assert_array_almost_equal

import pyspike as spk

import os
TEST_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_DATA = os.path.join(TEST_PATH, "..", "SPIKE_Sync_Test.txt")


def test_regression_15_isi():
    # load spike trains
    spike_trains = spk.load_spike_trains_from_txt(TEST_DATA, edges=[0, 4000])

    N = len(spike_trains)

    dist_mat = spk.isi_distance_matrix(spike_trains)
    assert_allclose(dist_mat.shape, (N, N))

    ind = np.arange(N//2)
    dist_mat = spk.isi_distance_matrix(spike_trains, ind)
    assert_allclose(dist_mat.shape, (N//2, N//2))

    ind = np.arange(N//2, N)
    dist_mat = spk.isi_distance_matrix(spike_trains, ind)
    assert_allclose(dist_mat.shape, (N//2, N//2))


def test_regression_15_spike():
    # load spike trains
    spike_trains = spk.load_spike_trains_from_txt(TEST_DATA, edges=[0, 4000])

    N = len(spike_trains)

    dist_mat = spk.spike_distance_matrix(spike_trains)
    assert_allclose(dist_mat.shape, (N, N))

    ind = np.arange(N//2)
    dist_mat = spk.spike_distance_matrix(spike_trains, ind)
    assert_allclose(dist_mat.shape, (N//2, N//2))

    ind = np.arange(N//2, N)
    dist_mat = spk.spike_distance_matrix(spike_trains, ind)
    assert_allclose(dist_mat.shape, (N//2, N//2))


def test_regression_15_sync():
    # load spike trains
    spike_trains = spk.load_spike_trains_from_txt(TEST_DATA, edges=[0, 4000])

    N = len(spike_trains)

    dist_mat = spk.spike_sync_matrix(spike_trains)
    assert_allclose(dist_mat.shape, (N, N))

    ind = np.arange(N//2)
    dist_mat = spk.spike_sync_matrix(spike_trains, ind)
    assert_allclose(dist_mat.shape, (N//2, N//2))

    ind = np.arange(N//2, N)
    dist_mat = spk.spike_sync_matrix(spike_trains, ind)
    assert_allclose(dist_mat.shape, (N//2, N//2))


if __name__ == "__main__":
    test_regression_15_isi()
    test_regression_15_spike()
    test_regression_15_sync()
