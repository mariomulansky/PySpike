""" regression benchmark

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""
from __future__ import print_function

import os
import numpy as np
from scipy.io import loadmat
import pyspike as spk

from numpy.testing import assert_almost_equal

spk.disable_backend_warning = True

max_trr_trials = 100  # speed things up

def test_regression_random():

    spike_file = os.path.join("test", "numeric", "regression_random_spikes.mat")
    spikes_name = "spikes"
    result_name = "Distances"
    result_file = os.path.join("test", "numeric", "regression_random_results_cSPIKY.mat")

    spike_train_sets = loadmat(spike_file)[spikes_name][0]
    results_cSPIKY = loadmat(result_file)[result_name]

    for i, spike_train_data in enumerate(spike_train_sets):
        if i >= max_trr_trials:
            break
        spike_trains = []
        for spikes in spike_train_data[0]:
            spike_trains.append(spk.SpikeTrain(spikes.flatten(), 100.0))

        isi = spk.isi_distance_multi(spike_trains)
        isi_prof = spk.isi_profile_multi(spike_trains).avrg()

        spike = spk.spike_distance_multi(spike_trains)
        spike_prof = spk.spike_profile_multi(spike_trains).avrg()

        spike_sync = spk.spike_sync_multi(spike_trains)
        spike_sync_prof = spk.spike_sync_profile_multi(spike_trains).avrg()

        assert_almost_equal(isi, results_cSPIKY[i][0], decimal=14,
                            err_msg="Index: %d, ISI" % i)
        assert_almost_equal(isi_prof, results_cSPIKY[i][0], decimal=14,
                            err_msg="Index: %d, ISI" % i)

        assert_almost_equal(spike, results_cSPIKY[i][1], decimal=14,
                            err_msg="Index: %d, SPIKE" % i)
        assert_almost_equal(spike_prof, results_cSPIKY[i][1], decimal=14,
                            err_msg="Index: %d, SPIKE" % i)

        assert_almost_equal(spike_sync, spike_sync_prof, decimal=14,
                            err_msg="Index: %d, SPIKE-Sync" % i)


def check_regression_dataset(spike_file="benchmark.mat",
                             spikes_name="spikes",
                             result_file="results_cSPIKY.mat",
                             result_name="Distances"):
    """ Debuging function """
    np.set_printoptions(precision=15)

    spike_train_sets = loadmat(spike_file)[spikes_name][0]

    results_cSPIKY = loadmat(result_file)[result_name]

    err_max = 0.0
    err_max_ind = -1
    err_count = 0

    for i, spike_train_data in enumerate(spike_train_sets):
        if i >= max_trr_trials:
            break
        spike_trains = []
        for spikes in spike_train_data[0]:
            spike_trains.append(spk.SpikeTrain(spikes.flatten(), 100.0))

        isi = spk.isi_distance_multi(spike_trains)
        spike = spk.spike_distance_multi(spike_trains)
        # spike_sync = spk.spike_sync_multi(spike_trains)

        if abs(isi - results_cSPIKY[i][0]) > 1E-14:
            print("Error in ISI:", i, isi, results_cSPIKY[i][0])
            print("Spike trains:")
            for st in spike_trains:
                print(st.spikes)

        err = abs(spike - results_cSPIKY[i][1])
        if err > 1E-14:
            err_count += 1
        if err > err_max:
            err_max = err
            err_max_ind = i

    print("Total Errors:", err_count)

    if err_max_ind > -1:
        print("Max SPIKE distance error:", err_max, "at index:", err_max_ind)
        spike_train_data = spike_train_sets[err_max_ind]
        for spikes in spike_train_data[0]:
            print(spikes.flatten())


def check_single_spike_train_set(index):
    """ Debuging function """
    np.set_printoptions(precision=15)
    spike_file = os.path.join("test", "numeric", "regression_random_spikes.mat")
    spikes_name = "spikes"
    result_name = "Distances"
    result_file = os.path.join("test", "numeric", "regression_random_results_cSPIKY.mat")

    spike_train_sets = loadmat(spike_file)[spikes_name][0]

    results_cSPIKY = loadmat(result_file)[result_name]

    spike_train_data = spike_train_sets[index]

    spike_trains = []
    N = 0
    for spikes in spike_train_data[0]:
        N += len(spikes.flatten())
        print("Spikes:", len(spikes.flatten()))
        spikes_array = spikes.flatten()
        if len(spikes_array > 0) and (spikes_array[-1] > 100.0):
            spikes_array[-1] = 100.0
        spike_trains.append(spk.SpikeTrain(spikes_array, 100.0))
        print(spike_trains[-1].spikes)

    print(N)

    print(spk.spike_sync_multi(spike_trains))

    print(spk.spike_sync_profile_multi(spike_trains).integral())


if __name__ == "__main__":

    test_regression_random()
    check_regression_dataset()
    check_single_spike_train_set(4)
