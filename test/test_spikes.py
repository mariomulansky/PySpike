""" test_load.py

Test loading of spike trains from text files

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_equal

import pyspike as spk


def test_auxiliary_spikes():
    t = np.array([0.2, 0.4, 0.6, 0.7])
    t_aux = spk.add_auxiliary_spikes(t, time_interval=(0.1, 1.0))
    assert_equal(t_aux, [0.1, 0.2, 0.4, 0.6, 0.7, 1.0])
    t_aux = spk.add_auxiliary_spikes(t_aux, time_interval=(0.0, 1.0))
    assert_equal(t_aux, [0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 1.0])


def test_load_from_txt():
    spike_trains = spk.load_spike_trains_from_txt("test/PySpike_testdata.txt",
                                                  time_interval=(0, 4000))
    assert len(spike_trains) == 40

    # check the first spike train
    spike_times = [0, 64.886, 305.81, 696, 937.77, 1059.7, 1322.2, 1576.1,
                   1808.1, 2121.5, 2381.1, 2728.6, 2966.9, 3223.7, 3473.7,
                   3644.3, 3936.3, 4000]
    assert_equal(spike_times, spike_trains[0])

    # check auxiliary spikes
    for spike_train in spike_trains:
        assert spike_train[0] == 0.0
        assert spike_train[-1] == 4000

    # load without adding auxiliary spikes
    spike_trains2 = spk.load_spike_trains_from_txt("test/PySpike_testdata.txt",
                                                   time_interval=None)
    assert len(spike_trains2) == 40
    # check auxiliary spikes
    for i in xrange(len(spike_trains)):
        assert len(spike_trains[i]) == len(spike_trains2[i])+2  # 2 spikes less
    

def check_merged_spikes(merged_spikes, spike_trains):
    # create a flat array with all spike events
    all_spikes = np.array([])
    for spike_train in spike_trains:
        all_spikes = np.append(all_spikes, spike_train)
    indices = np.zeros_like(all_spikes, dtype='bool')
    # check if we find all the spike events in the original spike trains
    for x in merged_spikes:
        i = np.where(all_spikes == x)[0][0]  # first axis and first entry
        # change to something impossible so we dont find this event again
        all_spikes[i] = -1.0
        indices[i] = True
    assert indices.all()


def test_merge_spike_trains():
    # first load the data
    spike_trains = spk.load_spike_trains_from_txt("test/PySpike_testdata.txt",
                                                  time_interval=(0, 4000))

    spikes = spk.merge_spike_trains([spike_trains[0], spike_trains[1]])
    # test if result is sorted
    assert((spikes == np.sort(spikes)).all())
    # check merging
    check_merged_spikes(spikes, [spike_trains[0], spike_trains[1]])

    spikes = spk.merge_spike_trains(spike_trains)
    # test if result is sorted
    assert((spikes == np.sort(spikes)).all())
    # check merging
    check_merged_spikes(spikes, spike_trains)

if __name__ == "main":
    test_auxiliary_spikes()
    test_load_from_txt()
    test_merge_spike_trains()
