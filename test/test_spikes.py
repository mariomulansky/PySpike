""" test_load.py

Test loading of spike trains from text files

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_equal

import pyspike as spk

import os
TEST_PATH = os.path.dirname(os.path.realpath(__file__))
TEST_DATA = os.path.join(TEST_PATH, "PySpike_testdata.txt")

def test_load_from_txt():
    spike_trains = spk.load_spike_trains_from_txt(TEST_DATA, edges=(0, 4000))
    assert len(spike_trains) == 40

    # check the first spike train
    spike_times = [64.886, 305.81, 696, 937.77, 1059.7, 1322.2, 1576.1,
                   1808.1, 2121.5, 2381.1, 2728.6, 2966.9, 3223.7, 3473.7,
                   3644.3, 3936.3]
    assert_equal(spike_times, spike_trains[0].spikes)

    # check auxiliary spikes
    for spike_train in spike_trains:
        assert spike_train.t_start == 0.0
        assert spike_train.t_end == 4000


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
    spike_trains = spk.load_spike_trains_from_txt(TEST_DATA, edges=(0, 4000))

    merged_spikes = spk.merge_spike_trains([spike_trains[0], spike_trains[1]])
    # test if result is sorted
    assert((merged_spikes.spikes == np.sort(merged_spikes.spikes)).all())
    # check merging
    check_merged_spikes(merged_spikes.spikes, [spike_trains[0].spikes,
                                               spike_trains[1].spikes])

    merged_spikes = spk.merge_spike_trains(spike_trains)
    # test if result is sorted
    assert((merged_spikes.spikes == np.sort(merged_spikes.spikes)).all())
    # check merging
    check_merged_spikes(merged_spikes.spikes,
                        [st.spikes for st in spike_trains])


if __name__ == "main":
    test_load_from_txt()
    test_merge_spike_trains()
