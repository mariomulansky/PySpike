""" test_merge_spikes.py

Tests merging spikes

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""
from __future__ import print_function
import numpy as np

import pyspike as spk

def check_merged_spikes( merged_spikes, spike_trains ):
    # create a flat array with all spike events
    all_spikes = np.array([])
    for spike_train in spike_trains:
        all_spikes = np.append(all_spikes, spike_train)
    indices = np.zeros_like(all_spikes, dtype='bool')
    # check if we find all the spike events in the original spike trains
    for x in merged_spikes:
        i = np.where(all_spikes == x)[0][0] # the first axis and the first entry
        # change to something impossible so we dont find this event again
        all_spikes[i] = -1.0
        indices[i] = True
    assert( indices.all() )

def test_merge_spike_trains():

    # first load the data
    spike_trains = []
    spike_file = open("SPIKY_testdata.txt", 'r')
    for line in spike_file:
        spike_trains.append(spk.spike_train_from_string(line))
        
    spikes = spk.merge_spike_trains([spike_trains[0], spike_trains[1]])
    # test if result is sorted
    assert((spikes == np.sort(spikes)).all())
    # check merging
    check_merged_spikes( spikes, [spike_trains[0], spike_trains[1]] )

    spikes = spk.merge_spike_trains(spike_trains)
    # test if result is sorted
    assert((spikes == np.sort(spikes)).all())
    # check merging
    check_merged_spikes( spikes, spike_trains )


if __name__ == "main":
    test_merge_spike_trains()

