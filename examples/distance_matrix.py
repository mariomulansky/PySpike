""" distance_matrix.py

Simple example showing how to compute the isi distance matrix of a set of spike
trains.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""


from __future__ import print_function

import matplotlib.pyplot as plt

import pyspike as spk

# first load the data, interval ending time = 4000, start=0 (default)
spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt", 4000)

print(len(spike_trains))

plt.figure()
isi_distance = spk.isi_distance_matrix(spike_trains)
plt.imshow(isi_distance, interpolation='none')
plt.title("ISI-distance")

plt.figure()
spike_distance = spk.spike_distance_matrix(spike_trains, interval=(0, 1000))
plt.imshow(spike_distance, interpolation='none')
plt.title("SPIKE-distance, T=0-1000")

plt.figure()
spike_sync = spk.spike_sync_matrix(spike_trains, interval=(2000, 4000))
plt.imshow(spike_sync, interpolation='none')
plt.title("SPIKE-Sync, T=2000-4000")

plt.show()
