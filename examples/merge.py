""" merge.py

Simple example showing the merging of two spike trains.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

# first load the data, ending time = 4000
spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt", 4000)

merged_spike_train = spk.merge_spike_trains([spike_trains[0], spike_trains[1]])

print(merged_spike_train.spikes)

plt.plot(spike_trains[0], np.ones_like(spike_trains[0]), 'o')
plt.plot(spike_trains[1], np.ones_like(spike_trains[1]), 'x')
plt.plot(merged_spike_train.spikes,
         2*np.ones_like(merged_spike_train), 'o')

plt.show()
