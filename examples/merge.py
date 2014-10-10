""" merge.py

Simple example showing the merging of two spike trains.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the MIT License (MIT)
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

# first load the data, ending time = 4000
spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 4000)

spikes = spk.merge_spike_trains([spike_trains[0], spike_trains[1]])

print(spikes)

plt.plot(spike_trains[0], np.ones_like(spike_trains[0]), 'o')
plt.plot(spike_trains[1], np.ones_like(spike_trains[1]), 'x')
plt.plot(spikes, 2*np.ones_like(spikes), 'o')

plt.show()
