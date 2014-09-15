# compute the isi distance of some test data
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

# first load the data
spike_trains = []
spike_file = open("SPIKY_testdata.txt", 'r')
for line in spike_file:
    spike_trains.append(spk.spike_train_from_string(line))

spikes = spk.merge_spike_trains([spike_trains[0], spike_trains[1]])

print(spikes)

plt.plot(spike_trains[0], np.ones_like(spike_trains[0]), 'o')
plt.plot(spike_trains[1], np.ones_like(spike_trains[1]), 'x')
plt.plot(spikes, 2*np.ones_like(spikes), 'o')

plt.show()
