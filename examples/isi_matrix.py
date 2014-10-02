from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

# first load the data
spike_trains = []
spike_file = open("SPIKY_testdata.txt", 'r')
for line in spike_file:
    spike_trains.append(spk.add_auxiliary_spikes(
        spk.spike_train_from_string(line), 4000))

print(len(spike_trains))

m = spk.isi_distance_matrix(spike_trains)

plt.imshow(m, interpolation='none')
plt.show()

