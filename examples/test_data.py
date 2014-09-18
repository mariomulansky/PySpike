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

# plot the spike time
for (i,spikes) in enumerate(spike_trains):
    plt.plot(spikes, i*np.ones_like(spikes), 'o')

f = spk.isi_distance(spike_trains[0], spike_trains[10], 4000)
x, y = f.get_plottable_data()

plt.figure()
plt.plot(x, np.abs(y), '--k')

print("Average: %.8f" % f.avrg())
print("Absolute average: %.8f" % f.abs_avrg())


f = spk.spike_distance(spike_trains[0], spike_trains[10], 4000)
x, y = f.get_plottable_data()
print(x)
print(y)
#plt.figure()
plt.plot(x, y, '-b')

plt.show()
