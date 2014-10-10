from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

# first load the data, interval ending time = 4000, start=0 (default)
spike_trains = spk.load_spike_trains_from_txt("SPIKY_testdata.txt", 4000)

print(len(spike_trains))

m = spk.isi_distance_matrix(spike_trains)

plt.imshow(m, interpolation='none')
plt.show()

