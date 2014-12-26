from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                              time_interval=(0, 4000))

print(spike_trains[0])
print(spike_trains[1])

# plt.plot(spike_trains[0], np.ones_like(spike_trains[0]), 'o')
# plt.plot(spike_trains[1], np.zeros_like(spike_trains[1]), 'o')

plt.figure()

f = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
x, y = f.get_plottable_data()
plt.plot(x, y, '--k', label="SPIKE-SYNC profile")
print(x)
print(y)

f = spk.spike_profile(spike_trains[0], spike_trains[1])
x, y = f.get_plottable_data()

plt.plot(x, y, '-b', label="SPIKE-profile")

plt.legend(loc="upper left")

plt.show()
