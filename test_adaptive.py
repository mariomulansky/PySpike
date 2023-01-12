import matplotlib.pyplot as plt
import pyspike as spk
import numpy as np

# from pyspike.cython.python_backend import (
#                                      isi_distance_python, 
#                                      spike_distance_python,  
#                                      coincidence_single_python)
# from pyspike.cython.cython_distances import (
#                                       isi_distance_cython,
#                                       spike_distance_cython
# )

threshold=100

t1 = np.array([64.88600, 305.81000, 696.00000])
t2 = np.array([63.76400, 318.45000, 697.48000])
edges = [0, 1000]
# from TK email 1/7/23
ISI_dist = 0.040407393279293
SPIKE_dist = 0.015370279259221

sp1 = spk.SpikeTrain(t1, edges)
sp2 = spk.SpikeTrain(t2, edges)

#times, values = isi_distance_python(sp1, sp2, edges[0], edges[1], Athresh=threshold)
#isiDist = spk.PieceWiseConstFunc(times, values).avrg(None)

isiDist = spk.isi_distance(sp1,sp2, Athresh=threshold)
print('ISI dist', isiDist)
np.testing.assert_almost_equal(isiDist, ISI_dist)

#times, valStarts, valEnds = spike_distance_python(sp1, sp2, edges[0], edges[1], Athresh=threshold)
#spikeDist = spk.PieceWiseLinFunc(times, valStarts, valEnds).avrg(None)

spikeDist = spk.spike_distance(sp1,sp2, Athresh=threshold)
print('SPIKE dist', spikeDist)
np.testing.assert_almost_equal(spikeDist, SPIKE_dist)

##################

#threshold=0  #hack

## larger scale example (except spike_sync)
spikes1 = [64.88600, 305.81000, 696.00000]
spikes2 = [66.415]
spikes3 = [66.449]
spikes4 = []
spikes5 = []
spikes6 = [63.76400, 318.45000, 697.48000]
spike_trains = [
    spk.SpikeTrain(spikes1, edges),
    spk.SpikeTrain(spikes2, edges),
    spk.SpikeTrain(spikes3, edges),
    spk.SpikeTrain(spikes4, edges),
    spk.SpikeTrain(spikes5, edges),
    spk.SpikeTrain(spikes6, edges),
]
isi_mat       = spk.isi_distance_matrix(spike_trains, Athresh=threshold)
spike_mat     = spk.spike_distance_matrix(spike_trains, Athresh=threshold)

iv,sv=[],[]
for i in range(5):
    for j in range(i+1,6):
        iv.append(isi_mat[i,j])
        sv.append(spike_mat[i,j])
print('Means. isi:%.10f, spike:%.10f'%(np.mean(iv), np.mean(sv)))



# expected, from TK email:
isi_dist_mat = [[0,        0.629777, 0.629754, 0.655457, 0.655457, 0.040407],
                [0.629777, 0,        0.000099, 0.124008, 0.124008, 0.637219],
                [0.629754, 0.000099, 0,        0.124067, 0.124067, 0.637198],
                [0.655457, 0.124008, 0.124067, 0,        0,        0.660567],
                [0.655457, 0.124008, 0.124067, 0,        0,        0.660567],
                [0.040407, 0.637219, 0.637198, 0.660567, 0.660567, 0]]

spike_dist_mat = [[0,        0.251968, 0.251993, 0.310829, 0.310829, 0.015370],
                  [0.251968, 0,        0.000067, 0.040925, 0.040925, 0.258082],
                  [0.251993, 0.000067, 0,        0.040949, 0.040949, 0.258107],
                  [0.310829, 0.040925, 0.040949, 0,        0,        0.313399],
                  [0.310829, 0.040925, 0.040949, 0,        0,        0.313399],
                  [0.015370, 0.258082, 0.258107, 0.313399, 0.313399, 0]]

print("Errors: isi mat=%.5f,spike mat=%.5f"%(np.linalg.norm(isi_dist_mat-isi_mat),
                                             np.linalg.norm(spike_dist_mat-spike_mat)))

## example for spike_sync:
spikes1 = [64.88600, 305.81000, 696.00000, 800.0000]
spikes2 = [67.88600, 302.81000, 699.00000]
spikes3 = [164.88600, 205.81000, 796.00000, 900.0000]
spikes4 = [263.76400, 418.45000, 997.48000]
spike_trains = [
    spk.SpikeTrain(spikes1, edges),
    spk.SpikeTrain(spikes2, edges),
    spk.SpikeTrain(spikes3, edges),
    spk.SpikeTrain(spikes4, edges),
]

spikesyncAdapt=0.666666666666667  #from email
spikesyncOther=0.285714285714286
spikesync_dist1 = spk.spike_sync_multi(spike_trains, Athresh=threshold)
spikesync_dist2 = spk.spike_sync_multi(spike_trains, Athresh=0.)
print("spikesync values")
print("%.15f"%spikesyncOther, "from email")
print("%.15f"%spikesync_dist1, "thresh=1000")
print("%.15f"%spikesync_dist1, "thresh=0")
