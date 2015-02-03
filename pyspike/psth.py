"""

Module containing functions to compute the PSTH profile

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

import numpy as np
from pyspike import PieceWiseConstFunc


# Computes the Peristimulus time histogram of a set of spike trains
def psth(spike_trains, bin_size):

    bins = int((spike_trains[0][-1] - spike_trains[0][0]) / bin_size)

    N = len(spike_trains)
    combined_spike_train = spike_trains[0][1:-1]
    for i in xrange(1, len(spike_trains)):
        combined_spike_train = np.append(combined_spike_train,
                                         spike_trains[i][1:-1])

    vals, edges = np.histogram(combined_spike_train, bins, density=False)
    bin_size = edges[1]-edges[0]
    return PieceWiseConstFunc(edges, vals/(N*bin_size))
