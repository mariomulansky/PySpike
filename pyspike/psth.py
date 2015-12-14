# Module containing functions to compute the PSTH profile
# Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

import numpy as np
from pyspike import PieceWiseConstFunc


# Computes the peri-stimulus time histogram of a set of spike trains
def psth(spike_trains, bin_size):
    """ Computes the peri-stimulus time histogram of a set of
    :class:`.SpikeTrain`. The PSTH is simply the histogram of merged spike
    events. The :code:`bin_size` defines the width of the histogram bins.

    :param spike_trains: list of :class:`.SpikeTrain`
    :param bin_size: width of the histogram bins.
    :return: The PSTH as a :class:`.PieceWiseConstFunc`
    """

    bin_count = int((spike_trains[0].t_end - spike_trains[0].t_start) /
                    bin_size)
    bins = np.linspace(spike_trains[0].t_start, spike_trains[0].t_end,
                       bin_count+1)

    # N = len(spike_trains)
    combined_spike_train = spike_trains[0].spikes
    for i in range(1, len(spike_trains)):
        combined_spike_train = np.append(combined_spike_train,
                                         spike_trains[i].spikes)

    vals, edges = np.histogram(combined_spike_train, bins, density=False)

    bin_size = edges[1]-edges[0]
    return PieceWiseConstFunc(edges, vals)  # /(N*bin_size))
