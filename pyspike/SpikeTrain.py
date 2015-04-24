""" Module containing the class representing spike trains for PySpike.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

import numpy as np
import collections


class SpikeTrain:
    """ Class representing spike trains for the PySpike Module."""

    def __init__(self, spike_times, interval):
        """ Constructs the SpikeTrain
        :param spike_times: ordered array of spike times.
        :param interval: interval defining the edges of the spike train.
        Given as a pair of floats (T0, T1) or a single float T1, where T0=0 is
        assumed.
        """

        # TODO: sanity checks
        self.spikes = np.array(spike_times, dtype=float)

        # check if interval is as sequence
        if not isinstance(interval, collections.Sequence):
            # treat value as end time and assume t_start = 0
            self.t_start = 0.0
            self.t_end = float(interval)
        else:
            # extract times from sequence
            self.t_start = float(interval[0])
            self.t_end = float(interval[1])
