# Module containing the class representing spike trains for PySpike.
# Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>
# Distributed under the BSD License

import numpy as np


class SpikeTrain(object):
    """ Class representing spike trains for the PySpike Module."""

    def __init__(self, spike_times, edges, is_sorted=True):
        """ Constructs the SpikeTrain.

        :param spike_times: ordered array of spike times.
        :param edges: The edges of the spike train. Given as a pair of floats
                      (T0, T1) or a single float T1, where then T0=0 is
                      assumed.
        :param is_sorted: If `False`, the spike times will sorted by `np.sort`.

        """

        # TODO: sanity checks
        if is_sorted:
            self.spikes = np.array(spike_times, dtype=float)
        else:
            self.spikes = np.sort(np.array(spike_times, dtype=float))

        try:
            self.t_start = float(edges[0])
            self.t_end = float(edges[1])
        except:
            self.t_start = 0.0
            self.t_end = float(edges)

    def __getitem__(self, index):
        """ Returns the time of the spike given by index.

        :param index: Index of the spike.
        :return: spike time.
        """
        return self.spikes[index]

    def __len__(self):
        """ Returns the number of spikes.
        
        :return: Number of spikes.
        """
        return len(self.spikes)

    def sort(self):
        """ Sorts the spike times of this spike train using `np.sort`
        """
        self.spikes = np.sort(self.spikes)

    def copy(self):
        """ Returns a copy of this spike train.
        Use this function if you want to create a real (deep) copy of this
        spike train. Simple assignment `t2 = t1` does not create a copy of the
        spike train data, but a reference as `numpy.array` is used for storing
        the data.

        :return: :class:`.SpikeTrain` copy of this spike train.

        """
        return SpikeTrain(self.spikes.copy(), [self.t_start, self.t_end])

    def get_spikes_non_empty(self):
        """Returns the spikes of this spike train with auxiliary spikes in case
        of empty spike trains.
        """
        if len(self.spikes) < 1:
            return np.unique(np.insert([self.t_start, self.t_end], 1,
                                       self.spikes))
        else:
            return self.spikes
