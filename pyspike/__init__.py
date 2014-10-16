"""
Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

__all__ = ["function", "distances", "spikes"]

from function import PieceWiseConstFunc, PieceWiseLinFunc, average_profile
from distances import isi_profile, isi_distance, \
    spike_profile, spike_distance, \
    isi_profile_multi, isi_distance_multi, isi_distance_matrix, \
    spike_profile_multi, spike_distance_multi, spike_distance_matrix
from spikes import add_auxiliary_spikes, load_spike_trains_from_txt, \
    spike_train_from_string, merge_spike_trains
