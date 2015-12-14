"""
Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import absolute_import

__all__ = ["isi_distance", "spike_distance", "spike_sync", "psth",
           "spikes", "SpikeTrain", "PieceWiseConstFunc", "PieceWiseLinFunc",
           "DiscreteFunc", "directionality"]

from .PieceWiseConstFunc import PieceWiseConstFunc
from .PieceWiseLinFunc import PieceWiseLinFunc
from .DiscreteFunc import DiscreteFunc
from .SpikeTrain import SpikeTrain

from .isi_distance import isi_profile, isi_distance, isi_profile_multi,\
    isi_distance_multi, isi_distance_matrix
from .spike_distance import spike_profile, spike_distance, spike_profile_multi,\
    spike_distance_multi, spike_distance_matrix
from .spike_sync import spike_sync_profile, spike_sync,\
    spike_sync_profile_multi, spike_sync_multi, spike_sync_matrix
from .psth import psth

from .spikes import load_spike_trains_from_txt, spike_train_from_string, \
    merge_spike_trains, generate_poisson_spikes

# define the __version__ following
# http://stackoverflow.com/questions/17583443
from pkg_resources import get_distribution, DistributionNotFound
import os.path

try:
    _dist = get_distribution('pyspike')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'pyspike')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version

disable_backend_warning = False
