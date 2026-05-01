"""
Copyright 2014-2018, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

__all__ = [
    "DiscreteFunc",
    "PieceWiseConstFunc",
    "PieceWiseLinFunc",
    "SpikeTrain",
    "filter_by_spike_sync",
    "generate_poisson_spikes",
    "import_spike_trains_from_time_series",
    "isi_distance",
    "isi_distance_matrix",
    "isi_distance_multi",
    "isi_profile",
    "isi_profile_multi",
    "load_spike_trains_from_txt",
    "psth",
    "spike_directionality",
    "spike_directionality_matrix",
    "spike_distance",
    "spike_distance_matrix",
    "spike_profile",
    "spike_sync",
    "spike_sync_matrix",
    "spike_sync_profile",
    "spike_sync_profile_multi",
    "spike_train_order",
    "spike_train_order_profile",
    "spikes",
]

from importlib.metadata import PackageNotFoundError, version

from .DiscreteFunc import DiscreteFunc
from .isi_distance import (
    isi_distance,
    isi_distance_matrix,
    isi_distance_multi,
    isi_profile,
    isi_profile_multi,
)
from .PieceWiseConstFunc import PieceWiseConstFunc
from .PieceWiseLinFunc import PieceWiseLinFunc
from .psth import psth
from .spike_directionality import (
    optimal_spike_train_sorting,
    permutate_matrix,
    spike_directionality,
    spike_directionality_matrix,
    spike_directionality_values,
    spike_train_order,
    spike_train_order_profile,
    spike_train_order_profile_multi,
)
from .spike_distance import (
    spike_distance,
    spike_distance_matrix,
    spike_distance_multi,
    spike_profile,
    spike_profile_multi,
)
from .spike_sync import (
    filter_by_spike_sync,
    spike_sync,
    spike_sync_matrix,
    spike_sync_multi,
    spike_sync_profile,
    spike_sync_profile_multi,
)
from .spikes import (
    generate_poisson_spikes,
    import_spike_trains_from_time_series,
    load_spike_trains_from_txt,
    merge_spike_trains,
    save_spike_trains_to_txt,
    spike_train_from_string,
)
from .SpikeTrain import SpikeTrain

try:
    __version__ = version("pyspike")
except PackageNotFoundError:
    __version__ = "Please install this project with setup.py"

disable_backend_warning = False


def NoCythonWarn():
    """Warn exactly once
    (called when an import of one of the cython_...so modules failed)
    """
    global disable_backend_warning  # initialized False in __init__.py
    if not disable_backend_warning:
        print(
            "Warning: Cython implementation not found."
            + " Make sure that PySpike is installed by running\n"
            + " 'python setup.py build_ext --inplace'\n"
            + "Falling back to slow python backend.\n"
        )
    disable_backend_warning = True
