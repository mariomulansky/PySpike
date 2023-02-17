import numpy as np
from numpy.testing import assert_allclose
import pyspike as spk
from pyspike import SpikeTrain
from pyspike.isi_lengths import default_thresh
import pdb

def gen_spike_trains():
        # generate spike trains:
    t1 = SpikeTrain([0.2, 0.4, 0.6, 0.7], 1.0)
    t2 = SpikeTrain([0.3, 0.45, 0.8, 0.9, 0.95], 1.0)
    t3 = SpikeTrain([0.2, 0.4, 0.6], 1.0)
    t4 = SpikeTrain([0.1, 0.4, 0.5, 0.6], 1.0)
    return [t1, t2, t3, t4]

def test_auto_multi(profile_func, profile_func_multi, dist_func_multi, dist_func_matrix, **kwargs):
    spike_trains = gen_spike_trains()

    Thresh = default_thresh(spike_trains)

    if profile_func is not None:
        r1 = profile_func(spike_trains, MRTS=Thresh, **kwargs)
        r2 = profile_func(spike_trains, MRTS='auto', **kwargs)
        r1.almost_equal(r2)

    if profile_func_multi is not None:
        r1 = profile_func_multi(spike_trains, MRTS=Thresh, **kwargs)
        r2 = profile_func_multi(spike_trains, MRTS='auto', **kwargs)
        r1.almost_equal(r2)

    if dist_func_multi is not None:
        r1 = dist_func_multi(spike_trains, MRTS=Thresh, **kwargs)
        r2 = dist_func_multi(spike_trains, MRTS='auto', **kwargs)
        assert_allclose(r1, r2)

    if dist_func_matrix is not None:
        r1 = dist_func_matrix(spike_trains, MRTS=Thresh, **kwargs)
        r2 = dist_func_matrix(spike_trains, MRTS='auto', **kwargs)
        assert_allclose(r1, r2)

if __name__ == "__main__":
    test_auto_multi(spk.isi_profile, 
                    spk.isi_profile_multi,
                    spk.isi_distance_multi,
                    spk.isi_distance_matrix)
    test_auto_multi(spk.spike_profile, 
                    spk.spike_profile_multi,
                    spk.spike_distance_multi,                          
                    spk.spike_distance_matrix)
    test_auto_multi(spk.spike_sync_profile,
                    spk.spike_sync_profile_multi,
                    None, 
                    spk.spike_sync_matrix)
    test_auto_multi(spk.spike_profile, 
                    spk.spike_profile_multi,
                    spk.spike_distance_multi, 
                    spk.spike_distance_matrix, 
                    RIA=True)
    test_auto_multi(None,
                    None,
                    None, 
                    spk.spike_directionality_matrix)
                     
        