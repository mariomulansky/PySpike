import numpy as np
from numpy.testing import assert_allclose
import pyspike as spk
from pyspike import SpikeTrain
from pyspike.isi_lengths import default_thresh

def gen_spike_trains():
    """ generate spike trains
    """
    t1 = SpikeTrain([0.2, 0.4, 0.6, 0.7], 1.0)
    t2 = SpikeTrain([0.3, 0.45, 0.8, 0.9, 0.95], 1.0)
    t3 = SpikeTrain([0.2, 0.4, 0.6], 1.0)
    t4 = SpikeTrain([0.1, 0.4, 0.5, 0.6], 1.0)
    return [t1, t2, t3, t4]

def auto_test(profile_func, profile_func_multi, dist_func_multi, dist_func_matrix, **kwargs):
    """ verify that MRTS='auto' works for the non-pair interfaces
        In: profile_func, profile_func_multi, dist_func_multi, dist_func_matrix
              -- functions to test for a particular distance
        asserts on error
    """
    spike_trains = gen_spike_trains()

    Thresh = default_thresh(spike_trains)
    Thresh2 = Thresh/1000

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
        r3 = dist_func_multi(spike_trains, MRTS=Thresh2, **kwargs)
        try:
            r1.almost_equal(r3)
        except:
            pass
        else:
            raise Exception('dist_func_multi ignores Thresh')

    if dist_func_matrix is not None:
        r1 = dist_func_matrix(spike_trains, MRTS=Thresh, **kwargs)
        r2 = dist_func_matrix(spike_trains, MRTS='auto', **kwargs)
        assert_allclose(r1, r2)

if __name__ == "__main__":
    """ driver for testing MRTS='auto' for non-pair interfaces
          goes through the various distances
    """
    auto_test(spk.isi_profile, 
                spk.isi_profile_multi,
                spk.isi_distance_multi,
                spk.isi_distance_matrix)
    auto_test(spk.spike_profile, 
                spk.spike_profile_multi,
                spk.spike_distance_multi,                          
                spk.spike_distance_matrix)
    auto_test(spk.spike_sync_profile,
                spk.spike_sync_profile_multi,
                None, 
                spk.spike_sync_matrix)
    auto_test(spk.spike_profile, 
                spk.spike_profile_multi,
                spk.spike_distance_multi, 
                spk.spike_distance_matrix, 
                RI=True)
    auto_test(None,
                None,
                None,
                spk.spike_directionality_matrix)
                     
        