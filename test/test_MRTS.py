""" test_MRTS.py

Tests the MRTS logic in all distances
Also, test automatic generation of the threshold

Copyright 2023, Thomas Kreuz
Distributed under the BSD License
"""
import numpy as np
import pyspike as spk
from pyspike import SpikeTrain
from pyspike.isi_lengths import default_thresh

def test_MRTS():
    """ single testcase for all 4 algorithms changing MRTS:
    """
    v1 = [12.0000, 16.0000, 28.0000, 32.0000, 44.0000, 48.0000, 60.0000, 64.0000, 76.0000, 80.0000, ];
    v2 = [7.5376, 19.9131, 24.2137, 35.7255, 40.0961, 51.7076, 55.9124, 68.1017, 71.9863, 83.5994, ];
    edges=[0, 300]
    max_tau=1000

    sp1 = spk.SpikeTrain(v1, edges)
    sp2 = spk.SpikeTrain(v2, edges)

    ## SPIKE-SYNC
    Results1 = {14:0., 15:.3, 16:.6, 17:.9, 18:1.}
    for r in Results1:
        c = spk.spike_sync(sp1, sp2, MRTS=r)
        np.testing.assert_almost_equal(c, Results1[r])

    ## SPIKE
    Results2 = {
        0 : 0.12095,
        1 : 0.12095,
        2 : 0.12095,
        3 : 0.12095,
        4 : 0.12095,
        5 : 0.12095,
        6 : 0.12095,
        7 : 0.12095,
        8 : 0.12039,
        9 : 0.11434,
        10 : 0.10900,
        11 : 0.10464,
        12 : 0.10064,
        13 : 0.09418,
        14 : 0.08833,
        15 : 0.08326,
        16 : 0.07882,
        17 : 0.07491,
        18 : 0.07143,
        19 : 0.06832,
        20 : 0.06551,
        21 : 0.06298,
        22 : 0.06067,
        23 : 0.05857,
        24 : 0.05664,
        25 : 0.05487,
        26 : 0.05323,
        27 : 0.05171,
        28 : 0.05030,
        29 : 0.04899,
        30 : 0.04777,
        31 : 0.04662,
        32 : 0.04555,
        33 : 0.04454,
        34 : 0.04359,
        35 : 0.04270,
        36 : 0.04185,
        37 : 0.04105,
        38 : 0.04030,
        39 : 0.03958,
        40 : 0.03890,
        41 : 0.03825,
        42 : 0.03763,
        43 : 0.03704,
        44 : 0.03648,
        45 : 0.03594,
        46 : 0.03542,
        47 : 0.03493,
        48 : 0.03446,
        49 : 0.03401,
        50 : 0.03357,
    }
    for r in Results2:
        d = spk.spike_distance(sp1, sp2, MRTS=r)
        np.testing.assert_almost_equal(d, Results2[r], decimal=5)


    ## RI
    Results3 = {
        0 : 0.12094,
        1 : 0.12094,
        2 : 0.12094,
        3 : 0.12094,
        4 : 0.12094,
        5 : 0.12094,
        6 : 0.12094,
        7 : 0.12094,
        8 : 0.12038,
        9 : 0.11432,
        10 : 0.10899,
        11 : 0.10463,
        12 : 0.10063,
        13 : 0.09417,
        14 : 0.08832,
        15 : 0.08325,
        16 : 0.07882,
        17 : 0.07490,
        18 : 0.07142,
        19 : 0.06831,
        20 : 0.06551,
        21 : 0.06297,
        22 : 0.06067,
        23 : 0.05856,
        24 : 0.05664,
        25 : 0.05486,
        26 : 0.05322,
        27 : 0.05171,
        28 : 0.05030,
        29 : 0.04899,
        30 : 0.04776,
        31 : 0.04662,
        32 : 0.04555,
        33 : 0.04454,
        34 : 0.04359,
        35 : 0.04269,
        36 : 0.04185,
        37 : 0.04105,
        38 : 0.04029,
        39 : 0.03957,
        40 : 0.03889,
        41 : 0.03824,
        42 : 0.03762,
        43 : 0.03703,
        44 : 0.03647,
        45 : 0.03593,
        46 : 0.03542,
        47 : 0.03493,
        48 : 0.03446,
        49 : 0.03400,
        50 : 0.03357,
    }

    for r in Results3:
        d = spk.spike_distance(sp1, sp2, MRTS=r, RI=True)
        #print('%d : %.5f,'%(r, d))
        np.testing.assert_almost_equal(d, Results3[r], decimal=5)

    ## ISI
    Results4 = {
        0 : 0.10796,
        1 : 0.10796,
        2 : 0.10796,
        3 : 0.10796,
        4 : 0.10796,
        5 : 0.10796,
        6 : 0.10796,
        7 : 0.10796,
        8 : 0.10796,
        9 : 0.10796,
        10 : 0.10796,
        11 : 0.10796,
        12 : 0.10704,
        13 : 0.10103,
        14 : 0.09547,
        15 : 0.09065,
        16 : 0.08643,
        17 : 0.08271,
        18 : 0.07940,
        19 : 0.07644,
        20 : 0.07378,
        21 : 0.07137,
        22 : 0.06918,
        23 : 0.06718,
        24 : 0.06534,
        25 : 0.06366,
        26 : 0.06210,
        27 : 0.06066,
        28 : 0.05932,
        29 : 0.05807,
        30 : 0.05691,
        31 : 0.05582,
        32 : 0.05480,
        33 : 0.05384,
        34 : 0.05294,
        35 : 0.05209,
        36 : 0.05128,
        37 : 0.05052,
        38 : 0.04980,
        39 : 0.04912,
        40 : 0.04847,
        41 : 0.04785,
        42 : 0.04727,
        43 : 0.04671,
        44 : 0.04617,
        45 : 0.04566,
        46 : 0.04517,
        47 : 0.04470,
        48 : 0.04425,
        49 : 0.04382,
        50 : 0.04341,    
    }    

    for r in Results4:
        d = spk.isi_distance(sp1, sp2, MRTS=r)
        np.testing.assert_almost_equal(d, Results4[r], decimal=5)

    print('OK1')

def test_autoThresh():
    """ Automatic determination of MRTS
    """
    edges = [0, 1000]
    spikes1 = SpikeTrain([64.88600, 305.81000, 696.00000, 800.0000], edges)
    spikes2 = SpikeTrain([67.88600, 302.81000, 699.00000], edges)
    spikes3 = SpikeTrain([164.88600, 205.81000, 796.00000, 900.0000], edges)
    spikes4 = SpikeTrain([263.76400, 418.45000, 997.48000], edges)
    spike_train_list = [spikes1, spikes2, spikes3, spikes4]

    Thresh = default_thresh(spike_train_list)
    print('default_thresh got %.4f'%Thresh)
    np.testing.assert_almost_equal(Thresh, 325.4342, decimal=4, err_msg="default_thresh")

    c1 = spk.spike_sync(spikes1, spikes2, MRTS=Thresh)
    c2 = spk.spike_sync(spikes1, spikes2, MRTS='auto')
    np.testing.assert_almost_equal(c1, c2, err_msg="spike_sync")

    # apply it to the first example avove
    v1 = [12.0000, 16.0000, 28.0000, 32.0000, 44.0000, 48.0000, 60.0000, 64.0000, 76.0000, 80.0000, ];
    v2 = [7.5376, 19.9131, 24.2137, 35.7255, 40.0961, 51.7076, 55.9124, 68.1017, 71.9863, 83.5994, ];
    edges=[0, 300]

    sp1 = spk.SpikeTrain(v1, edges)
    sp2 = spk.SpikeTrain(v2, edges)

    t = default_thresh([sp1, sp2])
    ## Look at all 4 algorithms

    c1 = spk.spike_sync(sp1, sp2, MRTS=t)
    c2 = spk.spike_sync(sp1, sp2, MRTS='auto')
    np.testing.assert_almost_equal(c1, c2, err_msg="spike_sync2")
    print('SS thresh %.3f, results %.3f'%(t,c1))
    # compare with: {14:0., 15:.3, 16:.6, 17:.9, 18:1.}

    c1 = spk.spike_distance(sp1, sp2, MRTS=t)
    c2 = spk.spike_distance(sp1, sp2, MRTS='auto')
    np.testing.assert_almost_equal(c1, c2, err_msg="spike_distance")

    c1 = spk.spike_distance(sp1, sp2, MRTS=t, RI=True)
    c2 = spk.spike_distance(sp1, sp2, MRTS='auto', RI=True)
    np.testing.assert_almost_equal(c1, c2, err_msg="RI")

    c1 = spk.isi_distance(sp1, sp2, MRTS=t)
    c2 = spk.isi_distance(sp1, sp2, MRTS='auto')
    np.testing.assert_almost_equal(c1, c2, err_msg="ISI")

    c1 = spk.spike_directionality(sp1, sp2, MRTS=t)
    c2 = spk.spike_directionality(sp1, sp2, MRTS='auto')
    np.testing.assert_almost_equal(c1, c2, err_msg="directionality")

    print('OK2')

if __name__ == "__main__":
    test_MRTS()
    test_autoThresh()