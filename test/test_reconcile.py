import numpy as np
from numpy.testing import assert_allclose
from pyspike import SpikeTrain
from pyspike.spikes import reconcile_spike_trains
import pdb

def test_reconcile():
    ##input:
    tr1 = np.array([1,3,2,5])
    tr2 = np.array([1,4,4,10])

    edges1=[0,5]
    edges2=[3,9]

    ##expected output:
    edges=[0,9]
    trOut = [np.array([1,2,3,5]),
             np.array([1,4])]

    spike_trains = [SpikeTrain(tr1, edges1), SpikeTrain(tr2,edges2)]
    st_fixed = reconcile_spike_trains(spike_trains)

    assert len(st_fixed) == 2
    assert(st_fixed[0].t_start==edges[0])
    assert(st_fixed[0].t_end  ==edges[1])
    for i in range(2):
        assert_allclose(st_fixed[i].spikes, trOut[i])
        assert_allclose(st_fixed[i].t_start, 0)
        assert_allclose(st_fixed[i].t_end, 9)

if __name__ == "__main__":
    test_reconcile()