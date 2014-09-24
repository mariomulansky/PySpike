""" distances.py

Module containing several functions to compute spike distances

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>
"""

import numpy as np

from pyspike import PieceWiseConstFunc, PieceWiseLinFunc

def isi_distance(spikes1, spikes2, T_end, T_start=0.0):
    """ Computes the instantaneous isi-distance S_isi (t) of the two given 
    spike trains. 
    Args:
    - spikes1, spikes2: ordered arrays of spike times.
    - T_end: end time of the observation interval.
    - T_start: begin of the observation interval (default=0.0).
    Returns:
    - PieceWiseConstFunc describing the isi-distance.
    """
    # add spikes at the beginning and end of the interval
    s1 = np.empty(len(spikes1)+2)
    s1[0] = T_start
    s1[-1] = T_end
    s1[1:-1] = spikes1
    s2 = np.empty(len(spikes2)+2)
    s2[0] = T_start
    s2[-1] = T_end
    s2[1:-1] = spikes2

    # compute the interspike interval
    nu1 = s1[1:]-s1[:-1]
    nu2 = s2[1:]-s2[:-1]
    
    # compute the isi-distance
    spike_events = np.empty(len(nu1)+len(nu2))
    spike_events[0] = T_start
    # the values have one entry less - the number of intervals between events
    isi_values = np.empty(len(spike_events)-1)
    # add the distance of the first events
    # isi_values[0] = nu1[0]/nu2[0] - 1.0 if nu1[0] <= nu2[0] \
    #                 else 1.0 - nu2[0]/nu1[0]
    isi_values[0] = (nu1[0]-nu2[0])/max(nu1[0],nu2[0])
    index1 = 0
    index2 = 0
    index = 1
    while True:
        # check which spike is next - from s1 or s2
        if s1[index1+1] < s2[index2+1]:
            index1 += 1
            # break condition relies on existence of spikes at T_end
            if index1 >= len(nu1):
                break
            spike_events[index] = s1[index1]
        elif s1[index1+1] > s2[index2+1]:
            index2 += 1
            if index2 >= len(nu2):
                break
            spike_events[index] = s2[index2]
        else: # s1[index1+1] == s2[index2+1]
            index1 += 1
            index2 += 1
            if (index1 >= len(nu1)) or (index2 >= len(nu2)):
                break
            spike_events[index] = s1[index1]
        # compute the corresponding isi-distance
        isi_values[index] = (nu1[index1]-nu2[index2]) / \
                            max(nu1[index1], nu2[index2])
        index += 1
    # the last event is the interval end
    spike_events[index] = T_end
    # use only the data added above 
    # could be less than original length due to equal spike times
    return PieceWiseConstFunc(spike_events[:index+1], isi_values[:index])


def get_min_dist(spike_time, spike_train, start_index=0):
    """ Returns the minimal distance |spike_time - spike_train[i]| 
    with i>=start_index
    """
    d = abs(spike_time - spike_train[start_index])
    start_index += 1
    while start_index < len(spike_train):
        d_temp = abs(spike_time - spike_train[start_index])
        if d_temp > d:
            break
        else:
            d = d_temp
        start_index += 1
    return d


def spike_distance(spikes1, spikes2, T_end, T_start=0.0):
    """ Computes the instantaneous spike-distance S_spike (t) of the two given
    spike trains.
    Args:
    - spikes1, spikes2: ordered arrays of spike times.
    - T_end: end time of the observation interval.
    - T_start: begin of the observation interval (default=0.0).
    Returns:
    - PieceWiseLinFunc describing the spike-distance.
    """
    # add spikes at the beginning and end of the interval
    t1 = np.empty(len(spikes1)+2)
    t1[0] = T_start
    t1[-1] = T_end
    t1[1:-1] = spikes1
    t2 = np.empty(len(spikes2)+2)
    t2[0] = T_start
    t2[-1] = T_end
    t2[1:-1] = spikes2

    spike_events = np.empty(len(t1)+len(t2)-2)
    spike_events[0] = T_start
    spike_events[-1] = T_end
    y_starts = np.empty(len(spike_events)-1)
    y_starts[0] = 0.0
    y_ends = np.empty(len(spike_events)-1)

    index1 = 0
    index2 = 0
    index = 1
    dt_p1 = 0.0
    dt_f1 = get_min_dist(t1[1], t2, 0)
    dt_p2 = 0.0
    dt_f2 = get_min_dist(t2[1], t1, 0)
    isi1 = t1[1]-t1[0]
    isi2 = t2[1]-t2[0]
    while True:
        print(index, index1, index2)
        if t1[index1+1] < t2[index2+1]:
            index1 += 1
            # break condition relies on existence of spikes at T_end
            if index1+1 >= len(t1):
                break
            spike_events[index] = t1[index1]
            # first calculate the previous interval end value
            dt_p1 = dt_f1 # the previous time now was the following time before
            s1 = dt_p1
            s2 = (dt_p2*(t2[index2+1]-t1[index1]) + dt_f2*(t1[index1]-t2[index2])) / isi2
            y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
            # now the next interval start value
            dt_f1 = get_min_dist(t1[index1+1], t2, index2)
            isi1 = t1[index1+1]-t1[index1]
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
        elif t1[index1+1] > t2[index2+1]:
            index2 += 1
            if index2+1 >= len(t2):
                break
            spike_events[index] = t2[index2]
            # first calculate the previous interval end value
            dt_p2 = dt_f2 # the previous time now was the following time before
            s1 = (dt_p1*(t1[index1+1]-t2[index2]) + dt_f1*(t2[index2]-t1[index1])) / isi1
            s2 = dt_p2
            y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
            # now the next interval start value
            dt_f2 = get_min_dist(t2[index2+1], t1, index1)
            #s2 = dt_f2
            isi2 = t2[index2+1]-t2[index2]
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
        else: # t1[index1+1] == t2[index2+1] - generate only one event
            index1 += 1
            index2 += 1
            if (index1+1 >= len(t1)) or (index2+1 >= len(t2)):
                break
            assert( dt_f2 == 0.0 )
            assert( dt_f1 == 0.0 )
            spike_events[index] = t1[index1]
            y_ends[index-1] = 0.0
            y_starts[index] = 0.0
            dt_p1 = 0.0
            dt_p2 = 0.0
            dt_f1 = get_min_dist(t1[index1+1], t2, index2)
            dt_f2 = get_min_dist(t2[index2+1], t1, index1)
            isi1 = t1[index1+1]-t1[index1]
            isi2 = t2[index2+1]-t2[index2]
        index += 1
    # the last event is the interval end
    spike_events[index] = T_end
    # the ending value of the last interval is 0
    y_ends[index-1] = 0.0
    # use only the data added above 
    # could be less than original length due to equal spike times
    return PieceWiseLinFunc(spike_events[:index+1], 
                            y_starts[:index], y_ends[:index])
