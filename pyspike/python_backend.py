""" python_backend.py

Collection of python functions that can be used instead of the cython
implementation.

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

import numpy as np


############################################################
# isi_distance_python
############################################################
def isi_distance_python(s1, s2):
    """ Plain Python implementation of the isi distance.
    """
    # compute the interspike interval
    nu1 = s1[1:] - s1[:-1]
    nu2 = s2[1:] - s2[:-1]

    # compute the isi-distance
    spike_events = np.empty(len(nu1) + len(nu2))
    spike_events[0] = s1[0]
    # the values have one entry less - the number of intervals between events
    isi_values = np.empty(len(spike_events) - 1)
    # add the distance of the first events
    # isi_values[0] = nu1[0]/nu2[0] - 1.0 if nu1[0] <= nu2[0] \
    #                 else 1.0 - nu2[0]/nu1[0]
    isi_values[0] = abs(nu1[0] - nu2[0]) / max(nu1[0], nu2[0])
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
        else:  # s1[index1 + 1] == s2[index2 + 1]
            index1 += 1
            index2 += 1
            if (index1 >= len(nu1)) or (index2 >= len(nu2)):
                break
            spike_events[index] = s1[index1]
        # compute the corresponding isi-distance
        isi_values[index] = abs(nu1[index1] - nu2[index2]) / \
            max(nu1[index1], nu2[index2])
        index += 1
    # the last event is the interval end
    spike_events[index] = s1[-1]
    # use only the data added above
    # could be less than original length due to equal spike times
    return spike_events[:index + 1], isi_values[:index]


############################################################
# get_min_dist
############################################################
def get_min_dist(spike_time, spike_train, start_index=0):
    """ Returns the minimal distance |spike_time - spike_train[i]|
    with i>=start_index.
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


############################################################
# spike_distance_python
############################################################
def spike_distance_python(spikes1, spikes2):
    """ Computes the instantaneous spike-distance S_spike (t) of the two given
    spike trains. The spike trains are expected to have auxiliary spikes at the
    beginning and end of the interval. Use the function add_auxiliary_spikes to
    add those spikes to the spike train.
    Args:
    - spikes1, spikes2: ordered arrays of spike times with auxiliary spikes.
    Returns:
    - PieceWiseLinFunc describing the spike-distance.
    """
    # check for auxiliary spikes - first and last spikes should be identical
    assert spikes1[0] == spikes2[0], \
        "Given spike trains seems not to have auxiliary spikes!"
    assert spikes1[-1] == spikes2[-1], \
        "Given spike trains seems not to have auxiliary spikes!"
    # shorter variables
    t1 = spikes1
    t2 = spikes2

    spike_events = np.empty(len(t1) + len(t2) - 2)
    spike_events[0] = t1[0]
    y_starts = np.empty(len(spike_events) - 1)
    y_ends = np.empty(len(spike_events) - 1)

    index1 = 0
    index2 = 0
    index = 1
    dt_p1 = 0.0
    dt_f1 = get_min_dist(t1[1], t2, 0)
    dt_p2 = 0.0
    dt_f2 = get_min_dist(t2[1], t1, 0)
    isi1 = max(t1[1]-t1[0], t1[2]-t1[1])
    isi2 = max(t2[1]-t2[0], t2[2]-t2[1])
    s1 = dt_f1*(t1[1]-t1[0])/isi1
    s2 = dt_f2*(t2[1]-t2[0])/isi2
    y_starts[0] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
    while True:
        # print(index, index1, index2)
        if t1[index1+1] < t2[index2+1]:
            index1 += 1
            # break condition relies on existence of spikes at T_end
            if index1+1 >= len(t1):
                break
            spike_events[index] = t1[index1]
            # first calculate the previous interval end value
            dt_p1 = dt_f1  # the previous time was the following time before
            s1 = dt_p1
            s2 = (dt_p2*(t2[index2+1]-t1[index1]) +
                  dt_f2*(t1[index1]-t2[index2])) / isi2
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
            dt_p2 = dt_f2  # the previous time was the following time before
            s1 = (dt_p1*(t1[index1+1]-t2[index2]) +
                  dt_f1*(t2[index2]-t1[index1])) / isi1
            s2 = dt_p2
            y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
            # now the next interval start value
            dt_f2 = get_min_dist(t2[index2+1], t1, index1)
            #s2 = dt_f2
            isi2 = t2[index2+1]-t2[index2]
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
        else:  # t1[index1+1] == t2[index2+1] - generate only one event
            index1 += 1
            index2 += 1
            if (index1+1 >= len(t1)) or (index2+1 >= len(t2)):
                break
            assert dt_f2 == 0.0
            assert dt_f1 == 0.0
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
    spike_events[index] = t1[-1]
    # the ending value of the last interval
    isi1 = max(t1[-1]-t1[-2], t1[-2]-t1[-3])
    isi2 = max(t2[-1]-t2[-2], t2[-2]-t2[-3])
    s1 = dt_p1*(t1[-1]-t1[-2])/isi1
    s2 = dt_p2*(t2[-1]-t2[-2])/isi2
    y_ends[index-1] = (s1*isi2 + s2*isi1) / ((isi1+isi2)**2/2)
    # use only the data added above
    # could be less than original length due to equal spike times
    return spike_events[:index+1], y_starts[:index], y_ends[:index]


############################################################
# add_piece_wise_const_python
############################################################
def add_piece_wise_const_python(x1, y1, x2, y2):
    x_new = np.empty(len(x1) + len(x2))
    y_new = np.empty(len(x_new)-1)
    x_new[0] = x1[0]
    y_new[0] = y1[0] + y2[0]
    index1 = 0
    index2 = 0
    index = 0
    while (index1+1 < len(y1)) and (index2+1 < len(y2)):
        index += 1
        # print(index1+1, x1[index1+1], y1[index1+1], x_new[index])
        if x1[index1+1] < x2[index2+1]:
            index1 += 1
            x_new[index] = x1[index1]
        elif x1[index1+1] > x2[index2+1]:
            index2 += 1
            x_new[index] = x2[index2]
        else:  # x1[index1+1] == x2[index2+1]:
            index1 += 1
            index2 += 1
            x_new[index] = x1[index1]
        y_new[index] = y1[index1] + y2[index2]
    # one array reached the end -> copy the contents of the other to the end
    if index1+1 < len(y1):
        x_new[index+1:index+1+len(x1)-index1-1] = x1[index1+1:]
        y_new[index+1:index+1+len(y1)-index1-1] = y1[index1+1:] + y2[-1]
        index += len(x1)-index1-2
    elif index2+1 < len(y2):
        x_new[index+1:index+1+len(x2)-index2-1] = x2[index2+1:]
        y_new[index+1:index+1+len(y2)-index2-1] = y2[index2+1:] + y1[-1]
        index += len(x2)-index2-2
    else:  # both arrays reached the end simultaneously
        # only the last x-value missing
        x_new[index+1] = x1[-1]
    # the last value is again the end of the interval
    # x_new[index+1] = x1[-1]
    # only use the data that was actually filled

    return x_new[:index+2], y_new[:index+1]


############################################################
# add_piece_lin_const_python
############################################################
def add_piece_wise_lin_python(x1, y11, y12, x2, y21, y22):
    x_new = np.empty(len(x1) + len(x2))
    y1_new = np.empty(len(x_new)-1)
    y2_new = np.empty_like(y1_new)
    x_new[0] = x1[0]
    y1_new[0] = y11[0] + y21[0]
    index1 = 0  # index for self
    index2 = 0  # index for f
    index = 0   # index for new
    while (index1+1 < len(y11)) and (index2+1 < len(y21)):
        # print(index1+1, x1[index1+1], self.y[index1+1], x_new[index])
        if x1[index1+1] < x2[index2+1]:
            # first compute the end value of the previous interval
            # linear interpolation of the interval
            y = y21[index2] + (y22[index2]-y21[index2]) * \
                (x1[index1+1]-x2[index2]) / (x2[index2+1]-x2[index2])
            y2_new[index] = y12[index1] + y
            index1 += 1
            index += 1
            x_new[index] = x1[index1]
            # and the starting value for the next interval
            y1_new[index] = y11[index1] + y
        elif x1[index1+1] > x2[index2+1]:
            # first compute the end value of the previous interval
            # linear interpolation of the interval
            y = y11[index1] + (y12[index1]-y11[index1]) * \
                (x2[index2+1]-x1[index1]) / \
                (x1[index1+1]-x1[index1])
            y2_new[index] = y22[index2] + y
            index2 += 1
            index += 1
            x_new[index] = x2[index2]
            # and the starting value for the next interval
            y1_new[index] = y21[index2] + y
        else:  # x1[index1+1] == x2[index2+1]:
            y2_new[index] = y12[index1] + y22[index2]
            index1 += 1
            index2 += 1
            index += 1
            x_new[index] = x1[index1]
            y1_new[index] = y11[index1] + y21[index2]
    # one array reached the end -> copy the contents of the other to the end
    if index1+1 < len(y11):
        # compute the linear interpolations values
        y = y21[index2] + (y22[index2]-y21[index2]) * \
            (x1[index1+1:-1]-x2[index2]) / (x2[index2+1]-x2[index2])
        x_new[index+1:index+1+len(x1)-index1-1] = x1[index1+1:]
        y1_new[index+1:index+1+len(y11)-index1-1] = y11[index1+1:]+y
        y2_new[index:index+len(y12)-index1-1] = y12[index1:-1] + y
        index += len(x1)-index1-2
    elif index2+1 < len(y21):
        # compute the linear interpolations values
        y = y11[index1] + (y12[index1]-y11[index1]) * \
            (x2[index2+1:-1]-x1[index1]) / \
            (x1[index1+1]-x1[index1])
        x_new[index+1:index+1+len(x2)-index2-1] = x2[index2+1:]
        y1_new[index+1:index+1+len(y21)-index2-1] = y21[index2+1:] + y
        y2_new[index:index+len(y22)-index2-1] = y22[index2:-1] + y
        index += len(x2)-index2-2
    else:  # both arrays reached the end simultaneously
        # only the last x-value missing
        x_new[index+1] = x1[-1]
    # finally, the end value for the last interval
    y2_new[index] = y12[-1]+y22[-1]
    # only use the data that was actually filled
    return x_new[:index+2], y1_new[:index+1], y2_new[:index+1]
