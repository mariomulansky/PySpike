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
def isi_distance_python(s1, s2, t_start, t_end):
    """ Plain Python implementation of the isi distance.
    """
    N1 = len(s1)
    N2 = len(s2)

    # compute the isi-distance
    spike_events = np.empty(N1+N2+2)
    spike_events[0] = t_start
    # the values have one entry less - the number of intervals between events
    isi_values = np.empty(len(spike_events) - 1)
    if s1[0] > t_start:
        # edge correction
        nu1 = max(s1[0] - t_start, s1[1] - s1[0])
        index1 = -1
    else:
        nu1 = s1[1] - s1[0]
        index1 = 0
    if s2[0] > t_start:
        # edge correction
        nu2 = max(s2[0] - t_start, s2[1] - s2[0])
        index2 = -1
    else:
        nu2 = s2[1] - s2[0]
        index2 = 0

    isi_values[0] = abs(nu1 - nu2) / max(nu1, nu2)
    index = 1
    while index1+index2 < N1+N2-2:
        # check which spike is next - from s1 or s2
        if (index1 < N1-1) and (index2 == N2-1 or s1[index1+1] < s2[index2+1]):
            index1 += 1
            spike_events[index] = s1[index1]
            if index1 < N1-1:
                nu1 = s1[index1+1]-s1[index1]
            else:
                # edge correction
                nu1 = max(t_end-s1[N1-1], s1[N1-1]-s1[N1-2])

        elif (index2 < N2-1) and (index1 == N1-1 or
                                  s1[index1+1] > s2[index2+1]):
            index2 += 1
            spike_events[index] = s2[index2]
            if index2 < N2-1:
                nu2 = s2[index2+1]-s2[index2]
            else:
                # edge correction
                nu2 = max(t_end-s2[N2-1], s2[N2-1]-s2[N2-2])

        else:  # s1[index1 + 1] == s2[index2 + 1]
            index1 += 1
            index2 += 1
            spike_events[index] = s1[index1]
            if index1 < N1-1:
                nu1 = s1[index1+1]-s1[index1]
            else:
                # edge correction
                nu1 = max(t_end-s1[N1-1], s1[N1-1]-s1[N1-2])
            if index2 < N2-1:
                nu2 = s2[index2+1]-s2[index2]
            else:
                # edge correction
                nu2 = max(t_end-s2[N2-1], s2[N2-1]-s2[N2-2])
        # compute the corresponding isi-distance
        isi_values[index] = abs(nu1 - nu2) / \
            max(nu1, nu2)
        index += 1
    # the last event is the interval end
    if spike_events[index-1] == t_end:
        index -= 1
    else:
        spike_events[index] = t_end
    # use only the data added above
    # could be less than original length due to equal spike times
    return spike_events[:index + 1], isi_values[:index]


############################################################
# get_min_dist
############################################################
def get_min_dist(spike_time, spike_train, start_index, t_start, t_end):
    """ Returns the minimal distance |spike_time - spike_train[i]|
    with i>=start_index.
    """
    d = abs(spike_time - t_start)
    if start_index < 0:
        start_index = 0
    while start_index < len(spike_train):
        d_temp = abs(spike_time - spike_train[start_index])
        if d_temp > d:
            return d
        else:
            d = d_temp
        start_index += 1
    # finally, check the distance to end time
    d_temp = abs(t_end - spike_time)
    if d_temp > d:
        return d
    else:
        return d_temp


############################################################
# spike_distance_python
############################################################
def spike_distance_python(spikes1, spikes2, t_start, t_end):
    """ Computes the instantaneous spike-distance S_spike (t) of the two given
    spike trains. The spike trains are expected to have auxiliary spikes at the
    beginning and end of the interval. Use the function add_auxiliary_spikes to
    add those spikes to the spike train.
    Args:
    - spikes1, spikes2: ordered arrays of spike times with auxiliary spikes.
    - t_start, t_end: edges of the spike train
    Returns:
    - PieceWiseLinFunc describing the spike-distance.
    """

    # shorter variables
    t1 = spikes1
    t2 = spikes2

    N1 = len(t1)
    N2 = len(t2)

    spike_events = np.empty(N1+N2+2)

    y_starts = np.empty(len(spike_events)-1)
    y_ends = np.empty(len(spike_events)-1)

    t_aux1 = np.zeros(2)
    t_aux2 = np.zeros(2)
    t_aux1[0] = min(t_start, t1[0]-(t1[1]-t1[0]))
    t_aux1[1] = max(t_end, t1[N1-1]+(t1[N1-1]-t1[N1-2]))
    t_aux2[0] = min(t_start, t2[0]-(t2[1]-t2[0]))
    t_aux2[1] = max(t_end, t2[N2-1]+(t2[N2-1]-t2[N2-2]))
    t_p1 = t_start if (t1[0] == t_start) else t_aux1[0]
    t_p2 = t_start if (t2[0] == t_start) else t_aux2[0]

    spike_events[0] = t_start
    if t1[0] > t_start:
        t_f1 = t1[0]
        dt_f1 = get_min_dist(t_f1, t2, 0, t_aux2[0], t_aux2[1])
        dt_p1 = dt_f1
        isi1 = max(t_f1-t_start, t1[1]-t1[0])
        # s1 = dt_p1*(t_f1-t_start)/isi1
        s1 = dt_p1
        index1 = -1
    else:
        dt_p1 = 0.0
        t_f1 = t1[1]
        dt_f1 = get_min_dist(t_f1, t2, 0, t_aux2[0], t_aux2[1])
        isi1 = t1[1]-t1[0]
        s1 = dt_p1
        index1 = 0
    if t2[0] > t_start:
        # dt_p1 = t2[0]-t_start
        t_f2 = t2[0]
        dt_f2 = get_min_dist(t_f2, t1, 0, t_aux1[0], t_aux1[1])
        dt_p2 = dt_f2
        isi2 = max(t_f2-t_start, t2[1]-t2[0])
        # s2 = dt_p2*(t_f2-t_start)/isi2
        s2 = dt_p2
        index2 = -1
    else:
        dt_p2 = 0.0
        t_f2 = t2[1]
        dt_f2 = get_min_dist(t_f2, t1, 0, t_aux1[0], t_aux1[1])
        isi2 = t2[1]-t2[0]
        s2 = dt_p2
        index2 = 0

    y_starts[0] = (s1*isi2 + s2*isi1) / (0.5*(isi1+isi2)**2)
    index = 1

    while index1+index2 < N1+N2-2:
        # print(index, index1, index2)
        if (index1 < N1-1) and (t_f1 < t_f2 or index2 == N2-1):
            index1 += 1
            # first calculate the previous interval end value
            s1 = dt_f1*(t_f1-t_p1) / isi1
            # the previous time now was the following time before:
            dt_p1 = dt_f1
            t_p1 = t_f1    # t_p1 contains the current time point
            # get the next time
            if index1 < N1-1:
                t_f1 = t1[index1+1]
            else:
                t_f1 = t_aux1[1]
            spike_events[index] = t_p1
            s2 = (dt_p2*(t_f2-t_p1) + dt_f2*(t_p1-t_p2)) / isi2
            y_ends[index-1] = (s1*isi2 + s2*isi1) / (0.5*(isi1+isi2)**2)
            # now the next interval start value
            if index1 < N1-1:
                dt_f1 = get_min_dist(t_f1, t2, index2, t_aux2[0], t_aux2[1])
                isi1 = t_f1-t_p1
                s1 = dt_p1
            else:
                dt_f1 = dt_p1
                isi1 = max(t_end-t1[N1-1], t1[N1-1]-t1[N1-2])
                # s1 needs adjustment due to change of isi1
                # s1 = dt_p1*(t_end-t1[N1-1])/isi1
                # Eero's correction: no adjustment
                s1 = dt_p1
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / (0.5*(isi1+isi2)**2)
        elif (index2 < N2-1) and (t_f1 > t_f2 or index1 == N1-1):
            index2 += 1
            # first calculate the previous interval end value
            s2 = dt_f2*(t_f2-t_p2) / isi2
            # the previous time now was the following time before:
            dt_p2 = dt_f2
            t_p2 = t_f2    # t_p1 contains the current time point
            # get the next time
            if index2 < N2-1:
                t_f2 = t2[index2+1]
            else:
                t_f2 = t_aux2[1]
            spike_events[index] = t_p2
            s1 = (dt_p1*(t_f1-t_p2) + dt_f1*(t_p2-t_p1)) / isi1
            y_ends[index-1] = (s1*isi2 + s2*isi1) / (0.5*(isi1+isi2)**2)
            # now the next interval start value
            if index2 < N2-1:
                dt_f2 = get_min_dist(t_f2, t1, index1, t_aux1[0], t_aux1[1])
                isi2 = t_f2-t_p2
                s2 = dt_p2
            else:
                dt_f2 = dt_p2
                isi2 = max(t_end-t2[N2-1], t2[N2-1]-t2[N2-2])
                # s2 needs adjustment due to change of isi2
                # s2 = dt_p2*(t_end-t2[N2-1])/isi2
                # Eero's adjustment: no correction
                s2 = dt_p2
            # s2 is the same as above, thus we can compute y2 immediately
            y_starts[index] = (s1*isi2 + s2*isi1) / (0.5*(isi1+isi2)**2)
        else:  # t_f1 == t_f2 - generate only one event
            index1 += 1
            index2 += 1
            t_p1 = t_f1
            t_p2 = t_f2
            dt_p1 = 0.0
            dt_p2 = 0.0
            spike_events[index] = t_f1
            y_ends[index-1] = 0.0
            y_starts[index] = 0.0
            if index1 < N1-1:
                t_f1 = t1[index1+1]
                dt_f1 = get_min_dist(t_f1, t2, index2, t_aux2[0], t_aux2[1])
                isi1 = t_f1 - t_p1
            else:
                t_f1 = t_aux1[1]
                dt_f1 = dt_p1
                isi1 = max(t_end-t1[N1-1], t1[N1-1]-t1[N1-2])
            if index2 < N2-1:
                t_f2 = t2[index2+1]
                dt_f2 = get_min_dist(t_f2, t1, index1, t_aux1[0], t_aux1[1])
                isi2 = t_f2 - t_p2
            else:
                t_f2 = t_aux2[1]
                dt_f2 = dt_p2
                isi2 = max(t_end-t2[N2-1], t2[N2-1]-t2[N2-2])
        index += 1
    # the last event is the interval end
    if spike_events[index-1] == t_end:
        index -= 1
    else:
        spike_events[index] = t_end
        # the ending value of the last interval
        isi1 = max(t_end-t1[N1-1], t1[N1-1]-t1[N1-2])
        isi2 = max(t_end-t2[N2-1], t2[N2-1]-t2[N2-2])
        s1 = dt_f1  # *(t_end-t1[N1-1])/isi1
        s2 = dt_f2  # *(t_end-t2[N2-1])/isi2
        y_ends[index-1] = (s1*isi2 + s2*isi1) / (0.5*(isi1+isi2)**2)

    # use only the data added above
    # could be less than original length due to equal spike times
    return spike_events[:index+1], y_starts[:index], y_ends[:index]


############################################################
# cumulative_sync_python
############################################################
def cumulative_sync_python(spikes1, spikes2):

    def get_tau(spikes1, spikes2, i, j):
        return 0.5*min([spikes1[i]-spikes1[i-1], spikes1[i+1]-spikes1[i],
                        spikes2[j]-spikes2[j-1], spikes2[j+1]-spikes2[j]])
    N1 = len(spikes1)
    N2 = len(spikes2)
    i = 0
    j = 0
    n = 0
    st = np.zeros(N1 + N2 - 2)
    c = np.zeros(N1 + N2 - 3)
    c[0] = 0
    st[0] = 0
    while n < N1 + N2:
        if spikes1[i+1] < spikes2[j+1]:
            i += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j)
            st[n] = spikes1[i]
            if spikes1[i]-spikes2[j] > tau:
                c[n] = c[n-1]
            else:
                c[n] = c[n-1]+1
        elif spikes1[i+1] > spikes2[j+1]:
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j)
            st[n] = spikes2[j]
            if spikes2[j]-spikes1[i] > tau:
                c[n] = c[n-1]
            else:
                c[n] = c[n-1]+1
        else:   # spikes1[i+1] = spikes2[j+1]
            j += 1
            i += 1
            if i == N1-1 or j == N2-1:
                break
            n += 1
            st[n] = spikes1[i]
            c[n] = c[n-1]
            n += 1
            st[n] = spikes1[i]
            c[n] = c[n-1]+1
    c[0] = 0
    st[0] = spikes1[0]
    st[-1] = spikes1[-1]

    return st, c


############################################################
# coincidence_python
############################################################
def coincidence_python(spikes1, spikes2, t_start, t_end, max_tau):

    def get_tau(spikes1, spikes2, i, j, max_tau):
        m = t_end - t_start   # use interval as initial tau
        if i < len(spikes1)-1 and i > -1:
            m = min(m, spikes1[i+1]-spikes1[i])
        if j < len(spikes2)-1 and j > -1:
            m = min(m, spikes2[j+1]-spikes2[j])
        if i > 0:
            m = min(m, spikes1[i]-spikes1[i-1])
        if j > 0:
            m = min(m, spikes2[j]-spikes2[j-1])
        m *= 0.5
        if max_tau > 0.0:
            m = min(m, max_tau)
        return m

    N1 = len(spikes1)
    N2 = len(spikes2)
    i = -1
    j = -1
    n = 0
    st = np.zeros(N1 + N2 + 2)  # spike times
    c = np.zeros(N1 + N2 + 2)   # coincidences
    mp = np.ones(N1 + N2 + 2)   # multiplicity
    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            st[n] = spikes1[i]
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # both get marked with 1
                c[n] = 1
                c[n-1] = 1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            st[n] = spikes2[j]
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # both get marked with 1
                c[n] = 1
                c[n-1] = 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            n += 1
            # add only one event, but with coincidence 2 and multiplicity 2
            st[n] = spikes1[i]
            c[n] = 2
            mp[n] = 2

    st = st[:n+2]
    c = c[:n+2]
    mp = mp[:n+2]

    st[0] = t_start
    st[len(st)-1] = t_end
    if N1 + N2 > 0:
        c[0] = c[1]
        c[len(c)-1] = c[len(c)-2]
        mp[0] = mp[1]
        mp[len(mp)-1] = mp[len(mp)-2]
    else:
        c[0] = 1
        c[1] = 1

    return st, c, mp


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


############################################################
# add_discrete_function_python
############################################################
def add_discrete_function_python(x1, y1, mp1, x2, y2, mp2):

    x_new = np.empty(len(x1) + len(x2))
    y_new = np.empty_like(x_new)
    mp_new = np.empty_like(x_new)
    x_new[0] = x1[0]
    index1 = 0
    index2 = 0
    index = 0
    while (index1+1 < len(y1)) and (index2+1 < len(y2)):
        if x1[index1+1] < x2[index2+1]:
            index1 += 1
            index += 1
            x_new[index] = x1[index1]
            y_new[index] = y1[index1]
            mp_new[index] = mp1[index1]
        elif x1[index1+1] > x2[index2+1]:
            index2 += 1
            index += 1
            x_new[index] = x2[index2]
            y_new[index] = y2[index2]
            mp_new[index] = mp2[index2]
        else:  # x1[index1+1] == x2[index2+1]
            index1 += 1
            index2 += 1
            index += 1
            x_new[index] = x1[index1]
            y_new[index] = y1[index1] + y2[index2]
            mp_new[index] = mp1[index1] + mp2[index2]
    # one array reached the end -> copy the contents of the other to the end
    if index1+1 < len(y1):
        x_new[index+1:index+1+len(x1)-index1-1] = x1[index1+1:]
        y_new[index+1:index+1+len(x1)-index1-1] = y1[index1+1:]
        mp_new[index+1:index+1+len(x1)-index1-1] = mp1[index1+1:]
        index += len(x1)-index1-1
    elif index2+1 < len(y2):
        x_new[index+1:index+1+len(x2)-index2-1] = x2[index2+1:]
        y_new[index+1:index+1+len(x2)-index2-1] = y2[index2+1:]
        mp_new[index+1:index+1+len(x2)-index2-1] = mp2[index2+1:]
        index += len(x2)-index2-1
    # else:  # both arrays reached the end simultaneously
    #     x_new[index+1] = x1[-1]
    #     y_new[index+1] = y1[-1] + y2[-1]
    #     mp_new[index+1] = mp1[-1] + mp2[-1]

    y_new[0] = y_new[1]
    mp_new[0] = mp_new[1]

    # the last value is again the end of the interval
    # only use the data that was actually filled
    return x_new[:index+1], y_new[:index+1], mp_new[:index+1]
