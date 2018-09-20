""" directionality_python_backend.py

Collection of python functions that can be used instead of the cython
implementation.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

import numpy as np


############################################################
# spike_train_order_python
############################################################
def spike_directionality_profile_python(spikes1, spikes2, t_start, t_end,
                                        max_tau):

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
    d1 = np.zeros(N1)   # directionality values
    d2 = np.zeros(N2)   # directionality values
    while i + j < N1 + N2 - 2:
        if (i < N1-1) and (j == N2-1 or spikes1[i+1] < spikes2[j+1]):
            i += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            if j > -1 and spikes1[i]-spikes2[j] < tau:
                # coincidence between the current spike and the previous spike
                # spike in first spike train occurs after second
                d1[i] = -1
                d2[j] = +1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # spike in second spike train occurs after first
                d1[i] = +1
                d2[j] = -1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            d1[i] = 0
            d2[j] = 0

    return d1, d2


############################################################
# spike_train_order_python
############################################################
def spike_train_order_profile_python(spikes1, spikes2, t_start, t_end,
                                     max_tau):

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
    a = np.zeros(N1 + N2 + 2)   # coincidences
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
                a[n] = -1
                a[n-1] = -1
        elif (j < N2-1) and (i == N1-1 or spikes1[i+1] > spikes2[j+1]):
            j += 1
            n += 1
            tau = get_tau(spikes1, spikes2, i, j, max_tau)
            st[n] = spikes2[j]
            if i > -1 and spikes2[j]-spikes1[i] < tau:
                # coincidence between the current spike and the previous spike
                # both get marked with 1
                a[n] = 1
                a[n-1] = 1
        else:   # spikes1[i+1] = spikes2[j+1]
            # advance in both spike trains
            j += 1
            i += 1
            n += 1
            # add only one event with zero asymmetry value and multiplicity 2
            st[n] = spikes1[i]
            a[n] = 0
            mp[n] = 2

    st = st[:n+2]
    a = a[:n+2]
    mp = mp[:n+2]

    st[0] = t_start
    st[len(st)-1] = t_end
    if N1 + N2 > 0:
        a[0] = a[1]
        a[len(a)-1] = a[len(a)-2]
        mp[0] = mp[1]
        mp[len(mp)-1] = mp[len(mp)-2]
    else:
        a[0] = 1
        a[1] = 1

    return st, a, mp
