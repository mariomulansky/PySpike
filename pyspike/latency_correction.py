import numpy as np
import pyspike as spk
import matplotlib.pyplot as plt

def Spike_time_difference_matrix(spike_trains):
    num_trains = len(spike_trains)

    try:
        from pyspike.plotting import Multi_Profile
    except ImportError:
        raise ImportError("Error: Could not import Multi_profile from pyspike.plotting.")
    
    indices = np.arange(num_trains)
    assert (indices < num_trains).all() and (indices >= 0).all(),"Invalid index list."
    pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]
    matrix = np.zeros((num_trains,num_trains))
    for i, j in pairs:
        [times, e_values] = Multi_Profile([spike_trains[i],spike_trains[j]], 3)
        value = 0
        k = 0
        num_coin = 0
        while k < len(e_values)-1:
            if e_values[k] == -1:
                value += times[k]-times[k+1]
                num_coin += 1
                k += 2
            elif e_values[k] == 1:
                value += times[k+1]-times[k]
                num_coin += 1
                k += 2
            else:
                k += 1
        for k in range(len(times)-1):
            if times[k] == times[k+1]:
                num_coin += 1
        if num_coin == 0:
            matrix[i][j] = 0
            matrix[j][i] = 0
        else:
            matrix[i][j] = value/num_coin
            matrix[j][i] = value/num_coin
    return matrix

