import matplotlib.pyplot as plt
import pyspike as spk
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pyspike.generate_surrogate import generate_surro


def Multi_Profile(spike_trains, variable):
    """
    Computes and returns the multivariate profile of the given spike trains based on the specified variable.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param variable: The type of data represented by the profile:
                     1 for Spike-Synchro,
                     2 for Spike Order,
                     3 for Spike train order.
    :type variable: int
    :returns: Tuple containing the spike times and the profile values for each spike.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    if variable == 1:
        try:
            from pyspike.spike_sync import spike_sync_profile as prof
        except ImportError:
            raise ImportError("Error: Could not import spike_sync_profile from pyspike.spike_sync.")
    elif variable == 2:
        try:
            from pyspike.spike_order import spike_order_values as prof
        except ImportError:
            raise ImportError("Error: Could not import spike_order_values from pyspike.spike_order.")
    elif variable == 3:
        try:
            from pyspike.spike_order import spike_train_order_profile as prof
        except ImportError:
            raise ImportError("Error: Could not import spike_train_order_profile from pyspike.spike_order.")
    else:
        raise ValueError("Error: variable must be 1, 2, or 3.")
    
    spike_time = []
    for i in range(len(spike_trains)):
        for j in range(len(spike_trains[i])):
            spike_time.append(spike_trains[i][j])

    if variable == 2:
        prof_without = prof(spike_trains)
        prof_values = [item for sublist in prof_without for item in sublist]
        combined = list(zip(spike_time, prof_values))
        combined_sorted = sorted(combined, key=lambda x: x[0])
        spike_time, prof_values = zip(*combined_sorted)
    else:
        prof_without = prof(spike_trains).get_plottable_data()[1]
        prof_values = list(prof_without[1:-1])
        spike_time.sort()

        L = []
        k = 1
        for i in range(1,len(spike_time)):
            if spike_time[i-1] == spike_time[i]:
                k += 1
            else:
                L.append(k)
                k = 1
        L.append(k)
        k = 0
        for i in range(len(prof_values)):
            for j in range(L[i]-1):
                prof_values.insert(i+k, prof_values[i+k])
                k += 1

        """
        L = []
        i = 0
        while i < (len(spike_time)-1):
            k = 0
            if spike_time[i+k-1] == spike_time[i+k]:
                while spike_time[i+k-1] == spike_time[i+k] and i+k != len(spike_time)-1:
                    k += 1
                L.append([i-1,k])
            i += k + 1
        if len(prof_values) > 1 and spike_time[-2] == spike_time[-1]:
            L.append([len(spike_time)-2,1])

        for i, k in L:
            for j in range(1,k+1):
                prof_values.insert(i, prof_values[i])
        """
    return np.array(spike_time), np.array(prof_values)
    
def Multi_Profile_Matrix(spike_trains, variable):
    """
    Computes and returns a matrix representing the profiles of spike trains.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param variable: The type of data represented by the profile:
                     1 for Spike-Synchro,
                     2 for Spike Order,
                     3 for Spike train order.
    :type variable: int
    :returns: A matrix containing the profile values for each pair of spike trains.
    :rtype: numpy.ndarray
    """
    all_trains = f_all_trains(spike_trains)[0]
    indices = np.arange(len(spike_trains))
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(),"Invalid index list."
    pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]
    num_pairs = len(pairs)
    num_spikes = len(all_trains)
    Mat = np.zeros((num_pairs, num_spikes))
    
    pairscount = 0
    for i, j in pairs:
        bi_spike_trains = [spike_trains[i], spike_trains[j]]
        sto_prof_bi = Multi_Profile(bi_spike_trains, variable)[1]
        sto_prof_bi_count = 0
        for k in range(num_spikes):
            if all_trains[k] == i+1 or all_trains[k] == j+1:
                Mat[pairscount][k] = sto_prof_bi[sto_prof_bi_count]
                sto_prof_bi_count += 1
        pairscount += 1
    return Mat
    
def f_all_trains(spikes):
    """
    Flattens and pools all spike trains into a single list while maintaining the association of spikes with their original trains.

    :param spikes: List of spike trains.
    :type spikes: List of :class:`pyspike.SpikeTrain`
    :returns: A tuple containing two lists:
              - all_trains: List indicating the original train for each spike in the pooled list.
              - pooled: List of all spike times from all trains, sorted in ascending order.
    :rtype: tuple of lists
    """
    num_trains = len(spikes)
    num_spikes_per_train = [len(train) for train in spikes]
    dummy = [0] + num_spikes_per_train
    all_indy = [0] * sum(num_spikes_per_train)
   
    for trc in range(num_trains):
        start_idx = sum(dummy[0:trc+1])
        end_idx = start_idx + num_spikes_per_train[trc]
        all_indy[start_idx:end_idx] = [trc+1] * num_spikes_per_train[trc]
   
    sp_flat = [spike for train in spikes for spike in train]
    sp_indy = sorted(range(len(sp_flat)), key=lambda i: sp_flat[i])
    all_trains = [all_indy[idx] for idx in sp_indy]
    pooled = [sp_flat[idx] for idx in sp_indy]
   
    return all_trains, pooled

def plot_matrix(matrix, variable, variable_value=None, showing=0):
    """
    Plots a matrix representing spike train data.

    :param matrix: The matrix to be plotted.
    :type matrix: numpy.ndarray
    :param variable: The type of data represented by the matrix:
                     1 for ISI-distance,
                     2 for SPIKE-distance,
                     3 for Spike-Synchro,
                     4 for SPIKE-Order,
                     5 for Spike train order.
    :type variable: int
    :param variable_value: The value associated with the variable. If None, it's not displayed.
    :type variable_value: float or None
    :param showing: Determines whether to print the matrix before plotting (0 or 1).
    :type showing: int, optional
    """

    possible_variable = ['ISI-distance', 'SPIKE-distance', 'Spike-Synchro', 'SPIKE-Order', 'Spike train order']
    
    num_trains = len(matrix)
    if showing == 1:
        print(f"\n{variable}:")
        for i in range(num_trains):
            print("\n%i     " % (i+1), end = "")
            for j in range(num_trains):
                print("%.8f " % (matrix[i][j]), end = "")
        print("\n")

    plt.figure(figsize=(17, 10), dpi=80)
    plt.imshow(matrix, interpolation='none')
    if variable_value is not None:
        plt.title("%s Matrix (%s = %.8f)" % (possible_variable[variable-1], possible_variable[variable-1], variable_value), color='k', fontsize=24)
    else:
        plt.title("%s Matrix" % possible_variable[variable-1], color='k', fontsize=24)
    plt.xlabel('Spike Trains', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.xticks(np.arange(num_trains),np.arange(num_trains)+1, fontsize=14)
    plt.yticks(np.arange(num_trains),np.arange(num_trains)+1, fontsize=14)
    plt.jet()
    plt.colorbar()
    plt.show()

def plot_profile(x_prof, y_prof, variable, variable_value=None, showing=0):
    """
    Plots a profile representing spike train data.

    :param x_prof: The x-values of the profile.
    :type x_prof: list or numpy.ndarray
    :param y_prof: The y-values of the profile.
    :type y_prof: list or numpy.ndarray
    :param variable: The type of data represented by the profile:
                     1 for ISI-distance,
                     2 for SPIKE-distance,
                     3 for Spike-Synchro,
                     4 for SPIKE-Order,
                     5 for Spike train order.
    :type variable: int
    :param variable_value: The value associated with the variable. If None, it's not displayed.
    :type variable_value: float or None
    :param showing: Determines whether to print the profile before plotting (0 or 1).
    :type showing: int, optional
    """

    possible_variable = ['ISI-distance', 'SPIKE-distance', 'Spike-Synchro', 'SPIKE-Order', 'Spike train order']
    tmin = x_prof[0]
    tmax = x_prof[-1]

    if showing == 1:
        print(f"\n{variable}: %.8f\n" % (variable_value))
        print("\nSPIKE-train-Order-Profile:\n")
        print("x            y\n")
        for i in range(len(x_prof)):
            print("%.8f   %.8f\n" % (x_prof[i], y_prof[i]), end = "")
        print("\n")

    plt.figure(figsize=(17, 10), dpi=80)
    plt.plot(x_prof, y_prof, '-k*')
    
    if variable == 4 or variable == 5:
        plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -1.05, 1.05])
        plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (-1, -1), ':', color='k', linewidth=1)
        plt.plot((tmin, tmin), (-1, 1), ':', color='k', linewidth=1)
        plt.plot((tmax, tmax), (-1, 1), ':', color='k', linewidth=1)
    else:
        plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
        plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
        plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
        plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (variable_value, variable_value), '--', color='k', linewidth=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if variable_value is not None:
        plt.title("%s Profile (%s = %.8f )" % (possible_variable[variable-1], possible_variable[variable-1], variable_value), color='k', fontsize=24)
    else:
        plt.title("%s Profile" % possible_variable[variable-1], color='k', fontsize=24)
    
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel("%s" %(possible_variable[variable-1]), color='k', fontsize=18)
    plt.show()

def plot_spike_trains(spikes, phi=None, showing=0, order_color=0):
    """
    Plots spike trains in a raster plot.

    :param spikes: List of spike trains.
    :type spikes: List of :class:`pyspike.SpikeTrain`
    :param phi: Order of spike trains to be plotted, if None, default order is used.
    :type phi: list or None, optional
    :param showing: Determines whether to print the spike trains before plotting (0 or 1).
    :type showing: int, optional
    :param order_color: Determines whether to plot spike trains with color based on their order (0 or 1).
    :type order_color: int, optional
    """
    tmin = min([spikes[i][0] for i in range(len(spikes))])
    tmax = max([spikes[i][-1] for i in range(len(spikes))])
    num_trains = len(spikes)

    if showing == 1:
        for i in range(num_trains):
            print("\nSpike Train %3i:" % (i+1))
            for j in range(len(spikes[i])):
                print("%i %.8f" % (j+1, spikes[i][j]))
        print("\n")

    if order_color == 1:
        D = spk.spike_order_values(spikes)
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
        positions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
        cm = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)), N=256)
        fig, ax = plt.subplots(figsize=(17, 10), dpi=80)
        sm = plt.cm.ScalarMappable(cmap=cm)
        plt.colorbar(sm, ax=ax)

        C = Multi_Profile(spikes, 1)[1]
        order = f_all_trains(spikes)[0]

        if phi == None:
            indexed_liste1 = list(enumerate(order))
            zipped_lists = list(zip(indexed_liste1, C))
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: (x[0][1], x[0][0]))
            _, sorted_C = zip(*sorted_zipped_lists)
        else:
            zipped_lists = list(zip(order, C))
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: phi[x[0] - 1])
            _, sorted_C = zip(*sorted_zipped_lists)
            sorted_C = list(sorted_C)

    else:
        plt.figure(figsize=(17, 10), dpi=80)
    plt.title("Rasterplot", color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
    plt.xticks(fontsize=14)

    if phi==None:
        plt.yticks(np.arange(num_trains)+1, np.arange(num_trains,0,-1), fontsize=14)
    else:
        plt.yticks(np.arange(num_trains)+1, reversed([x+1 for x in phi]), fontsize=14)

    plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1)

    N = 0
    if phi == None:
        for i in range(num_trains):
            for j in range(len(spikes[i])):
                if order_color == 1:
                    color = cm((D[i][j] + 1) / 2)
                    if sorted_C[N] == 0:
                        plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                    else:
                        plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color=color, linewidth=1+2*sorted_C[N])
                else:
                    plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                N += 1
    else:
        for i in range(num_trains):
            for j in range(len(spikes[phi[i]])):
                if order_color == 1:
                    color = cm((D[phi[i]][j] + 1) / 2)
                    if sorted_C[N] == 0:
                        plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                    else:
                        plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color=color, linewidth=1+2*sorted_C[N])
                else:
                    plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                N += 1
    plt.show()

def plot_spike_trainsV2(spikes, phi=None, showing=0, order_color=0):
    """
    Plots spike trains in a raster plot.

    :param spikes: List of spike trains.
    :type spikes: List of :class:`pyspike.SpikeTrain`
    :param phi: Order of spike trains to be plotted, if None, default order is used.
    :type phi: list or None, optional
    :param showing: Determines whether to print the spike trains before plotting (0 or 1).
    :type showing: int, optional
    :param order_color: Determines whether to plot spike trains with color based on their order (0 or 1).
    :type order_color: int, optional
    """
    tmin = min([spikes[i][0] for i in range(len(spikes))])
    tmax = max([spikes[i][-1] for i in range(len(spikes))])
    num_trains = len(spikes)

    if showing == 1:
        for i in range(num_trains):
            print("\nSpike Train %3i:" % (i+1))
            for j in range(len(spikes[i])):
                print("%i %.8f" % (j+1, spikes[i][j]))
        print("\n")

    if order_color == 1:
        D = spk.spike_order_values(spikes)
        flat_D = [item for sublist in D for item in sublist]
        min_D, max_D = np.min(flat_D), np.max(flat_D)
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
        positions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
        cm = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)), N=256)
        fig, ax = plt.subplots(figsize=(17, 10), dpi=80)
        norm = plt.Normalize(vmin=min_D, vmax=max_D)

        C = Multi_Profile(spikes, 1)[1]
        order = f_all_trains(spikes)[0]

        if phi == None:
            indexed_liste1 = list(enumerate(order))
            zipped_lists = list(zip(indexed_liste1, C))
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: (x[0][1], x[0][0]))
            _, sorted_C = zip(*sorted_zipped_lists)
        else:
            zipped_lists = list(zip(order, C))
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: phi[x[0] - 1])
            _, sorted_C = zip(*sorted_zipped_lists)
            sorted_C = list(sorted_C)

    else:
        plt.figure(figsize=(17, 10), dpi=80)
    plt.title("Rasterplot", color='k', fontsize=24)
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
    plt.xticks(fontsize=14)

    if phi==None:
        plt.yticks(np.arange(num_trains)+1, np.arange(num_trains,0,-1), fontsize=14)
    else:
        plt.yticks(np.arange(num_trains)+1, reversed([x+1 for x in phi]), fontsize=14)

    plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1)
    N = 0
    
    if phi == None:
        for i in range(num_trains):
            for j in range(len(spikes[i])):
                if order_color == 1:
                    color = cm((D[i][j] + 1) / 2)
                    if sorted_C[N] == 0:
                        plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                    else:
                        plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color=color, linewidth=1+2*sorted_C[N])
                else:
                    plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                N += 1
    else:
        for i in range(num_trains):
            for j in range(len(spikes[phi[i]])):
                if order_color == 1:
                    color = cm((D[phi[i]][j] + 1) / 2)
                    if sorted_C[N] == 0:
                        plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                    else:
                        plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color=color, linewidth=1+2*sorted_C[N])
                else:
                    plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1)
                N += 1
    colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
    up = int(((max_D+1)/2)*8)+1
    down = int(((min_D+1)/2)*8)-1
    colors = colors[down:up]
    positions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    positions = positions[down+1:up-1]
    positions.append(1.0)
    positions = [0.0] + positions
    print(down)
    print(up)
    print(colors)
    print(positions)
    cm = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)), N=256)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks([min_D, (min_D + max_D) / 2, max_D])
    plt.show()
    
def plot_surrogates(spike_trains, num_surros):
    """
    Plots the surrogate distribution of spike train order values and the optimal spike train order value.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param num_surros: Number of surrogates to generate.
    :type num_surros: int
    """
    sto_profs = Multi_Profile_Matrix(spike_trains, 3)
    values = generate_surro(sto_profs, num_surros)

    phi, _ = spk.optimal_spike_train_sorting(spike_trains)
    F_opt = spk.spike_train_order_value(spike_trains, indices=phi)

    num_interval = int(num_surros*2)
    interval = 1/num_interval
    count = [0 for i in range(num_interval)]
    
    for i in range(len(values)):
        N = int(values[i]/interval)
        count[N] += 1

    mean_value = np.mean(values)
    std_dev = np.std(values)
    max_count = max(count)

    plt.figure(figsize=(10, 6))
    plt.xlim(0, 1)
    plt.ylim(0, max_count)
    plt.yticks(np.arange(0, max_count + 2, 1))

    for i in range(len(count)):
        plt.vlines(x=i * interval, ymin=0, ymax=count[i], color='red', linestyle='-', linewidth=1)
    plt.axvline(x=F_opt, color='black', linestyle='--', linewidth=2, label='F_s')

    plt.axvline(x=mean_value, color='r', linestyle='-', linewidth=3, label='Mean')
    plt.hlines(y=max_count * 0.8, xmin=mean_value - std_dev, xmax=mean_value + std_dev, color='r', linestyle='-', linewidth=3)

    plt.xlabel('F', fontsize=18)
    plt.ylabel('#', fontsize=18)

    z = (F_opt - mean_value)/std_dev
    if F_opt >= max(values):
        if num_surros == 9:
            plt.title("z = %.8f ; p = 0.1*" %(z), color='k', fontsize=24)
        if num_surros == 19:
            plt.title("z = %.8f ; p = 0.05**" %(z), color='k', fontsize=24)
        if num_surros == 999:
            plt.title("z = %.8f ; p = 0.001***" %(z), color='k', fontsize=24)
        else:
            p = 1/(num_surros+1)
            plt.title("z = %.8f ; p = %.8f" %(z, p), color='k', fontsize=24)
    else:
        p = 1/(num_surros+1)
        plt.title("z = %.8f ; p >> %.8f" %(z, p), color='k', fontsize=24)
    plt.legend()
    plt.show()
