import numpy as np
import random

def generate_surro(sto_profs, num_surros):

    num_pairs = sto_profs.shape[0]
    num_trains = int((1 + np.sqrt(1 + 8 * num_pairs)) / 2)
    firsts, seconds = np.where(np.triu(np.ones((num_trains, num_trains)), 1))

    pair_indies, spike_indies  = np.where(sto_profs != 0)
    values = sto_profs[pair_indies, spike_indies]
    leader_pos = spike_indies[::2]
    follower_pos = spike_indies[1::2]
    pair = pair_indies[::2]
    coins = values[::2]
    num_coins = len(pair)
    num_spikes = sto_profs.shape[1]


    indies = np.array([pair, firsts[pair] * (coins == 1) + seconds[pair] * (coins == -1), seconds[pair] * (coins == 1) + firsts[pair] * (coins == -1), leader_pos, follower_pos])

    num_swaps = num_spikes  # eliminate transients !!!!!
    synf = np.zeros(num_surros)
    synf_norm = np.zeros(num_surros)
    for suc in range(num_surros):
        if suc == 1:
            num_swaps = round(num_swaps / 2)

        indies, error_count = Spike_Order_surro(indies, firsts, seconds, num_swaps)

        surro_sto_profs = np.zeros_like(sto_profs)
        for cc in range(num_coins):
            surro_sto_profs[indies[0, cc], indies[3:5, cc]] = (indies[1, cc] < indies[2, cc]) - (indies[1, cc] > indies[2, cc]) * np.ones(2)

        surro_mat_entries = np.sum(surro_sto_profs, axis=1) / 2
        surro_mat = np.tril(np.ones((num_trains, num_trains)), -1)
        surro_mat[np.where(surro_mat)] = surro_mat_entries
        surro_mat = surro_mat.T - surro_mat
        
        st_indy_simann, synf[suc], total_iter = Spike_Order_sim_ann(surro_mat)
        
        synf_norm[suc] = synf[suc]*2 / ((num_trains-1)*len(sto_profs[0]))
        
        """
        try:
            from pyspike.cython.cython_simulated_annealing import sim_ann_cython
        except ImportError:
            pyspike.NoCythonWarn()
        st_indy_simann, synf[suc - 1], total_iter = sim_ann_cython(surro_mat, T_start, T_end, alpha)
        """
        if suc == num_surros:
            pass  # print results if needed
    return synf_norm


def Spike_Order_surro(indies1, firsts, seconds, num_swaps):
    num_pairs = firsts.shape[0]
    num_coins = indies1.shape[1]
    
    error_count = 0
    sc = 0
    while sc < num_swaps:
        indies2 = indies1
        brk = False
        coin = random.randint(0, num_coins-1)
        
        train1 = indies1[1, coin]
        train2 = indies1[2, coin]
        pos1 = indies1[3, coin]
        pos2 = indies1[4, coin]
        
        fi11 = np.where(indies1[3,:] == pos1)[0]
        fi21= np.where(indies1[4,:] == pos1)[0]
        fi12= np.where(indies1[3,:] == pos2)[0]
        fi22= np.where(indies1[4,:] == pos2)[0]
        fiu = np.unique(np.concatenate((fi11, fi21, fi12, fi22)))

        indies1[1, fi11] = train2
        indies1[2, fi21] = train2
        indies1[1, fi12] = train1
        indies1[2, fi22] = train1

        for fc in fiu:
            new_trains = np.sort(indies1[1:3, fc])
            for i in range(len(firsts)):
                if firsts[i] == new_trains[0] and seconds[i] == new_trains[1]:
                    indies1[0, fc] = i
                    break
        for fc in fiu:
            sed = np.setdiff1d(np.where(indies1[0, :] == indies1[0, fc])[0], fc)
            for sedc in range(len(sed)):
                if len(np.intersect1d(indies1[3:5, sed[sedc]], indies1[3:5, fc])) > 0:
                    error_count += 1
                    indies1 = indies2
                    brk = 1
                    break
            if brk:
                break
        if brk:
            if error_count <= num_coins:
                continue
            else:
                sc = num_swaps
        sc += 1
    return indies1, error_count

def Spike_Order_sim_ann(D):
    N = D.shape[0]
    A = np.sum(np.triu(D, 1))
    p = np.arange(N)
    T = 2 * np.max(D)
    T_end = 1e-5 * T
    alpha = 0.9
    total_iter = 0

    while T > T_end:
        iterations = 0
        succ_iter = 0
        while iterations < 100 * N and succ_iter < 10 * N:
            ind1 = np.random.randint(0, N-1)
            delta_A = -2 * D[p[ind1], p[ind1+1]]
            if delta_A > 0 or np.exp(delta_A / T) > np.random.rand():
                p[ind1], p[ind1+1] = p[ind1+1], p[ind1]
                A += delta_A
                succ_iter += 1
            iterations += 1

        total_iter += iterations
        T *= alpha
        if succ_iter == 0:
            break
    return p, A, total_iter