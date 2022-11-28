# coding: utf-8

import mdptoolbox.example
import numpy as np
import matplotlib
import matplotlib.patches
import random
import utils

from hiive.mdptoolbox import mdp
from matplotlib import pyplot as plt
from matplotlib import colors


def run_forest_pi(size):
    seed_val = 128
    np.random.seed(seed_val)
    random.seed(seed_val)

    S = size
    r1 = 20  # reward when action ‘Wait’ is performed
    r2 = 10  # reward when action ‘Cut’ is performed
    p = 0.1

    P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    iters_hm = np.zeros((len(gammas), 1))
    time_hm = np.zeros((len(gammas), 1))

    # best_reward = -1
    best_pol_arr = []
    iter = 0

    for g in gammas:
        best_pol = []
        best_rew = -1

        pi = mdp.PolicyIteration(P, R, g)
        pi.run()
        reward = utils.run_episodes(pi.policy, S, R, p, 1000, 100)
        if reward > best_rew:
            # best_reward = reward
            best_pol = pi.policy
        iters_hm[iter][0] = pi.iter
        time_hm[iter][0] = pi.time * 1000
        best_pol_arr.append(list(best_pol))
        iter += 1

    v = [i["Mean V"] for i in pi.run_stats]
    error = [i["Error"] for i in pi.run_stats]
    reward = [i["Reward"] for i in pi.run_stats]

    utils.plot_stats(v, error, reward,
                     'Images/forest_pi_v_iteration_' + str(size) + '.png')
    utils.plot_iteration_heatmap(iters_hm[:,0],
                                 'Images/forest_pi_g_iteration_' + str(size) + '.png')
    utils.plot_optimal_policy(best_pol_arr[::-1], gammas,
                              'Images/forest_pi_policy_heatmap_' + str(size) + '.png')


# run experiments
# run_forest_pi(10)
run_forest_pi(50)
