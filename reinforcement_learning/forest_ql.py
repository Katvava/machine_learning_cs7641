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


def run_forest_ql(size):
    seed_val = 128
    np.random.seed(seed_val)
    random.seed(seed_val)

    S = size
    r1 = 20  # reward when action ‘Wait’ is performed
    r2 = 10  # reward when action ‘Cut’ is performed
    p = 0.1

    P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)

    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    best_pol_arr = []

    for g in gammas:
        qlearning = mdp.QLearning(P, R, g, n_iter=10000)
        qlearning.run()
        best_pol_arr.append(list(qlearning.policy))

    v = [i["Mean V"] for i in qlearning.run_stats]

    # Plot out optimal policy
    utils.plot_optimal_policy(best_pol_arr[::-1], gammas,
                              'Images/forest_ql_policy_heatmap_' + str(size) + '.png')
    utils.plot_v_iteration(v, 'Images/forest_ql_v_iteration_' + str(size) + '.png')


# run experiments
# run_forest_ql(10)
run_forest_ql(50)
