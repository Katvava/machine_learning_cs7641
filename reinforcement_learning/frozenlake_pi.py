import numpy as np
import gym
from matplotlib import pyplot as plt
import time
import matplotlib
from gym.envs.toy_text.frozen_lake import generate_random_map
import random
import utils
import environments


def run_frozenlake_pi(size):
    seed_val = 128
    np.random.seed(seed_val)
    random.seed(seed_val)
    if size == 4:
        env = gym.make("FrozenLake-v1")
    elif size == 20:
        env = environments.get_large_rewarding_no_reward_frozen_lake_environment()
    else:
        print("size {} is not supported".format(size))
        exit(0)
    env.seed(seed_val)
    env.reset()
    env = env.unwrapped

    nA = env.action_space.n
    nS = env.observation_space.n

    best_V = ''
    best_won = -99999
    best_policy = []

    gammas = [0.1, 0.3, 0.4, 0.7, 0.9, 0.99]
    epsilons = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    per_won_hm = np.zeros((len(gammas), len(epsilons)))
    iters_hm = np.zeros((len(gammas), len(epsilons)))
    time_hm = np.zeros((len(gammas), len(epsilons)))

    g_cnt = 0
    e_cnt = 0
    best_e = 0
    best_g = 0
    for g in gammas:
        e_cnt = 0
        for e in epsilons:
            if g >= 0.99 and e <= 0.001:
                per_won_hm[g_cnt][e_cnt] = 0
                iters_hm[g_cnt][e_cnt] = 0
                time_hm[g_cnt][e_cnt] = 0
            else:
                start = time.time()
                V = np.zeros(nS)
                policy = np.zeros(nS)

                V, policy, iter = utils.policy_iteration(env, V,  policy, nS, nA, e, g)
                run_time = time.time() - start
                per_won = utils.run_pi_episodes(env, V, policy)

                per_won_hm[g_cnt][e_cnt] = per_won
                iters_hm[g_cnt][e_cnt] = iter
                time_hm[g_cnt][e_cnt] = run_time * 1000
                if per_won > best_won:
                    best_e = e
                    best_g = g
                    best_V = V
                    best_policy = policy
                    best_won = per_won
            e_cnt += 1
        g_cnt += 1

    # Plot Percent Games Won Heatmap
    fig, ax = plt.subplots()
    im, cbar = utils.heatmap(per_won_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="% Games Won")
    texts = utils.annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.savefig('Images/frozenlake_pi_percent_heatmap_' + str(size) + '.png')

    # Plot Iterations Heatmap
    fig, ax = plt.subplots()
    im, cbar = utils.heatmap(iters_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="# of Iterations to Convergence")
    texts = utils.annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images/frozenlake_pi_iter_heatmap_' + str(size) + '.png')

    # Plot Run time Heatmap
    fig, ax = plt.subplots()
    im, cbar = utils.heatmap(time_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="Runtime (ms)")
    texts = utils.annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images/frozenlake_pi_runtime_heatmap' + str(size) + '.png')

    # Plot Optimal state values with directions
    utils.plot_values(V, best_policy, size)

    print(best_V.reshape((size, size)))
    print(best_policy.reshape((size, size)))
    print(best_e, best_g, best_won)


run_frozenlake_pi(4)
run_frozenlake_pi(20)
