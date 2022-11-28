
import time
import matplotlib
import gym
import random
import utils
import environments
import numpy as np

from matplotlib import pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map


def run_frozenlake_vi(size):
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

    V = np.zeros(nS)
    policy = np.zeros(nS)

    best_delta = []
    best_V = np.zeros(nS)
    best_won = -1
    best_pol = np.zeros(nS)

    gammas = [0.1, 0.3, 0.4, 0.7, 0.9, 0.99]
    epsilons = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    per_won_hm = np.zeros((len(gammas), len(epsilons)))
    iters_hm = np.zeros((len(gammas), len(epsilons)))
    time_hm = np.zeros((len(gammas), len(epsilons)))

    g_cnt = 0
    e_cnt = 0
    for g in gammas:
        e_cnt = 0
        for e in epsilons:
            print(g, e)
            if g >= 0.99 and e <= 0.001:
                per_won_hm[g_cnt][e_cnt] = 0
                iters_hm[g_cnt][e_cnt] = 0
                time_hm[g_cnt][e_cnt] = 0
            else:
                start = time.time()
                V, delta_vals, iterations, avg_V = utils.value_iteration(env, nA, nS, epsilon=e, gamma=g)

                run_time = time.time() - start
                per_won = utils.run_vi_episodes(env, V, nA, 1000)
                per_won_hm[g_cnt][e_cnt] = per_won
                iters_hm[g_cnt][e_cnt] = iterations
                time_hm[g_cnt][e_cnt] = run_time * 1000
                if per_won > best_won:
                    best_delta = delta_vals
                    best_V = V
                    best_won = per_won
                    best_e = e
                    best_g = g
            e_cnt += 1
        g_cnt += 1


    # Plot Percent Games Won Heatmap
    fig, ax = plt.subplots()

    im, cbar = utils.heatmap(per_won_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="% Games Won")
    texts = utils.annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.savefig('Images/frozenlake_vi_percent_heatmap_' + str(size) + '.png')

    # Plot Iterations Heatmap
    fig, ax = plt.subplots()

    im, cbar = utils.heatmap(iters_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="# of Iterations to Convergence")
    texts = utils.annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images/frozenlake_vi_iter_heatmap' + str(size) + '.png')

    # Plot Run time Heatmap
    fig, ax = plt.subplots()

    im, cbar = utils.heatmap(time_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="Runtime (ms)")
    texts = utils.annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images/frozenlake_vi_runtime_heatmap_' + str(size) + '.png')

    # Plot Delta vs iterations
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('Max Delta', color=color)
    ax1.semilogy(delta_vals, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Avg V', color=color)
    ax2.semilogy(avg_V, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Delta/V vs. Iterations')
    plt.savefig('Images/frozenlake_vi_delta_vals_' + str(size) + '.png')

    print(best_e, best_g, best_won)
    optimal_policy = utils.extract_policy(env, V, gamma=best_g)

    # Plot Optimal state values with directions
    print(V.reshape((size, size)))
    print(optimal_policy.reshape((size, size)))

    utils.plot_values(V, optimal_policy, size)


# run_frozenlake_vi(4)
run_frozenlake_vi(20)
