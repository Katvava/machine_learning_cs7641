import time
import numpy as np
import gym
import random
from matplotlib import pyplot as plt
import utils
import environments


def run_frozenlake_ql(size):
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

    iter_params = [100, 200, 500, 800, 1000, 2000, 5000, 8000, 10000]
    time_arr = []
    rew_arr = []
    V_arr = []

    temp_iter = []
    for i in iter_params:
        temp_iter.append(i)
        start = time.time()
        q_learning = utils.Q_learning(env, num_episodes=i)
        run_time = time.time() - start
        rew = utils.run_q_episodes(env, q_learning, 200) * 100
        tot_V = 0
        for s in range(env.observation_space.n):
            tot_V += q_learning[s][np.argmax(q_learning[s])]
        print(i, rew, tot_V / env.observation_space.n, run_time)
        time_arr.append(run_time)
        rew_arr.append(rew)
        V_arr.append((tot_V))

    # Plot Delta vs iterations
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel('Reward %/Avg V', color=color)
    ax1.plot(temp_iter, rew_arr, color=color, label='Reward %')
    ax1.plot(temp_iter, V_arr, color='darkblue', label='Avg V')
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Time', color=color)
    ax2.plot(temp_iter, time_arr, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('V/Reward/Time vs. Iterations fpr 20x20 FL')
    plt.savefig('Images/frozenlake_ql_20_RunStats_' + str(size) + '.png')

    V = np.zeros(env.observation_space.n)
    P = np.zeros(env.observation_space.n)
    for s in enumerate(q_learning):
        V[s[0]] = s[1][np.argmax(s[1])]
        P[s[0]] = np.argmax(s[1])

    utils.plot_values(V, P, size, 'Images/frozenlake_ql_vals_' + str(size) + '.png')


run_frozenlake_ql(4)
run_frozenlake_ql(20)
