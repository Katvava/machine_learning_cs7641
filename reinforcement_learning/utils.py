import numpy as np
import gym
from matplotlib import pyplot as plt
from matplotlib import colors
import time
import matplotlib
from gym.envs.toy_text.frozen_lake import generate_random_map
import random


# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.title('epsilon')
    plt.ylabel('gamma')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# https://github.com/udacity/deep-reinforcement-learning/blob/b23879aad656b653753c95213ebf1ac111c1d2a6/dynamic-programming/plot_utils.py
def plot_values(V, P, dim, filename):
    # reshape value function
    V_sq = np.reshape(V, (dim, dim))
    P_sq = np.reshape(P, (dim, dim))
    # plot the state-value function
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    if dim < 10:
        fontSize = 20
    else:
        fontSize = 10

    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 2), ha='center', va='top', fontsize=fontSize)
        # LEFT = 0
        # DOWN = 1
        # RIGHT = 2
        # UP = 3
        if np.round(label, 3) > -0.09  and P_sq[j][i] == 0:
            ax.text(i, j, 'LEFT', ha='center', va='bottom', fontsize=fontSize)
        elif np.round(label, 2) > -0.09  and P_sq[j][i] == 1:
            ax.text(i, j, 'DOWN', ha='center', va='bottom', fontsize=fontSize)
        elif np.round(label, 2) > -0.09  and P_sq[j][i] == 2:
            ax.text(i, j, 'RIGHT', ha='center', va='bottom', fontsize=fontSize)
        elif np.round(label, 2) > -0.09  and P_sq[j][i] == 3:
            ax.text(i, j, 'UP', ha='center', va='bottom', fontsize=fontSize)

    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title('State-Value Function')
    fig.tight_layout()
    plt.savefig(filename)


def run_episodes(policy, S, R, p, num_episodes, num_resets):
    rew_arr = []
    for y in range(num_resets):
        forest_state = 0
        tot_rew = 0
        for x in range(num_episodes):
            forest_state = min(forest_state, S - 1)
            if np.random.rand(1) <= p:
                forest_state = -1
            else:
                tot_rew += R[forest_state][policy[forest_state]]
                if policy[forest_state] == 1:
                    forest_state = -1
            forest_state += 1
        rew_arr.append(tot_rew)
    return np.mean(rew_arr)


# Plot out optimal policy
def plot_optimal_policy(best_policy, gammas, filename):
    cmap = colors.ListedColormap(['blue', 'red'])
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.title("Forest Optimal Policy - Red = Cut, Blue = Wait")
    gammas.reverse()
    ax.set_yticklabels(gammas, fontsize=15)
    plt.xticks(fontsize=15)
    ax.tick_params(left=False)  # remove the ticks
    plt.xlabel('State', fontsize=15)
    plt.ylabel('Gamma', fontsize=15)
    plt.pcolor(best_policy[::-1], cmap=cmap, edgecolors='k', linewidths=0)
    plt.savefig(filename)


# Plot Iterations Heatmap
def plot_iteration_heatmap(iters, filename):
    fig, ax1 = plt.subplots()
    ax1.plot(iters, label='iterations')
    ax1.legend()
    plt.xlabel('Gamma', fontsize=15)
    plt.ylabel('Iterations', fontsize=15)
    plt.title("iterations vs gamma")
    fig.tight_layout()
    plt.savefig(filename)


# plot stats vs iteration
def plot_stats(v, error, reward, filename):
    fig, ax1 = plt.subplots()
    ax1.plot(v, label='Mean V')
    ax1.plot(error, label='Error')
    ax1.plot(reward, label='Reward')
    ax1.legend()
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('V/Error/Reward', fontsize=15)
    plt.title("V/Error/ Reward vs iterations")
    fig.tight_layout()
    plt.savefig(filename)


# plot v vs iteration
def plot_v_iteration(v, filename):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mean V', color='r')
    ax1.semilogy(v, color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    plt.title('V vs. Iterations')
    plt.savefig(filename)


# reference: https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/ab06aa68-01f9-481e-94ac-4c6748c3b858.xhtml
def eval_state_action(env, V, s, a, gamma=0.99):
    return np.sum([p * ((rew - 0.01 * _) + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def extract_policy(env, value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)

    return policy


# Utils for policy iterations for frozen lake
def policy_evaluation(env, V, policy, nS, epsilon=0.0001, gamma=0.99):
    while True:
        delta = 0
        for s in range(nS):
            old_v = V[s]
            V[s] = eval_state_action(env, V, s, policy[s], gamma)
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < epsilon:
            break
    return V


def policy_improvement(env, V, policy, nA, nS, gamma=0.99):
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]
        policy[s] = np.argmax([eval_state_action(env, V, s, a, gamma) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False
    return policy, policy_stable


def run_pi_episodes(env, V, policy, num_games=100, num_iter=1000):
    tot_rew = 0
    state = env.reset()
    for _ in range(num_games):
        done = False
        tot_run = 0
        while not done:
            next_state, reward, done, _ = env.step(policy[state])
            state = next_state
            tot_rew += reward
            tot_run += 1
            if done or tot_run > num_iter:
                done = True
                state = env.reset()
    return float(tot_rew / num_games)


def policy_iteration(env, V,  policy, nS, nA, e, g):
    policy_stable = False
    iter = 0
    while not policy_stable:
        V = policy_evaluation(env, V, policy, nS, e, g)
        policy, policy_stable = policy_improvement(env, V, policy, nA, nS, g)
        iter += 1
    return V, policy, iter


# Utils for value iterations for frozen lake
def value_iteration(env, nA, nS, epsilon=0.0001, gamma=0.99):
    V = np.zeros(nS)
    it = 0
    delta_vals = []
    avg_V = []
    while True:
        delta = 0
        # update the value for each state
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(env, V, s, a, gamma) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))
        delta_vals.append(delta)
        avg_V.append(np.mean(V))
        if delta < epsilon:
            break
        it += 1
        if it > 10000:
            break
    return V, delta_vals, it, avg_V


def run_vi_episodes(env, V, nA, num_games=100, gamma=0.99):
    tot_rew = 0
    state = env.reset()
    max_run = 0

    for _ in range(num_games):
        done = False

        while not done:
            # choose the best action using the value function
            action = np.argmax([eval_state_action(env, V, state, a, gamma) for a in range(nA)])  # (11)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            tot_rew += reward
            max_run += 1
            if done:
                state = env.reset()
            if max_run > 10000:
                state = env.reset()
                done = True
                tot_rew += 0
                max_run = 0

    # print('Won %i of %i games!' % (tot_rew, num_games))
    return float(tot_rew / num_games)


# Utils for Q-Learning for frozen lake
# reference: https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python/blob/master/Chapter04/SARSA%20Q_learning%20Taxi-v2.py
def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q[s])


def run_q_episodes(env, Q, num_episodes=100):
    '''
    Run some episodes to test the policy in q learning
    '''
    tot_rew = []
    state = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        while not done:
            # select a greedy action
            next_state, rew, done, _ = env.step(greedy(Q, state))

            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    return np.mean(tot_rew)


def Q_learning(env, lr=0.01, lr_min=0.0001, lr_decay=0.99, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005,
               eps_min=0.0001):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    # test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > eps_min:
            eps *= eps_decay
            eps = max(eps, eps_min)

        # decay the learning rate until it reaches the threshold of lr_min
        if lr > lr_min:
            lr *= lr_decay
            lr = max(lr, lr_min)

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)

            next_state, rew, done, _ = env.step(action)  # Take one step in the environment
            rew -= (0.01 * done)

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

    return Q
