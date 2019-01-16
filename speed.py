import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from strategy import *
from lh_main import *

# Strategies we test
strategies = [fictitious_play]#, perturbed_fictitious_play, exp_weighted_average, regret_matching]

def simulate_ob_game(player_learning_strategy, 
                     step_number, sim_nb):
    reward_evo = np.zeros((sim_nb, step_number))
    for sim_step in tqdm(range(sim_nb)):
        # Defining the reward array
        loss_array = (1000*np.random.rand(20,20)).astype(int)
        #Solving
        opt_valeur, opt_strategies = solve(loss_array)
        # Optimal value is negative
        opt_valeur = -opt_valeur

        players = [player_learning_strategy(loss_array, step_number), 
                oblivious_play(-loss_array.T, np.reshape(opt_strategies[1], loss_array.shape[1]), step_number)]
        joint_past_actions = np.zeros(loss_array.shape)
        # Regret evolution for the first player
        regret_evolution = np.zeros(step_number)
        rewards_array = np.zeros(step_number)
        for step in range(step_number):
            drawn_actions = []
            for player_idx, player in enumerate(players):
                # each player plays his turn
                opponent_past_actions = players[(player_idx+1)%2].past_actions
                player_past_actions = players[(player_idx)%2].past_actions
                opponent_past_actions_index = players[(player_idx+1)%2].past_actions_index
                player_past_actions_index = players[(player_idx)%2].past_actions_index
                drawn_actions.append(player.draw_action(player_past_actions, opponent_past_actions, player_past_actions_index, opponent_past_actions_index))
            current_reward = loss_array[drawn_actions[0], drawn_actions[1]]
            rewards_array[step] = current_reward
            joint_past_actions[drawn_actions[0], drawn_actions[1]] += 1
            regret_evolution[step] = (np.sum(joint_past_actions*loss_array) - np.min(np.dot(loss_array, players[1].past_actions)))
            for player_idx, player in enumerate(players):
                player.take_reward(drawn_actions[player_idx], ((-1)**player_idx) * current_reward)
        rewards_array = np.cumsum(rewards_array) / np.arange(1, step_number+1)
        reward_evo[sim_step, :] = np.abs(rewards_array-opt_valeur)/opt_valeur
    return reward_evo

erased_values = 0
sns.set(style="darkgrid")
#plt.plot(range(erased_values, step_number), [-opt_valeur]*(step_number-erased_values))
# Number of steps for the algorithm
step_number = 10000
sim_nb = 10

for idx, sim_strategy in enumerate(strategies):
    reward_gaps = simulate_ob_game(sim_strategy, step_number, sim_nb)
    reward_means = np.mean(reward_gaps, axis=0)
    reward_std = np.std(reward_gaps, axis=0)
    plt.plot(range(erased_values, step_number), reward_means[erased_values:], "r-")
    plt.plot(range(erased_values, step_number), (reward_means-reward_std)[erased_values:], "g-")
    plt.plot(range(erased_values, step_number), (reward_means+reward_std)[erased_values:], "g-")

#plt.legend([sim_strategy.__name__ for sim_strategy in strategies])
plt.xlabel("Rounds")
plt.ylabel("Distance")
plt.show()

# Showing convergence to the optimal strategy
# The first steps are deleted so we can see the real convergence
# plt.plot(np.linalg.norm(frequency_evolutions[0] - opt_strategies[0], ord=2, axis=0)[20:])
# plt.show()