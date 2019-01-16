import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from strategy import *
from lh_main import *

# Defining the reward array
loss_array = (1000*np.random.rand(20,20)).astype(int)

# Number of steps for the algorithm
step_number = 80000
frequency_evolutions = [np.zeros((loss_array.shape[0], step_number)), 
                        np.zeros((loss_array.shape[1], step_number))]

#Solving
opt_valeur, opt_strategies = solve(loss_array)
# print(opt_strategies)
print("Value : " + str(opt_valeur))
print("")

# Strategies we test
strategies = [fictitious_play, perturbed_fictitious_play, exp_weighted_average, regret_matching, deterministic_explor_exploit]

def simulate_ob_game(loss_array, player_learning_strategy, ob_player_strategy_distrib):
    players = [player_learning_strategy(loss_array, step_number), 
               oblivious_play(-loss_array.T, ob_player_strategy_distrib, step_number)]
    joint_past_actions = np.zeros(loss_array.shape)
    # Regret evolution for the first player
    regret_evolution = np.zeros(step_number)
    rewards_array = np.zeros(step_number)
    print("Running the algorithm")
    for step in tqdm(range(step_number)):
        drawn_actions = []
        for player_idx, player in enumerate(players):
            # each player plays his turn
            opponent_past_actions = players[(player_idx+1)%2].past_actions
            player_past_actions = players[(player_idx)%2].past_actions
            opponent_past_actions_index = players[(player_idx+1)%2].past_actions_index
            player_past_actions_index = players[(player_idx)%2].past_actions_index
            drawn_actions.append(player.draw_action(player_past_actions, opponent_past_actions, player_past_actions_index, opponent_past_actions_index))
            frequency_evolutions[player_idx][:, step] = player_past_actions / (step+1)
        current_reward = loss_array[drawn_actions[0], drawn_actions[1]]
        rewards_array[step] = current_reward
        joint_past_actions[drawn_actions[0], drawn_actions[1]] += 1
        regret_evolution[step] = (np.sum(joint_past_actions*loss_array) - np.min(np.dot(loss_array, players[1].past_actions)))
        for player_idx, player in enumerate(players):
            player.take_reward(drawn_actions[player_idx], ((-1)**player_idx) * current_reward)
    return rewards_array

erased_values = 200
sns.set(style="darkgrid")
plt.plot(range(erased_values, step_number), [-opt_valeur]*(step_number-erased_values))

mean_reward_evo = np.zeros((len(strategies), step_number))
for idx, sim_strategy in enumerate(strategies):
    rewards = simulate_ob_game(loss_array, sim_strategy, np.reshape(opt_strategies[1], loss_array.shape[1]))
    mean_reward_evo[idx, :] = np.cumsum(rewards) / np.arange(1, step_number+1)
    plt.plot(range(erased_values, step_number), mean_reward_evo[idx, erased_values:])

plt.legend(["Optimal value"] + [sim_strategy.__name__ for sim_strategy in strategies])
plt.xlabel("Rounds")
plt.ylabel("Mean cumulated loss")
plt.show()

# print("")
# print("Loss array : ")
# print(loss_array)
# print("")
# print("Frequency of play for each action")
# for player_idx, player in enumerate(players):
#     print("For player " + str(player_idx) + " : ")
#     print(player.past_actions/float(step_number))
# print("Joint past actions : ")
# print(joint_past_actions/float(step_number))


# Showing convergence to the optimal strategy
# The first steps are deleted so we can see the real convergence
# plt.plot(np.linalg.norm(frequency_evolutions[0] - opt_strategies[0], ord=2, axis=0)[20:])
# plt.show()

print(-opt_valeur)
#print(np.dot(players[0].past_actions/float(step_number), loss_array).dot(players[1].past_actions/float(step_number)))
#print((joint_past_actions*loss_array).sum()/step_number)
print(rewards.mean())