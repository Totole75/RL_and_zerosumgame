import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from strategy import *
from lh_main import *

# Defining the reward array
loss_array = (1000*np.random.rand(3,3)).astype(int)

# Number of steps for the algorithm
step_number = 5000
frequency_evolutions = [np.zeros((loss_array.shape[0], step_number)), 
                        np.zeros((loss_array.shape[1], step_number))]
#Solving
opt_valeur, opt_strategies = solve(loss_array)
print(opt_strategies)
print("Value : " + str(opt_valeur))
print("")

player_1_strat = opt_strategies[1].reshape(-1)

# Defining the players
players_list = []
players_list.append([fictitious_play(loss_array, step_number), oblivious_play(-loss_array.T, player_1_strat, step_number)])
players_list.append([perturbed_fictitious_play(loss_array, step_number), oblivious_play(-loss_array.T, player_1_strat, step_number)])
players_list.append([bandit_UCB(loss_array, 0.2, step_number), oblivious_play(-loss_array.T, player_1_strat, step_number)])
players_list.append([exp_weighted_average(loss_array, step_number), oblivious_play(-loss_array.T, player_1_strat, step_number)])
players_list.append([deterministic_explor_exploit(loss_array, step_number), oblivious_play(-loss_array.T, player_1_strat, step_number)])
players_list.append([regret_matching(loss_array, step_number),  oblivious_play(-loss_array.T, player_1_strat, step_number)])

sns.set(style="darkgrid")

for players in players_list:
    #print(players)
    joint_past_actions = np.zeros(loss_array.shape)
    # Regret evolution for the first player
    regret_evolution = np.zeros(step_number)
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
        joint_past_actions[drawn_actions[0], drawn_actions[1]] += 1
        regret_evolution[step] = (np.sum(joint_past_actions*loss_array) - np.min(np.dot(loss_array, players[1].past_actions)))
        for player_idx, player in enumerate(players):
            player.take_reward(drawn_actions[player_idx], ((-1)**player_idx) * current_reward)
            

    # Showing convergence to the optimal strategy
    # The first steps are deleted so we can see the real convergence
    plt.plot(np.linalg.norm(frequency_evolutions[0] - opt_strategies[0], ord=2, axis=0)[20:])
plt.legend([str(type(players[0]))[17:-2] for players in players_list])
plt.show()