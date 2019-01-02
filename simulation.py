import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from solveur import *
from strategy import *

# Defining the reward array
#loss_array = np.array([[3,2],[2,1]])
#loss_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
#loss_array = np.array([[1,0],[2,3]])
loss_array = np.array([[1,-2],[-1,0]])
#loss_array = np.array([[3,0,-2,7],[1,0,4,-2]])
#loss_array = np.random.rand(20,20)

# Number of steps for the algorithm
step_number = 20000
frequency_evolutions = [np.zeros((loss_array.shape[0], step_number)), 
                        np.zeros((loss_array.shape[1], step_number))]

# Defining the players
#players = [fictitious_play(loss_array, step_number), fictitious_play(-loss_array.T, step_number)]
players = [perturbed_fictitious_play(loss_array, step_number), perturbed_fictitious_play(-loss_array.T, step_number)]
#players = [bandit_UCB(loss_array, 0.2, step_number), oblivious_play(-loss_array.T, np.array([0.5, 0.5]), step_number)]
#players = [bandit_UCB(loss_array, 0.2, step_number), bandit_UCB(-loss_array.T, 0.2, step_number)]
#players = [exp_weighted_average(loss_array, step_number), exp_weighted_average(-loss_array.T, step_number)]
#players = [deterministic_explor_exploit(loss_array, step_number), deterministic_explor_exploit(-loss_array.T, step_number)]
#players = [deterministic_explor_exploit(loss_array, step_number), exp_weighted_average(-loss_array.T, step_number)]
#players = [regret_matching(loss_array, step_number),  exp_weighted_average(-loss_array.T, step_number)]

opt_valeur, opt_strategies = solve_2players(loss_array, verbose=True)
#opt_valeur, opt_strategy = ce(-loss_array.transpose())
print("")

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

print("")
print("Loss array : ")
print(loss_array)
print("")
print("Frequency of play for each action")
for player_idx, player in enumerate(players):
    print("For player " + str(player_idx) + " : ")
    print(player.past_actions/float(step_number))
print("Joint past actions : ")
print(joint_past_actions/float(step_number))

# Showing convergence to the optimal strategy
# The first steps are deleted so we can see the real convergence
for player_idx, player in enumerate(players):
    plt.plot(np.linalg.norm(frequency_evolutions[player_idx] - opt_strategies[player_idx], ord=2, axis=0)[20:])
plt.show()

# Displaying bounds related to the regret
plt.plot(range(step_number), regret_evolution)

delta = 0.1
# Upper bound for the perturbed fictitious play regret (external) (corollary 4.4)
bound_values = [(2*np.sqrt(step*loss_array.shape[0]) + np.sqrt(-0.5*step*np.log(delta))) for step in range(1, step_number+1)]
plt.plot(range(step_number), bound_values)

plt.show()