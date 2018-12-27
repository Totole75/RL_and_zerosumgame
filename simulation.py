import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from solveur import *
from strategy import *

# Defining the reward array
#loss_array = np.array([[3,2],[2,1]])
loss_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
#loss_array = np.array([[1,0],[2,3]])
#loss_array = np.array([[1,-2],[-1,0]])
#loss_array = np.array([[3,0,-2,7],[1,0,4,-2]])

step_number = 1000
frequency_evolutions = [np.zeros((loss_array.shape[0], step_number)), 
                        np.zeros((loss_array.shape[1], step_number))]

# Defining the players
#players = [fictitious_play(loss_array), fictitious_play(-loss_array.T)]
#players = [perturbed_fictitious_play(loss_array), perturbed_fictitious_play(-loss_array.T)]
#players = [bandit_UCB(loss_array, confidence_coef=0.2), oblivious_play(-loss_array.T, proba_distrib=np.array([0.5, 0.5]))]
#players = [bandit_UCB(loss_array, confidence_coef=0.2), bandit_UCB(-loss_array.T, confidence_coef=0.2)]
players = [exp_weighted_average(loss_array), exp_weighted_average(-loss_array.T)]
#players = [deterministic_explor_exploit(loss_array, step_number), deterministic_explor_exploit(-loss_array.T, step_number)]

opt_valeur, opt_strategies = solve_2players(loss_array, verbose=True)
#opt_valeur, opt_strategy = ce(-loss_array.transpose())
print("")

# Number of steps for the algorithm
joint_past_actions = np.zeros(loss_array.shape)
print("Running the algorithm")
for step in tqdm(range(step_number)):
    drawn_actions = []
    for player_idx, player in enumerate(players):
        # each player plays his turn
        opponent_past_actions = players[(player_idx+1)%2].past_actions
        player_past_actions = players[(player_idx)%2].past_actions
        drawn_actions.append(player.draw_action(player_past_actions, opponent_past_actions))
        frequency_evolutions[player_idx][:, step] = player_past_actions / (step+1)
    current_reward = loss_array[drawn_actions[0], drawn_actions[1]]
    joint_past_actions[drawn_actions[0], drawn_actions[1]] += 1
    for player_idx, player in enumerate(players):
        player.take_reward(drawn_actions[player_idx], ((-1)**player_idx) * current_reward)

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