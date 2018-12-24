import numpy as np
from tqdm import tqdm

from solveur import *
from strategy import *

# Defining the reward array
#reward_array = np.array([[3,2],[2,1]])
#reward_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
#reward_array = np.array([[1,0],[2,3]])
#reward_array = np.array([[1,-2],[-1,0]])
reward_array = np.array([[3,0,-2,7],[1,0,4,-2]])

# Defining the players
#players = [fictitious_play(reward_array), fictitious_play(-reward_array.T)]
#players = [perturbed_fictitious_play(reward_array), perturbed_fictitious_play(-reward_array.T)]
#players = [bandit_UCB(reward_array, confidence_coef=0.2), oblivious_play(-reward_array.T, proba_distrib=np.array([0.5, 0.5]))]
#players = [bandit_UCB(reward_array, confidence_coef=0.2), bandit_UCB(-reward_array.T, confidence_coef=0.2)]
players = [exp_weighted_average(reward_array), exp_weighted_average(-reward_array.T)]
#players = [deterministic_explor_exploit(reward_array, step_number), deterministic_explor_exploit(-reward_array.T, step_number)]

# Number of steps for the algorithm
step_number = 10000
joint_past_actions = np.zeros(reward_array.shape)
for step in tqdm(range(step_number)):
    drawn_actions = []
    for player_idx, player in enumerate(players):
        # each player plays his turn
        opponent_past_actions = players[(player_idx+1)%2].past_actions
        player_past_actions = players[(player_idx)%2].past_actions
        drawn_actions.append(player.draw_action(player_past_actions, opponent_past_actions))
    current_reward = reward_array[drawn_actions[0], drawn_actions[1]]
    joint_past_actions[drawn_actions[0], drawn_actions[1]] += 1
    for player_idx, player in enumerate(players):
        player.take_reward(drawn_actions[player_idx], ((-1)**player_idx) * current_reward)

print("Reward array for this game : ")
print(reward_array)
print("")
print("Frequency of play for each action")
for player_idx, player in enumerate(players):
    print("For player " + str(player_idx) + " : ")
    print(player.past_actions/float(step_number))
print("")
print("Joint past actions : ")
print(joint_past_actions/float(step_number))