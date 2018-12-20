import numpy as np
from tqdm import tqdm

from strategy import fictitious_play

# Fictitious play in a two player game
#reward_array = np.array([[3,2],[2,1]])
reward_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
players = [fictitious_play(reward_array), fictitious_play(-reward_array.T)]

joint_past_actions = np.zeros(reward_array.shape)
step_number = 1000000
for step in tqdm(range(step_number)):
    drawn_actions = []
    for player_idx, player in enumerate(players):
        # each player plays his turn
        opponent_past_actions = players[(player_idx+1)%2].past_actions
        drawn_actions.append(player.draw_action(opponent_past_actions))
    joint_past_actions[drawn_actions[0], drawn_actions[1]] += 1

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