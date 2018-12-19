import numpy as np

# Fictitious play in a two player game
action_nb = 2
player_nb = 2
reward_array = np.array([[3,2],[2,1]])

past_actions = np.zeros((player_nb, action_nb))

step_nb = 100
for step in range(step_nb):
    played_action = np.zeros(player_nb)
    if (step==0):
        for player_id in range(player_nb):
            played_action[player_id] = np.random.randint(0, action_nb)
    else:
        for player_id in range(player_nb):
            #actions = numpy.identity(action_nb)
            empirical_adversary = past_actions[player_id, :]/(float(step))
            if (player_id == 0):
                # player 0 wants to maximize
                chosen_action_id = np.argmax(np.dot(reward_array, empirical_adversary))
            else:
                # player 1 wants to minimize
                chosen_action_id = np.argmin(np.dot(reward_array, empirical_adversary))
            past_actions[player_id, chosen_action_id] += 1

print("Reward array for this game : ")
print(reward_array)
print("Frequency of play for each action")
for player_id in range(player_nb):
    print("For player " + str(player_id) + " : ")
    print(past_actions[player_id, :]/float(step_nb))