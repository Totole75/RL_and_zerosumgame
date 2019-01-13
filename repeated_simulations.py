import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from strategy import *
from lh_main import *

# Defining the reward array
#loss_array = np.array([[3,2],[2,1]])
loss_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
#loss_array = np.array([[1,0],[2,3]])
#loss_array = np.array([[1,-2],[-1,0]])
#loss_array = np.array([[3,0,-2,7],[1,0,4,-2]])
#loss_array = np.array([[1,2, 3],[4,5, 6]])
#loss_array = np.random.rand(20,20)

# Number of steps for the algorithm
step_number = 1000
frequency_evolutions = [np.zeros((loss_array.shape[0], step_number)), 
                        np.zeros((loss_array.shape[1], step_number))]

sim_nb = 10000
emp_distrib_1 = np.zeros((loss_array.shape[0], sim_nb))
emp_distrib_2 = np.zeros((loss_array.shape[1], sim_nb))

final_regrets = np.zeros(sim_nb)

for sim_step in tqdm(range(sim_nb)):
    # Defining the players
    #players = [fictitious_play(loss_array, step_number), fictitious_play(-loss_array.T, step_number)]
    players = [perturbed_fictitious_play(loss_array, step_number), perturbed_fictitious_play(-loss_array.T, step_number)]
    #players = [bandit_UCB(loss_array, 0.2, step_number), oblivious_play(-loss_array.T, np.array([0.5, 0.5]), step_number)]
    #players = [bandit_UCB(loss_array, 0.2, step_number), bandit_UCB(-loss_array.T, 0.2, step_number)]
    #players = [exp_weighted_average(loss_array, step_number), exp_weighted_average(-loss_array.T, step_number)]
    #players = [deterministic_explor_exploit(loss_array, step_number), deterministic_explor_exploit(-loss_array.T, step_number)]
    #players = [deterministic_explor_exploit(loss_array, step_number), exp_weighted_average(-loss_array.T, step_number)]
    #players = [regret_matching(loss_array, step_number),  exp_weighted_average(-loss_array.T, step_number)]

    joint_past_actions = np.zeros(loss_array.shape)
    # Regret evolution for the first player
    regret_evolution = np.zeros(step_number)
    for step in range(step_number):
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
        if ((sim_step == sim_nb-1) and step == step_number-1):
            print(joint_past_actions / float(step_number))
    emp_distrib_1[:, sim_step] = players[0].past_actions / step_number
    emp_distrib_2[:, sim_step] = players[1].past_actions / step_number
    final_regrets[sim_step] = regret_evolution[-1]

#################################################
# Convergence to a Nash Equilibrium for shifumi #
#################################################

# def build_df(emp_distrib, player_tag):
#     result_data_rock = {"Mixed strategy for" : [player_tag]*emp_distrib.shape[1], 
#                "value" : ["Rock"]*emp_distrib.shape[1],
#                "action" : emp_distrib[0,:]}
#     result_data_paper = {"Mixed strategy for" : [player_tag]*emp_distrib.shape[1], 
#                "value" : ["Paper"]*emp_distrib.shape[1],
#                "action" : emp_distrib[1,:]}
#     result_data_scissors = {"Mixed strategy for" : [player_tag]*emp_distrib.shape[1], 
#                "value" : ["Scissors"]*emp_distrib.shape[1],
#                "action" : emp_distrib[2,:]}
#     df1 = pd.DataFrame(data=result_data_rock, index=np.arange(emp_distrib.shape[1]))
#     df2 = pd.DataFrame(data=result_data_scissors, index=emp_distrib.shape[1]+np.arange(emp_distrib.shape[1]))
#     df3 = pd.DataFrame(data=result_data_paper, index=2*emp_distrib.shape[1]+np.arange(emp_distrib.shape[1]))
#     return pd.concat([df1, df2, df3])

# optimal_df = build_df(np.resize(np.array([1.0/3, 1.0/3, 1.0/3]), (3, 1)), "Nash Equilibrium")
# frames = [build_df(emp_distrib_1, "Player 1"), build_df(emp_distrib_2, "Player 2"), optimal_df]
# df = pd.concat(frames)

# sns.set(style="darkgrid")
# sns.catplot(x='value', y='action', hue='Mixed strategy for', 
#                data=df, 
#                kind='bar', 
#                errwidth=1.5 ,capsize=.15)
# plt.ylim(0.32, 0.34)
# plt.show()

#####################################
# Regret going over bound frequency #
#####################################

print("max", np.max(final_regrets))

delta_values = np.arange(5, 55, 5)/100.0
print(delta_values)
df = pd.DataFrame()
for idx, delta in enumerate(delta_values):
    bound_value = 2*np.sqrt(step_number*loss_array.shape[0]) + np.sqrt(0.5*step_number*np.log(1/(delta)))
    overbound_bool_values = np.where(final_regrets>=bound_value, 1, 0)
    result_dict = {"Delta" : [delta]*sim_nb, "Empirical rate of regret going over bound" : overbound_bool_values}
    current_df = pd.DataFrame(data=result_dict, index=idx*sim_nb+np.arange(sim_nb))
    df = pd.concat([df, current_df])
    print(bound_value)
    #print("max", np.max(final_regrets), "bound", bound_value)

# print(df)
sns.set(style="darkgrid")
sns.catplot(x='Delta', y='Empirical rate of regret going over bound', hue='Delta',
               data=df, 
               markers="o", linestyles=" ",
               kind="point", palette=sns.cubehelix_palette(10, rot=-.25, light=.7),
               errwidth=1.5 ,capsize=.15)
# plt.ylim(0.32, 0.34)
plt.show()