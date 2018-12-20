import numpy as np

# Game with pure optimal action for each player
reward_array = np.array([[3,2],[2,1]])

# Rock Paper Scissors
reward_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])

# Did not compute the value and optimal strategies for this one yet
reward_array = np.array([[1,2],[0,3]])

# Game with value -1/2, x=[1/2, 1/2], y=[1/4, 3/4]
reward_array = np.array([[1,-1],[-2,0]])

# Did not compute the value and optimal strategies for this one yet
reward_array = np.array([[3,1],[0,0],[-2,4],[7,-2]])