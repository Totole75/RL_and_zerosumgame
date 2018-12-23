import numpy as np

# In this context, the first player is minimizing
# and the second player is maximizing (minimizng the
# transposed opposite of the reward array)

# Game with pure optimal action for each player
reward_array = np.array([[3,2],[2,1]])

# Rock Paper Scissors
reward_array = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])

# Game with value 1, x=[1, 0], y=[1, 0]
reward_array = np.array([[1,0],[2,3]])

# Game with value -1/2, x=[1/4, 3/4], y=[1/2, 1/2]
reward_array = np.array([[1,-2],[-1,0]])

# Game with value 7/4, x=[3/8, 5/8], y=[3/4, 0, 1/4, 0]
reward_array = np.array([[3,0,-2,7],[1,0,4,-2]])
