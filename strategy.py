import numpy as np
import math

class strategy:
    """
    Class presenting the general structure for a strategy
    """
    def __init__(self, reward_array):
        self.reward_array = reward_array
        self.action_nb = self.reward_array.shape[0]
        self.past_actions = np.zeros((self.action_nb))
        self.draws_nb = 0

    def draw_action(self):
        raise NotImplementedError()

    def take_reward(self, drawn_action, reward):
        raise NotImplementedError()

class oblivious_play(strategy):
    """
    Strategy playing according to a given probability distribution
    """
    def __init__(self, reward_array, proba_distrib):
        strategy.__init__(self, reward_array)
        self.proba_distrib = proba_distrib

    def draw_action(self, opponent_past_actions):
        drawn_action = np.random.choice(range(self.action_nb), p=self.proba_distrib)
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        return

class fictitious_play(strategy):
    """
    Strategy using fictitious play
    """
    def __init__(self, reward_array):
        strategy.__init__(self, reward_array)
        self.first_turn = True

    def draw_action(self, opponent_past_actions):
        if self.first_turn:
            drawn_action = np.random.randint(0, self.action_nb)
            self.first_turn = False
        else:
            empirical_adversary_distrib = opponent_past_actions/(float(self.draws_nb))
            drawn_action = np.argmax(np.dot(self.reward_array, empirical_adversary_distrib))
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        return

class bandit_UCB(strategy):
    """
    Strategy using the upper confidence bound method developed in
    multi armed bandit theory
    """
    def __init__(self, reward_array, confidence_coef):
        strategy.__init__(self, reward_array)
        self.confidence_coef = confidence_coef
        self.rewards = np.zeros((self.action_nb))

    def draw_action(self, opponent_past_actions):
        if (self.draws_nb < self.action_nb) :
            # Sampling each arm/action first
            drawn_action = self.draws_nb
            #print(self.rewards, self.past_actions, self.rewards/self.past_actions)
        else:
            bound_term = np.sqrt(math.log(self.draws_nb+1)/(2*self.past_actions))
            upper_bound_values = self.rewards/self.past_actions + self.confidence_coef * bound_term
            #print(self.rewards, self.past_actions, self.rewards/self.past_actions, upper_bound_values)
            drawn_action = np.argmax(upper_bound_values)
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        #print(reward, drawn_action)
        self.rewards[drawn_action] += reward