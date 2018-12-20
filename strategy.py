import numpy as np

class strategy:
    def __init__(self, reward_array):
        self.reward_array = reward_array
        self.action_nb = self.reward_array.shape[0]
        self.past_actions = np.zeros((self.action_nb))
        self.draws_nb = 0

    def draw_action(self):
        raise NotImplementedError()

class fictitious_play(strategy):
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