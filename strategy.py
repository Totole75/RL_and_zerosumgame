import numpy as np
import math
import tools
class strategy:
    """
    Class presenting the general structure for a strategy
    """
    def __init__(self, loss_array):
        self.loss_array = loss_array
        self.action_nb = self.loss_array.shape[0]
        self.draws_nb = 0
        

    def draw_action(self):
        raise NotImplementedError()

    def take_reward(self, drawn_action, reward):
        raise NotImplementedError()

class oblivious_play(strategy):
    """
    Strategy playing according to a given probability distribution
    """
    def __init__(self, loss_array, proba_distrib):
        strategy.__init__(self, loss_array)
        self.proba_distrib = proba_distrib
        self.past_actions = np.zeros(self.action_nb)
        
    def draw_action(self, player_past_actions, opponent_past_actions):
        drawn_action = np.random.choice(range(self.action_nb), p=self.proba_distrib)
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        return

class fictitious_play(strategy):
    """
    Strategy using fictitious play (not Hannan consistent)
    """
    def __init__(self, loss_array):
        strategy.__init__(self, loss_array)
        self.first_turn = True
        self.past_actions = np.zeros(self.action_nb)

    def draw_action(self, player_past_actions, opponent_past_actions):
        if self.first_turn:
            drawn_action = np.random.randint(0, self.action_nb)
            self.first_turn = False
        else:
            empirical_adversary_distrib = opponent_past_actions/(float(self.draws_nb))
            drawn_action = np.argmin(np.dot(self.reward_array, empirical_adversary_distrib))
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        return

class perturbed_fictitious_play(strategy):
    """
    Strategy using fictitious play with perturbed loss to make it
    Hannan consistent
    """
    def __init__(self, reward_array):
        strategy.__init__(self, reward_array)
        self.first_turn = True

    def draw_action(self, opponent_past_actions):
        if self.first_turn:
            drawn_action = np.random.randint(0, self.action_nb)
            self.first_turn = False
        else:
            perturbations = np.random.uniform(low=0, high=np.sqrt(self.action_nb*(1+self.draws_nb)), size=self.action_nb)
            drawn_action = np.argmin(np.dot(self.reward_array, opponent_past_actions) + perturbations)
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
    def __init__(self, loss_array, confidence_coef):
        strategy.__init__(self, loss_array)
        self.confidence_coef = confidence_coef
        self.rewards = np.zeros((self.action_nb))
        self.past_actions = np.zeros(self.action_nb)
        self.rewards = np.zeros((self.action_nb))
        
    def draw_action(self, player_past_actions, opponent_past_actions):
        if (self.draws_nb < self.action_nb) :
            # Sampling each arm/action first
            drawn_action = self.draws_nb
        else:
            bound_term = np.sqrt(math.log(self.draws_nb+1)/(2*self.past_actions))
            upper_bound_values = self.rewards/self.past_actions + self.confidence_coef * bound_term
            drawn_action = np.argmin(upper_bound_values)
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        self.rewards[drawn_action] += reward

class exp_weighted_average(strategy):
    """
    Strategy using the exponentially weighted average strategy (Hannan consistent)
    """
    def __init__(self, loss_array):
        strategy.__init__(self, loss_array)
        self.exp_coef = np.sqrt(8*np.log(self.action_nb))
        self.past_actions = np.zeros(self.action_nb)
        self.rewards = np.zeros((self.action_nb))

    def draw_action(self, player_past_actions, opponent_past_actions):
        exp_values = np.exp(np.dot(self.loss_array, opponent_past_actions) * (-self.exp_coef/np.sqrt(1+self.draws_nb)))
        action_probas = exp_values / np.linalg.norm(exp_values, ord=1)
        drawn_action = np.random.choice(range(self.action_nb), p=action_probas)
        self.past_actions[drawn_action] += 1
        self.draws_nb += 1
        return drawn_action

    def take_reward(self, drawn_action, reward):
        #print(reward, drawn_action)
        self.rewards[drawn_action] += reward
        
class deterministic_explor_exploit(strategy):
    """
    Strategy using the deterministic exploration_exploitation theory (cf p222)
    """
    def __init__(self, loss_array, nb_step):
        strategy.__init__(self, loss_array, nb_step)
        self.mu = np.zeros((self.action_nb, nb_step))
        self.rewards = np.zeros((self.action_nb))
        self.past_actions = np.zeros(nb_step).astype(int)
        
    def draw_action(self, player_past_actions, opponent_past_actions):
         # On peut probablement le mettre en version calcul vectoriel. Utile ?
        for action in range(self.action_nb):
            indexes_action = np.where(player_past_actions[:self.draws_nb] == action)[0]
            if len(indexes_action) ==0:
                self.mu[action, self.draws_nb] = 1
            else:
                Jt = opponent_past_actions[indexes_action]
                #print('a', type(action), action)
                #print('b', type(self.draws_nb), self.draws_nb)
                #print('c', type(Jt), Jt)
                self.mu[action, self.draws_nb] = np.mean(self.loss_array[action, Jt])
        
        # Exploration
        if (tools.is_square(self.draws_nb)):
            drawn_action = 0
        elif (tools.is_square(self.draws_nb - 1)):
            drawn_action = 1
        
        # Exploitation
        else:
            drawn_action = np.argmin(self.mu[:, self.draws_nb])
        
        self.past_actions[self.draws_nb] = drawn_action
        self.draws_nb += 1

        return drawn_action

    def take_reward(self, drawn_action, reward):
        #print(reward, drawn_action)
        self.rewards[drawn_action] += reward
        
    
    ## Remarquer que limsup mu_chap <= limsup min(mu)
