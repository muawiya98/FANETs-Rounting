from . import RL
import config as cf
import numpy as np
from .enviroment import Environment
from .reward import Reward


class QLearning:
    def __init__(self, quantization_actions=10, epsilon=0.9, decay=0.992, gamma=0.9, lr=0.3):
        self.epsilon = epsilon
        self.decay = decay
        self.gamma = gamma
        self.lr = lr
        self.environment = Environment(quantization_actions, q_learning=True)
        self.min_epsilon = 0.1
        self.reward = Reward()
        self.rl_measures = {'reward': [], 'role_change': [], 'energy_change': []}

    def run(self, network, time):

        # get delayed reward.
        if time in self.environment.transition_table_updates.keys():
            reward, role_count, energy_change = self.update_q_table(network, time)
            self.rl_measures['reward'].append(reward)
            self.rl_measures['role_change'].append(role_count)
            self.rl_measures['energy_change'].append(energy_change)

        current_state_idx, current_state = self.environment.get_state(network)

        # action --> is index ( to get the actual action: self.environment.actions[action])
        if cf.seed_RL.rand() < self.epsilon:
            action = self.environment.random_action()
        else:
            action = np.argmax(self.environment.Q_table[current_state_idx, :])

        # perform action...
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        # save action to get the reward later.
        self.environment.transition_table_updates[time + cf.RE_CLUSTERING_THRESHOLD] = (current_state_idx, action)
        return self.environment.actions[action]

    def update_q_table(self, network, time):

        next_state_idx, next_state = self.environment.get_state(network)

        rd, rcc, ec = 0, 0, 0
        for node in network:
            rd += self.reward(node.count_role, node.time_series_energy[-cf.RE_CLUSTERING_THRESHOLD], node.energy_source. \
                              energy, node.count_affirmations)
            rcc += node.count_role
            ec += node.time_series_energy[-cf.RE_CLUSTERING_THRESHOLD] - node.energy_source. \
                              energy

            node.count_role = 0
            node.count_affirmations = 0

        rd /= len(network)
        rcc /= len(network)
        ec /= len(network)

        current_state_idx, action = self.environment.transition_table_updates[time]

        # self.q_table[tuple(state)][action] = ((1 - self.lr) * self.q_table[tuple(state)][action]) + \
        #                                     (self.lr * (reward + self.gamma * max(self.q_table[tuple(next_state)])))
    
        self.environment.Q_table[current_state_idx, action] = ((1-rd) * self.environment.Q_table[current_state_idx, action]) + (rd + self.gamma * np.max(self.environment.Q_table[next_state_idx, :]))

        self.environment.transition_table_updates.pop(time)
        return rd, rcc, ec