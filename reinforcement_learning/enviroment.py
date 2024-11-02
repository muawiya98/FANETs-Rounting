import numpy as np
from utils.utils import entropy, prop, quantization
import config as cf


class Environment:
    def __init__(self, quantization_actions=0, ddpg=False, q_learning=False):
        self.transition_table_updates = {}
        self.ddpg = ddpg
        self.q_learning = q_learning

        if not ddpg:
            self.actions = {}
            self.initial_state_key = 0
            self.initial_state = [0.3, 0.3, 0.2, 0.2]
            self.get_actions(quantization_actions)
            self.action_space_size = len(self.actions)
            self.state_dict = {}
            self.state_counter = 1
            self.observation_space_size_allocation = 100000
            self.Q_table = np.zeros((self.observation_space_size_allocation, self.action_space_size))

            self.Q_table[:, self.initial_state_key] = 1

    def get_actions(self, quantization_actions):
        idx = 0
        for i in np.arange(1, quantization_actions + 1):
            for j in np.arange(1, quantization_actions + 1):
                for k in np.arange(1, quantization_actions + 1):
                    if (i + j + k + 1) <= quantization_actions:
                        x = quantization_actions - (i + j + k)
                        self.actions[idx] = [i / quantization_actions, j / quantization_actions, k /
                                             quantization_actions, x / quantization_actions]
                        if [i / quantization_actions, j / quantization_actions, k / quantization_actions,
                                                                            x / quantization_actions] == self.\
                                initial_state:
                            self.initial_state_key = idx
                        idx += 1

    def random_action(self):
        return cf.seed_RL.choice(self.action_space_size)

    def get_state(self, network):
        state = []
        utilities = ['s1', 's2', 's3', 's4']
        for attr_name in utilities:
            utility_i = quantization(list(map(lambda x: getattr(x, attr_name), network)),
                                     cf.UTILITY_FACTOR_QUANTIZATION_MAX,
                                     cf.UTILITY_FACTOR_QUANTIZATION_MIN)
            state.append(entropy(prop(utility_i)))
        if self.q_learning:
            state /= sum(state)
            state = list(np.round(np.array(state), decimals=1))
        # print(state)
        # print(np.round(np.array(state) % 0.1, decimals=4))
        # print(np.array(state) % 0.1)
        success_paths = round(sum(list(map(lambda x: np.mean(getattr(x, 'successfully_received_messages')[-10:])
                     , network))))
        state.append(success_paths)
        if not self.ddpg:
            return self.encode_state(state), state
        else:
            return [], state

    def encode_state(self, state):
        state = tuple(state)
        if state in self.state_dict.keys():
            return self.state_dict[state]

        self.state_dict[state] = self.state_counter
        self.state_counter += 1
        return self.state_dict[state]
