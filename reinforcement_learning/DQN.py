from . import RL
import config as cf
import numpy as np
import keras
from collections import deque
import copy
import tensorflow as tf
import random
from .enviroment import Environment
from .reward import Reward
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(
    cf.seed_weights_number
)


class DQN:
    def __init__(self, buffer_size=2000, input_dim=5, quantization_actions=10,
                 epsilon=0.9, decay=0.992, gamma=0.9, lr=0.1):
        # super(DQN, self).__init__(quantization_actions)
        self.epsilon = epsilon
        self.decay = decay
        self.gamma = gamma
        self.lr = lr
        self.environment = Environment(quantization_actions)
        self.min_epsilon = 0.1
        self.reward = Reward()
        self.rl_measures = {'reward': [], 'role_change': [], 'energy_change': []}
        self.number_of_target_network_train = 0
        self.number_of_evaluation_network_train = 0
        self.input_dim = input_dim

        self.evaluation_network = self.create_model()
        self.target_network = self.create_model()
        self.update_target_network()

        self.buffer_size = buffer_size
        self.fitted = False

        self.reply_buffer = deque(maxlen=self.buffer_size)
        self.update_target_network_counter = 0

    def create_model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(self.input_dim,)))
        model.add(keras.layers.Dense(64, activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=cf.
                                                                                              seed_weights_number)))
        model.add(keras.layers.Dense(32, activation='relu',
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(seed=cf.
                                                                                             seed_weights_number)))
        model.add(keras.layers.Dense(10))
        model.compile(loss='mse', optimizer='adam')
        return model

    def policy_function(self, expected_reward, not_fitted=False) -> int:
        """
        expected_reward: list of expected rewards for each possible action
        epsilon: .
        """
        if cf.seed_RL.rand() <= self.epsilon or not_fitted:
            return cf.seed_RL.choice(np.arange(len(expected_reward)))
        else:
            return np.argmax(expected_reward)

    def make_action(self, curr_state) -> int:
        # q_values represents the expected rewards for each possible action
        if self.fitted:
            q_values = self.evaluation_network.predict(np.array(curr_state).reshape(-1, self.input_dim))
            action = self.policy_function(q_values)
        else:
            action = self.policy_function(np.arange(0, len(self.environment.actions)), True)  # to be fixed.
        return action

    def update_evaluation_network(self, batch_size=32, epochs=5):
        # select random batch from the reply buffer
        batch = random.sample(self.reply_buffer, batch_size)

        # initialize some lists to store transition information
        states, actions, next_states, rewards = [], [], [], []

        # from each transition extract its values
        for transition in batch:
            states.append(transition[0])
            actions.append(transition[1])
            next_states.append(transition[2])
            rewards.append(transition[3])

        x_train = np.array(states).reshape(-1, self.input_dim)

        current_expected_reward = self.evaluation_network.predict(np.array(states).
                                                                  reshape(-1, self.input_dim))
        future_expected_reward = self.target_network.predict(np.array(next_states)
                                                             .reshape(-1, self.input_dim))

        # update expected rewards using Billiman equation
        # temp_diff = rd + self.gamma * np.max(self.environment.Q_table[next_state_idx, :]) - self.environment.Q_table[
        #     current_state_idx, action]
        # self.environment.Q_table[current_state_idx, action] += (self.lr * temp_diff)

        for i, act in enumerate(actions[:-1]):
            max_future_q = np.max(future_expected_reward[i])

            current_expected_reward[i, act] = rewards[i] + (self.gamma * max_future_q)

        y_train = current_expected_reward.copy()

        # train the DQN evaluation network.
        self.evaluation_network.fit(x_train, y_train, epochs=epochs, verbose=0)
        self.fitted = True
        return

    def update_target_network(self):
        self.target_network.set_weights(self.evaluation_network.get_weights())

    def update_transition_table(self, network, time):
        """
        get the next state and reward to update
        """
        _, next_state = self.environment.get_state(network)

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

        current_state, action = self.environment.transition_table_updates[time]
        self.environment.transition_table_updates.pop(time)
        self.reply_buffer.append([current_state, action, next_state, rd])

        return rd, rcc, ec

    def run(self, network, time):

        # get delayed reward.
        if time in self.environment.transition_table_updates.keys():
            reward, role_count, energy_change = self.update_transition_table(network, time)
            self.rl_measures['reward'].append(reward)
            self.rl_measures['role_change'].append(role_count)
            self.rl_measures['energy_change'].append(energy_change)

        _, current_state = self.environment.get_state(network)

        action = self.make_action(current_state)
        self.environment.transition_table_updates[time + cf.RE_CLUSTERING_THRESHOLD] = (current_state, action)

        if len(self.reply_buffer) > 32:
            self.update_evaluation_network()
            self.number_of_evaluation_network_train += 1

        if self.update_target_network_counter % 30 == 0 and self.fitted:  # to be fixed
            self.update_target_network()
            self.number_of_target_network_train += 1
            self.update_target_network_counter = 0

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        return self.environment.actions[action]
