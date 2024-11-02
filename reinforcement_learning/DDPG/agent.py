import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from . import buffer, networks
from reinforcement_learning.enviroment import Environment
from reinforcement_learning.reward import Reward
import config as cf
from .utils import normalized_actions

tf.random.set_seed(
    cf.seed_weights_number
)


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=4, max_size=20000, tau=0.01,
                 fc1=64, fc2=32, batch_size=32, noise=0.05):
        self.gamma = gamma
        self.tau = tau
        self.memory = buffer.ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = 0
        self.min_action = 1

        self.actor = networks.ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, n_actions=n_actions,
                                           name='actor')
        self.critic = networks.CriticNetwork(fc1_dims=fc1, fc2_dims=fc2,  name='critic')
        self.target_actor = networks.ActorNetwork(fc1_dims=fc1, fc2_dims=fc2,
                                                  n_actions=n_actions, name='target_actor')
        self.target_critic = networks.CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_target_network_parameters(tau=1)
        self.environment = Environment(ddpg=True)
        self.reward = Reward()
        self.rl_measures = {'reward': [], 'role_change': [], 'energy_change': []}
        self.number_of_target_network_train = 0
        self.number_of_evaluation_network_train = 0
        self.update_target_network_counter = 0

    def update_target_network_parameters(self, tau=None):
        """
        Input:
             1- tau: parameter determine copy ratio to target network.
             if tau wasn't none --> hard copy (only at step one).
        return:
        soft copy, update the target networks, in respect to tau.
        """
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state):
        """
        save transition:
        """
        self.memory.store_transition(state, action, reward, new_state)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def make_action(self, observation, evaluate=False):
        """

        """
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            noise = tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
            actions += noise

        actions = normalized_actions(actions)
        return actions

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.number_of_evaluation_network_train += 1
        state, action, reward, new_state = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_target_network_parameters()
        self.number_of_target_network_train += 1
        # self.update_target_network_counter = 0

    def update_q_table(self, network, time):

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
        self.remember(current_state, action, rd, next_state)

        return rd, rcc, ec

    def run(self, network, time):

        # get delayed reward.
        if time in self.environment.transition_table_updates.keys():
            reward, role_count, energy_change = self.update_q_table(network, time)
            self.rl_measures['reward'].append(reward)
            self.rl_measures['role_change'].append(role_count)
            self.rl_measures['energy_change'].append(energy_change)

        _, current_state = self.environment.get_state(network)

        action = self.make_action(current_state)
        self.environment.transition_table_updates[time + cf.RE_CLUSTERING_THRESHOLD] = (current_state, action)

        self.learn()
        return action.numpy().tolist()
