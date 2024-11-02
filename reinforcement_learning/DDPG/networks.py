import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import config as cf


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                             seed=cf.seed_weights_number))
        self.fc2 = Dense(self.fc2_dims, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                             seed=cf.seed_weights_number))
        self.q = Dense(1, activation=None)

    def __call__(self, state, action):
        """
        the first layer input is the [state, action] vector.
        """
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                             seed=cf.seed_weights_number))
        self.fc2 = Dense(self.fc2_dims, activation='relu',
                         kernel_initializer=tf.keras.initializers.glorot_uniform(
                             seed=cf.seed_weights_number))

        self.mu = Dense(self.n_actions, activation='softmax')

    def __call__(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu

