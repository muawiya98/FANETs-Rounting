from .enviroment import Environment
from .reward import Reward
import numpy as np
import config as cf
from abc import ABC, abstractmethod


class RL(ABC):
    def __init__(self,  quantization_actions=10, epsilon=0.9, decay=0.992, gamma=0.9, lr=0.3):

        self.epsilon = epsilon
        self.decay = decay
        self.gamma = gamma
        self.lr = lr
        self.environment = Environment(quantization_actions)
        self.min_epsilon = 0.1
        self.reward = Reward()
        self.over_all_rewards = []

    @abstractmethod
    def run(self, *args):
        pass

    @abstractmethod
    def update_q_table(self, *args):
        pass
