import numpy as np
import config as cf


class Reward:
    def __init__(self):
        self.r = 0

    def __call__(self, *args, **kwargs):
        return self.calculate_reward(*args)

    def calculate_reward(self, *args):
        """

        :param args: role changing, energies (before, after).
        :return:
        Equations 23 --> 26 of the article.
        """
        self.calculate_r(*args)
        return self.r if (self.r == 1 or self.r == -1) else (1-np.exp(-3*self.r)/(1+np.exp(3*self.r)))

    def calculate_r(self, count_role, energy_before, energy_after, count_affirmations):
        rc = 1 if count_role < cf.COUNT_ROLE_THRESHOLD else -1
        af = 1 if count_affirmations >= cf.COUNT_AFFIRMATION_THRESHOLD else 0

        delta_energy = (energy_before - energy_after) / cf.INITIAL_ENERGY
        ec = 1 - delta_energy
        self.r = cf.WEIGHT_OF_STRUCTURE_STABILITY * rc + \
                 cf.WEIGHT_OF_STRUCTURE_STABILITY * ec + 0.2 * af
