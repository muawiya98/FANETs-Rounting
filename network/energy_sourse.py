import logging
import numpy as np

import config
import config as cf


class EnergySource(object):
    def __init__(self):
        self.energy = np.random.RandomState().randint(2000, config.INITIAL_ENERGY)
        self.max_energy = cf.INITIAL_ENERGY

    def recharge(self):
        self.energy = self.max_energy


class Battery(EnergySource):
    def consume(self, energy):
        if self.energy >= energy:
            self.energy -= energy
        else:
            self.energy = 0


class PluggedIn(EnergySource):
    def consume(self, energy):
        pass
