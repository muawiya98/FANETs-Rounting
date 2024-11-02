from matplotlib.colors import hsv_to_rgb, rgb2hex
from random import choices, seed, random
from pickle import LIST
import numpy as np
import pickle
import math
import os

root_path = os.getcwd()

# Describe the scenarios that will be simulated
# scenarios should be described in the following format:
# scenario_name = (routing_topology, sleep_scheduling, aggregation_model)

# the 4th argument is the nickname of that plot, if not specified (None),
# then the name is: routing_topology + sleep_scheduling

# for convenience, the scenarios list also accepts commands that are
# executed in run.py

mode="routing_rl"
visualize =False#True

LIST_INFO_PLANES = [
    {"name": "Macro-phase", "transmission_range": [10, 500], "mobility": [66.7, 100]},
    {"name": "Pico-phase", "transmission_range": [10, 300], "mobility": [33.4, 66.6]},
    {"name": "Femto-phase", "transmission_range": [10, 100], "mobility": [0, 33.3]},
]

seed_simulator = np.random.RandomState(1)
seed_RL_number = 60
seed_RL = np.random.RandomState(seed_RL_number)
seed_weights_number = 50
seed_weights = np.random.RandomState(seed_weights_number)

episodes_routing=1000


seed_WR_number = 10  # weight random
seed_WR = np.random.RandomState(seed_WR_number)

seed_NB_number = 5  # number of node seed
seed_WS_number = 30  # walking speed seed
seed_CR_number = 40  # coverage radius seed

# scenario : (clustering method, nickname, type network, type visualization, NB_NODES, WALKING_SPEED, COVERAGE_RADIUS)
# type visualization : "cluster" OR "heat map" OR "vertical"

# scenario0 = ("ControlMessages", None, "Network()", "heat map")
# scenario1 = ("ControlMessages", None, "VerticalNetwork()", "cluster")
scenarios = []  # , scenario0]
numbper_of_nodes = 100
for i in range(1):
    seed_NB = np.random.RandomState(seed_NB_number)  # number of node seed
    seed_WS = np.random.RandomState(seed_WS_number)  # walking speed seed
    seed_CR = np.random.RandomState(seed_CR_number)  # coverage radius seed

    row = ["ControlMessages",
            f"scenario_{i + 1}",
            "VerticalNetwork()",
            "cluster-radius",
            "random",  # type of cluster random or RL
            '',]

    for plane in LIST_INFO_PLANES:
        WS = seed_WS.randint(plane['mobility'][0]/10, plane['mobility'][1]/10)
        row.append(WS)
    row.append(seed_CR.randint(100, 150))
    row.append(numbper_of_nodes)
    # row.append(seed_NB.randint(40, 50))

    # print(row)

    scenarios.append(row)
    seed_NB_number += 1  # number of node seed
    seed_WS_number += 1  # walking speed seed
    seed_CR_number += 1  # coverage radius seed

TRACE_ENERGY = 0
TRACE_ALIVE_NODES = 1
TRACE_COVERAGE = 1
TRACE_LEARNING_CURVE = 0
TIME_UNIT = 0.01  # 10 ms --> 0.01 second
CHANNEL_CAPACITY = 20 * 8  # mega bit per second.
# Runtime configuration
MAX_ROUNDS = 10000# 30 sec
# number of transmissions of sensed information to cluster heads or to
# base station (per round)


# Network configurations:
# number of nodes
NB_NODES = 5
# node sensor range
COVERAGE_RADIUS = 150  # meters
SUB_GRID_SIZE=10
# node transmission range
TX_RANGE = 1000  # meters
# area definition
AREA_WIDTH = 1200.0
AREA_LENGTH = 800.0
NUMBER_OF_SEND_PACKETS = 150
STATE_LEN = 5
ACTION_LEN = 4
LOWER_BOUND_PACKET_GENERATION = 1
UPPER_BOUND_PACKET_GENERATION = 10
# packet configs
MSG_LENGTH = 4000  # bits
CONTROL_MSG_LENGTH = 192  # bits
HEADER_LENGTH = 150  # bits
EXPIRATION_LIMIT = 1000  # 1 sec. (100 time-step * 10 ms per time step)
BUFFER_SIZE = 200  # 200 packets.
# initial energy at every node's battery
INITIAL_ENERGY = 200000000  # Joules

R2D = 180 / np.pi
# Energy Configurations
# energy dissipated at the transceiver electronic (/bit)
E_ELEC = 50e-9  # Joules
# energy dissipated at the data aggregation (/bit)
E_DA = 5e-9  # Joules
# energy dissipated at the power amplifier (supposing a multi-path
# fading channel) (/bin/m^4)
E_MP = 0.0013e-12  # Joules
WALKING_SPEED = {}  #5  # 50 m per sec --> 5 m per 100 ms (every 10 time steps).
FORBIDDEN_EDGES = 20  # m for each edge.
# energy dissipated at the power amplifier (supposing a line-of-sight
# free-space channel (/bin/m^2)+

E_FS = 10e-12  # Joules
THRESHOLD_DIST = math.sqrt(E_FS / E_MP)  # meters
LINK_THRESHOLD = 2
INITIAL_RE_CLUSTERING = 4
MAINTAIN_CH_VALIDITY = 8
MOVEMENT_RATE = 10  # move every 100 ms [1 m every 100ms --> 10m per sec]
SEND_HELLO_MESSAGES_RATE = MOVEMENT_RATE
SEND_CH_DECLARATION_MESSAGES_RATE = 70  # change clusters every 70 time steps.
# Other configurations:
# grid precision (the bigger the faster the simulation)
GRID_PRECISION = 1  # in meters
# useful constants (for readability)
INFINITY = float("inf")
MINUS_INFINITY = float("-inf")
QUANTIZATION_ACTION_SPACE = 10
RESULTS_PATH = "./results/"
WEIGHT_OF_STRUCTURE_STABILITY = 0.4
COUNT_ROLE_THRESHOLD = 2
COUNT_AFFIRMATION_THRESHOLD = 1
RE_CLUSTERING_THRESHOLD = 10
CLIPPING_THRESHOLD_S4 = 10
ADJUST_WEIGHTS = 50
UTILITY_FACTOR_QUANTIZATION_MIN = 0
UTILITY_FACTOR_QUANTIZATION_MAX = 10
NODES_HANGING_PERIOD = 30
WINDOW_DELTA_ENERGY = 10
NUMBER_OF_ALLOWED_ACTIONS = 10

LIST_NB_NODES_IN_PLANES = [0] * len(LIST_INFO_PLANES)

######################################################################
PLANES_COLORS = [
    rgb2hex(hsv_to_rgb((i * 1.0 / len(LIST_INFO_PLANES), 0.4, 0.9)))
    for i in range(len(LIST_INFO_PLANES))
]

# ****************Q-Learning Routing****************
number_of_steps=300
number_of_episodes=120

def save_object(obj, filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()
def load_object(filename, path):
    filename = os.path.join(path, filename)
    with open(filename + ".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object