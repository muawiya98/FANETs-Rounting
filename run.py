import config as cf
from network.vertical_network import VerticalNetwork
from network.network import *
import sys
from communication.control_messages import *
import pickle
from numpy import zeros
from tqdm import tqdm
from visualization import visualize_measures

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def run_scenarios():
    d = {}

    for i, scenario in enumerate(cf.scenarios):
        temp_d = {}

        row = scenario
        clustering_method, nickname, network, type_visual, type_cluster, rl_type = row[:6]
        cf.COVERAGE_RADIUS, cf.NB_NODES = row[-2:]
        speeds = row[6:-2]
        for idx, speed in enumerate(speeds):
            cf.WALKING_SPEED[cf.LIST_INFO_PLANES[idx]['name']] = speed
        # print(cf.WALKING_SPEED)

        if nickname:
            scenario_name = nickname
        else:
            scenario_name = clustering_method

        # print(scenario)
        # print(scenario_name + ": running scenario...")

        cf.LIST_NB_NODES_IN_PLANES = [0] * len(cf.LIST_INFO_PLANES)
        for node in range(cf.NB_NODES):
            plane = cf.seed_simulator.choice(len(cf.LIST_INFO_PLANES))
            cf.LIST_NB_NODES_IN_PLANES[plane] += 1
        network = eval(network)
        communication_class = eval(clustering_method)
        network(type_visual, communication_class(), rl_type)
        init_weights = random_weights(1)

        weights = network.simulate(random_weights=init_weights if type_cluster == "random" else None)
        if type_cluster != 'random':  # change seed.
            cf.seed_weights_number += 1
            cf.seed_weights = np.random.RandomState(cf.seed_weights_number)

            cf.seed_RL_number += 1
            cf.seed_RL = np.random.RandomState(cf.seed_RL_number)

        print("DONE")
        arrived, all_packet = 0, 0
        """
        since there is a lot of versions for the same message, packets_logger keeps two values
        for the packet arrived or dropped
        """
        for key, value in network.measures_logger.packets_logger.items():
            if value[1]:
                arrived += 1
            if value[0] or value[1]:
                all_packet += 1

        PDR = arrived / all_packet
        avg_e2e = np.mean(network.measures_logger.e2e_delay_list)
        avg_egc = np.mean(network.measures_logger.energy_consumption)

        # d is dict: name scenario : (scenario info, weights info, result info)
        #     scenario info: [num of nodes, walking speed, coverage raduis]
        d[scenario_name] = (
            [scenario[5], scenario[6], scenario[7]],
            weights,
            [PDR, avg_e2e, avg_egc],
        )
        temp_d[scenario_name] = (
            [scenario[5], scenario[6], scenario[7]],
            weights,
            [PDR, avg_e2e, avg_egc],
        )
        # print('count_alive_nodes', network.count_alive_nodes())


        if type_cluster != 'random':
            if rl_type != 'Q-learning':
                print('number_of_target_network_train:', network.rl_agent.number_of_target_network_train)
                print('number_of_evaluation_network_train:', network.rl_agent.number_of_evaluation_network_train)
        network_type = 'VN'

        with open(f'results_{type_cluster}_{network_type}{i}_rew.pkl', 'wb') as file:
            pickle.dump(temp_d, file)
        visualize_measures(network.measures_logger, False,i)

    network_type = 'VN'
    if len(cf.LIST_INFO_PLANES) == 1:
        network_type = 'HN'

    if type_cluster != 'random':
        with open('results_' + rl_type + '_' + network_type + '_rew.pkl', 'wb') as file:
            pickle.dump(d, file)
        with open('results_' + rl_type + '_' + network_type + '_rl_measures_rew.pkl', 'wb') as file:
            pickle.dump(network.rl_agent.rl_measures, file)

    else:
        with open('results_' + type_cluster + '_' + network_type + '_rew.pkl', 'wb') as file:
            pickle.dump(d, file)
            # print(d)

def random_weights(n):
    w = zeros(shape=(n, 4))
    for item in w:
        item[0] = cf.seed_WR.uniform(0, 1)
        item[1] = cf.seed_WR.uniform(0, 1 - (item[0]))
        item[2] = cf.seed_WR.uniform(0, 1 - (item[0] + item[1]))
        item[3] = 1 - (item[0] + item[1] + item[2])
        cf.seed_WR.shuffle(item)

    cf.seed_WR_number += 1
    cf.seed_WR = np.random.RandomState(cf.seed_WR_number)  # number of node seed
    return w.flatten()


def run_scenarios_with_q_learning_routing():
    d = {}

    for i, scenario in enumerate(cf.scenarios):
        temp_d = {}
        row = scenario
        clustering_method, nickname, network, type_visual, type_cluster, rl_type = row[:6]
        cf.COVERAGE_RADIUS, cf.NB_NODES = row[-2:]
        if cf.mode=="routing_rl":
            cf.MAX_ROUNDS=cf.NB_NODES*cf.episodes_routing
        cf.SUB_GRID_SIZE=round(cf.COVERAGE_RADIUS / math.sqrt(5))
        speeds = row[6:-2]
        for idx, speed in enumerate(speeds):
            cf.WALKING_SPEED[cf.LIST_INFO_PLANES[idx]['name']] = speed
        if nickname:
            scenario_name = nickname
        else:
            scenario_name = clustering_method
        cf.LIST_NB_NODES_IN_PLANES = [0] * len(cf.LIST_INFO_PLANES)
        for node in range(cf.NB_NODES):
            plane = cf.seed_simulator.choice(len(cf.LIST_INFO_PLANES))
            cf.LIST_NB_NODES_IN_PLANES[plane] += 1

        network = eval(network)
        communication_class = eval(clustering_method)
        network(type_visual, communication_class(), rl_type)
        init_weights = random_weights(1)
        weights = network.simulate(random_weights=init_weights if type_cluster == "random" else None)
        if type_cluster != 'random':  # change seed.
            cf.seed_weights_number += 1
            cf.seed_weights = np.random.RandomState(cf.seed_weights_number)

            cf.seed_RL_number += 1
            cf.seed_RL = np.random.RandomState(cf.seed_RL_number)

        # print("DONE")
        arrived, all_packet = 0, 0
        """
        since there is a lot of versions for the same message, packets_logger keeps two values
        for the packet arrived or dropped
        """
        for key, value in network.measures_logger.packets_logger.items():
            if value[1]:
                arrived += 1
            if value[0] or value[1]:
                all_packet += 1

        PDR = arrived / all_packet
        avg_e2e = np.mean(network.measures_logger.e2e_delay_list)
        avg_egc = np.mean(network.measures_logger.energy_consumption)

        # d is dict: name scenario : (scenario info, weights info, result info)
        #     scenario info: [num of nodes, walking speed, coverage raduis]
        d[scenario_name] = (
            [scenario[5], scenario[6], scenario[7]],
            weights,
            [PDR, avg_e2e, avg_egc],
        )
        temp_d[scenario_name] = (
            [scenario[5], scenario[6], scenario[7]],
            weights,
            [PDR, avg_e2e, avg_egc],
        )
        # print('count_alive_nodes', network.count_alive_nodes())


        if type_cluster != 'random':
            if rl_type != 'Q-learning':
                print('number_of_target_network_train:', network.rl_agent.number_of_target_network_train)
                print('number_of_evaluation_network_train:', network.rl_agent.number_of_evaluation_network_train)
        network_type = 'VN'

        with open(f'results_{type_cluster}_{network_type}{i}_rew.pkl', 'wb') as file:
            pickle.dump(temp_d, file)
        visualize_measures(network.measures_logger, False,i)

    network_type = 'VN'
    if len(cf.LIST_INFO_PLANES) == 1:
        network_type = 'HN'

    if type_cluster != 'random':
        with open('results_' + rl_type + '_' + network_type + '_rew.pkl', 'wb') as file:
            pickle.dump(d, file)
        with open('results_' + rl_type + '_' + network_type + '_rl_measures_rew.pkl', 'wb') as file:
            pickle.dump(network.rl_agent.rl_measures, file)

    else:
        with open('results_' + type_cluster + '_' + network_type + '_rew.pkl', 'wb') as file:
            pickle.dump(d, file)
            # print(d)


if __name__ == "__main__":
    run_scenarios_with_q_learning_routing()
