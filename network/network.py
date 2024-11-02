from config import episodes_routing, number_of_episodes, number_of_steps
from reinforcement_learning import DQN, Q_learning
from matplotlib.animation import FuncAnimation
from reinforcement_learning.DDPG import agent
from communication import packet_messages
import EventHandler, performance_logger
from utils.graphVisualization import *
from tqdm import tqdm
import visualization
import config as cf
from .node import *
import itertools
import logging
import math


class Network:
    def __init__(self):

        self.nodes = [Node(i) for i in range(0, cf.NB_NODES)]
        self.gcd_pos_x = cf.seed_simulator.uniform(0, cf.AREA_WIDTH)
        self.gcd_pos_y = cf.seed_simulator.uniform(0, cf.AREA_LENGTH)

        self.round = 0

        self.event_handler = EventHandler.EventHandler()
        self.packets_event_handler = EventHandler.EventHandler()
        self.packet_Messages = packet_messages.PacketMessages()
        self.measures_logger = performance_logger.PerformanceLogger()
        self.initial_energy = self.get_remaining_energy()
        self.energy_spent = []

    def __call__(self, type_visualization, communication_instance, rl_type):
        """
        :param type_visualization: controls what we want to see, the clustering
        or the packets exchange
        :param communication_instance: how the nodes communicate with each other, in this project
        the only considered way is control messages.
        :param rl_type: type of RL agent.
        """
        self.type_visualization = type_visualization
        self.communication_instance = communication_instance

        if rl_type == 'Q-learning':
            self.rl_agent = Q_learning.QLearning(quantization_actions=cf.QUANTIZATION_ACTION_SPACE)
        elif rl_type == 'DQN':
            self.rl_agent = DQN.DQN(input_dim=cf.STATE_LEN, quantization_actions=6)
        elif rl_type == 'DDPG':
            self.rl_agent = agent.Agent(input_dims=cf.STATE_LEN, n_actions=cf.ACTION_LEN)

    def restart(self):
        for node in self.nodes:
            node.buffer.clear()
    
    def calculate_distance(self, node1, node2):
        return math.sqrt((node1.pos_x - node2.pos_x) ** 2 + (node1.pos_y - node2.pos_y) ** 2)

    def selecte_nodes_for_trian(self, nodes):
        max_distance = 0
        best_combination = None
        for combination in itertools.combinations(nodes, 4):
            total_distance = sum(
                self.calculate_distance(node1, node2)
                for node1, node2 in itertools.combinations(combination, 2))
            if total_distance > max_distance:
                max_distance = total_distance
                best_combination = combination
        return [n.id for n in best_combination], best_combination # type: ignore

    def group_nodes_by_nearest(self, nodes, nodes_for_trian):
        groups = {node.id: [] for node in nodes_for_trian}
        for node in nodes:
            if node in nodes_for_trian:continue
            closest_node = min(nodes_for_trian, key=lambda fn: self.calculate_distance(node, fn))
            groups[closest_node.id].append(node.id)        
        return groups
    
    def simulate(self, random_weights=None):
        nodes_ids_for_trian, nodes_for_trian = self.selecte_nodes_for_trian(self.nodes)
        groups = self.group_nodes_by_nearest(self.nodes, nodes_for_trian)
        time, step, episod = 0, 0, 98
        phase_type = 'Train'
        while(True):
            self.round = time
            if (time % cf.episodes_routing)==0:
                if (time//cf.episodes_routing)-1>0:
                    self.nodes[(time // cf.episodes_routing)-1].trainable_node = False
                self.nodes[time//cf.episodes_routing].trainable_node=True
            nb_alive_nodes = self.count_alive_nodes()
            if nb_alive_nodes == 0:continue
            if not (time % cf.RE_CLUSTERING_THRESHOLD) and time != 0:
                if random_weights is not None:
                    weights = random_weights
                else:
                    weights = self.rl_agent.run(self.nodes, time)
                if not (time % cf.ADJUST_WEIGHTS):
                    for node in self.nodes:
                        node.weights = weights
            self.run_round(time, time % cf.SEND_CH_DECLARATION_MESSAGES_RATE == 1)
            self.communication_instance.exchange_messages(self.nodes,time,self.event_handler,
                                                          time % cf.MOVEMENT_RATE <= 1,time % cf.SEND_CH_DECLARATION_MESSAGES_RATE,)
            self.communication_instance.send_join_request_messages(self.nodes, time, self.event_handler)
            step = self.communication_phase(time, step, nodes_ids_for_trian, groups, phase_type)
            self.measures_logger.calculate_measures()
            self.calculate_network_delta_energy(time)
            if step>=number_of_steps:
                print(f"Episod Number {episod} Dnoe")
                step=0
                episod+=1
                self.restart()
                if episod>=100:phase_type='Test'
            if episod>=number_of_episodes:break
            time+=1
        return weights

    def nodes_routine(self, round_nb, ok_move):
        for node in self.nodes:
            node.do_routine(
                round_nb,
                ids=len(self.nodes),
                ok_move=ok_move,
                logger=self.measures_logger,
            )

        self.update_neighbors()

    def run_round(self, round_nb, re_cluster):
        """
        Run one round. Every node captures using its sensor. Then this
        information is forwarded through the intermediary nodes to the base
        station.
        """
        self.nodes_routine(
            round_nb, round_nb % cf.MOVEMENT_RATE == 0
        )
        # if you want the visualization, uncomment this part.
        if cf.visualize and  round_nb % 10 == 0 :
            build_adjacency_clusters(
                self.nodes,
                round_nb,
                (round_nb + 1) == cf.MAX_ROUNDS,
                re_cluster,
                self.type_visualization,
            )

    def communication_phase(self, time, step, nodes_for_trian, groups, phase_type='Train'):
        before_energy = self.get_remaining_energy()
        self.packet_Messages.listen(self.nodes, time, self.packets_event_handler, self.measures_logger)        
        step = self.send_packets_q_learning(time, step, nodes_for_trian, groups, phase_type)
        after_energy = self.get_remaining_energy()
        self.measures_logger.energy_consumption.append(before_energy - after_energy)
        return step
    
    def calculate_reward(self, node, neighbors, action, destination):
        if action == destination:return 100
        # elif action == node.id and action != destination and destination in neighbors:return -50
        elif action != destination and destination in neighbors:return -50
        return -10
    
    def send_packets_q_learning(self, time, step, nodes_for_trian, groups, phase_type='Train'):
        id_for_load = None
        steps = []
        for node in self.nodes:
            if node.id in groups.keys():
                id_for_load = node.id
            else:
                for key in groups.keys():
                    if node.id in groups[key]:
                        id_for_load = key
            if not node.alive:continue
            for _ in range(cf.NUMBER_OF_SEND_PACKETS):
                packet = node.buffer.get_top_packet()
                if packet is None:continue
                node.buffer -= 0
                packet_sent = node.packet_id_sended.get(packet.id, False)
                if packet_sent:
                    self.measures_logger.packets_logger[packet.id][0] = True
                    continue
                previous_neighbors = set(node.neighbors)
                if node.id in nodes_for_trian or phase_type=='Test':
                    destination, step = node.find_next_hob_q_learning(packet, self.nodes, step, id_for_load, phase_type)  
                else:
                    dst_node = get_destination_node(self.nodes, packet.destination)
                    destination = node.lar_route_discovery(dst_node, self.nodes, 2)
                    destination = destination[-1] if len(destination)==1 else np.random.choice(destination)
                if destination == node.id:
                    self.measures_logger.packets_logger[packet.id][0] = True
                else:
                    node.packet_id_sended[packet.id] = True
                    packet.path.append(node.id)
                    if self.nodes[destination].role == -1 or self.nodes[destination].forwarding_node: # type: ignore
                        node.messaging_obj("packet",node.id,
                                            destination, packet.id, packet.path,
                                            packet.destination, packet.length,
                                            packet.generation_moment, packet.expiration_date, time, self.packets_event_handler,)
                        node.consume_energy(calculate_distance(node, self.nodes[destination]), packet.length) # type: ignore
                if node.id==1:node.reward = self.calculate_reward(node, previous_neighbors, destination, packet.destination) # type: ignore
            steps.append(step)
        return min(steps)
    
    
    def get_alive_nodes(self):
        """Return nodes that have positive remaining energy."""
        return [node for node in self.nodes if node.alive]

    def get_remaining_energy(self):
        """Returns the sum of the remaining energies at all nodes."""
        set_alive = self.get_alive_nodes()
        if len(set_alive) == 0:
            return 0

        energies = [x.energy_source.energy for x in set_alive]
        return sum(energies)

    def count_alive_nodes(self):
        return sum(x.alive for x in self.nodes)

    def update_node_neighbors(self, target_node):
        """
        get the nodes who fell in the range of coverage of the target node.
        """
        target_node.forwarding_node = False
        target_node.neighbors.clear()

        for node in self.get_alive_nodes():
            if node == target_node:
                continue
            distance = calculate_distance(target_node, node)
            if distance <= cf.COVERAGE_RADIUS:
                if node.alive:
                    target_node.neighbors.append(node.id)
                    if target_node.role != -1 and target_node.role != node.role:
                        target_node.forwarding_node = True
        target_node.nb_neighbors = len(target_node.neighbors)

    def update_neighbors(self):
        for node in self.get_alive_nodes():
            self.update_node_neighbors(node)

    def calculate_network_delta_energy(self, round_np):
        for node in self.nodes:
            node.calculate_delta_energy(round_np)

    def find_nearest_cluster_for_node(self, target_node_id):
        """
        Find the nearest cluster head for the node with the given target_node_id.
        """
        target_node = next((node for node in self.nodes if node.id == target_node_id), None)
        if not target_node:
            # print(f"Node with ID {target_node_id} not found.")
            return None

        cluster_heads = self.get_cluster_heads() # type: ignore
        if not cluster_heads:
            print("No cluster heads available in the network.")
            return None

        nearest_ch = None
        min_distance = float('inf')

        for ch in cluster_heads:
            distance = calculate_distance(target_node,ch)
            if distance < min_distance:
                min_distance = distance
                nearest_ch = ch

        return nearest_ch