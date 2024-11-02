from matplotlib.style import available
from networkx.classes import neighbors, nodes

import config as cf
import numpy as np

from reinforcement_learning.q_learning_routing import QLearningRouting
from .energy_sourse import *
from .buffer import *
from utils import Mobility
from utils.utils import *
from Messages.Message import Message
from .packet import *
import copy


class Node:
    def __init__(self, id_, movement_type="random_sub_grid"):
        """
        :param id_: id for node --> to give it the right energy source.
        """
        self.pos_x = cf.seed_simulator.uniform(0, cf.AREA_WIDTH)
        self.pos_y = cf.seed_simulator.uniform(0, cf.AREA_LENGTH)

        self.org_pos_x = self.pos_x
        self.org_pos_y = self.pos_y

        self.id = id_
        self.role = id_  # -1 means cluster head.

        self.alive = 1
        self.tx_queue_size = 0
        self.amount_sensed = 0
        self.amount_transmitted = 0
        self.amount_received = 0

        # objects:
        self.buffer = Buffer(cf.BUFFER_SIZE)
        self.packets_counter = 0
        self.messaging_obj = Message()
        self.movement_obj = Mobility.Mobility(movement_type)
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.utility = 0
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        self.s4 = 0
        self.energy_source = Battery()
        self.last_energy = [self.energy_source.energy] * cf.WINDOW_DELTA_ENERGY
        self.delta_energy_in_timestep = 0
        self.successfully_received_messages = []
        self.func = None
        self.packet_generation_moment = 5 + cf.seed_simulator.randint(10)

        # to store the neighbors utilities and other information.
        self.w_table = {}
        self.p_table = {}
        self.s_table = {}
        self.role_table = {}
        self.ch_table = {}

        # for coverage purposes
        self.neighbors = []
        self.nb_neighbors = 0
        self.exclusive_radius = 0
        self.degree = 0
        self.velocity = 0
        self.angle_degree = 0
        self.count_role = 0
        self.count_affirmations = 0
        self.time_out_of_cluster = -1
        self.ch_time = -1  # the moment that this node became a CH.
        self.ray = [self.velocity, self.angle_degree]
        self.new_members = []
        self.forwarding_members = []
        self.forwarding_node = False
        self.trainable_node = False
        self.time_series_energy = []
        self.members = (
            []
        )  # empty if it was an ordinary node, or include members if it was a cluster head.
        self.packet_id_sended = dict()
        self.is_gateway = False  # Flag to identify if this node is a gateway
        self.gateway_table={}
        self.q_learning = QLearningRouting(num_actions=4, id=id_)  # Assuming there are 4 possible actions
        self.reward = 0

    @only_active_nodes
    def do_routine(self, time, ids, ok_move, logger):
        """
        :param logger:
        :param time: current time
        :param ids: the length of the nodes in the network to choose from.
        :param ok_move: its time to move.
        :return:
        Move node, calculate its utility, and generate packets if it was time to generate.
        """
        if ok_move:
            self.move(time)
            self.calculate_utility(ids)
        self.generate_packet(time, ids, logger)
        self.time_series_energy.append(self.energy_source.energy)
        self.successfully_received_messages.append(0)

    def calculate_delta_energy(self, round_np):
        """
        calculate the change of energy
        """
        if not self.alive:
            self.delta_energy_in_timestep = 0
            return
        if round_np < cf.WINDOW_DELTA_ENERGY:
            index_change = round_np
            index_used = 0
        else:
            index_change = round_np % cf.WINDOW_DELTA_ENERGY
            index_used = index_change

        self.delta_energy_in_timestep = (
            self.last_energy[index_used] - self.energy_source.energy
        )
        self.last_energy[index_change] = self.energy_source.energy

    @only_active_nodes
    def move(self, time):
        """
        :param time: current time
        :return:
        updates positions, and calculate the velocity and angle of movement.
        """
        new_x, new_y = self.movement_obj(time, self.pos_x, self.pos_y) # type: ignore

        self.org_pos_x = self.pos_x
        self.org_pos_y = self.pos_y

        self.pos_x = new_x
        self.pos_y = new_y

        velocity_x = self.pos_x - self.org_pos_x
        velocity_y = self.pos_y - self.org_pos_y
        self.velocity = np.sqrt(np.abs(velocity_x) ** 2 + np.abs(velocity_y) ** 2)
        self.angle_degree = np.arctan2(velocity_y, velocity_x)  # in radian

        if self.angle_degree < 0:
            self.angle_degree += 2 * math.pi

        self.ray = [self.velocity, self.angle_degree]

    def find_next_hob(self, packet):
        """
        :param packet: packet message.
        :return:
        either one of the neighbors if it was the destination, or the CH.
        """
        """
        Routing Protocol:
        1- if node is cluster member --> send to role.
        2- if node is a cluster head OR node is a cluster forwarding node:
            A. if the destination is one of the one hop neighbors sendto it.
            B. else --> send to all neighbors.
        """
        # this case should never happen
        if packet.destination == self.id:
            return self.id
        if self.role == -1 or self.forwarding_node:
            # the destination is one of my neighbors.
            if packet.destination in self.neighbors:
                idx = self.neighbors.index(packet.destination)
                return [self.neighbors[idx]]
            else:
                return self.neighbors

        # if role == -1: --> this packet should be dropped, if role != -1 --> send the CH id.
        return [self.role]

    def find_next_hob_greedy(self, packet, nodes):
        """
        :param packet: packet message.
        :return:
        either one of the neighbors if it was the destination, or the CH.
        """
        # this case should never happen
        if packet.destination == self.id:
            return [self.id]

        # source and destination are neighbors
        if packet.destination in self.neighbors:
            idx = self.neighbors.index(packet.destination)
            return [self.neighbors[idx]]
        else:
            if self.role == -1:
                nearest_cluster = find_nearest_cluster_for_node(nodes, packet.destination)
                if nearest_cluster is not None:
                    if packet.destination in self.ch_table:
                        return [nearest_cluster.id]
                    else:
                        gateways = get_gateways(nodes)
                        neighbor_ids = set(nearest_cluster.neighbors)  # Convert to a set for faster lookup
                        gateway_neighbors = [gateway.id for gateway in gateways if gateway.id in neighbor_ids]
                        return gateway_neighbors
            else:
                if not self.is_gateway:
                    return list(self.ch_table.keys())
                else:
                    if packet.destination in self.gateway_table:
                        return list(self.ch_table.keys())
                    else:
                        return list(self.ch_table.keys())

        # if role == -1: --> this packet should be dropped, if role != -1 --> send the CH id.
        return [self.role]

    def flatten_list(self, nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(self.flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    def lar_route_discovery(self, destination_node, node_list, time_elapsed, max_hops=10):
        last_known_x = destination_node.pos_x
        last_known_y = destination_node.pos_y
        destination_speed = destination_node.velocity
        if max_hops == 0:
            return [self.role]
        route = []
        if destination_node.id in self.neighbors:
            return [destination_node.id]
        request_zone_nodes = [node for node in node_list if
                              is_in_request_zone(node, self.pos_x, self.pos_y, last_known_x, last_known_y, destination_speed, time_elapsed)]
        for node in request_zone_nodes:
            if node.id in self.neighbors:
                route.append(node.id)
                if node.id == destination_node.id:
                    return route
                else:
                    next_route = node.lar_route_discovery(destination_node, node_list, time_elapsed,max_hops-1)
                    if next_route:
                        route.extend(next_route)
                        return route
        return route if route else [self.role]

    def find_next_hob_q_learning(self, packet, nodes_list, step, id_for_load, phase_type='Train'):
        available_action = get_action_for_rl(self, nodes_list)
        if phase_type=='Train':
            action = self.q_learning.choose_action(packet.destination, self.neighbors, available_action, self.reward) # type: ignore
        else:
            action = self.q_learning.choose_action_test_phase(packet.destination, self.neighbors, available_action, self.reward, id_for_load) # type: ignore
        return action, step+1

    def head(self):
        return self.role == -1

    def consume_energy(self, distance, msg_length):
        energy = cf.E_ELEC
        if distance > cf.THRESHOLD_DIST:
            energy += cf.E_MP * (distance**4)
        else:
            energy += cf.E_FS * (distance**2)
        energy *= msg_length
        self.energy_source.consume(energy)

        self.alive = False if self.energy_source.energy == 0 else True

    @only_active_nodes
    def generate_packet(self, time, ids, logger):
        """
        :param logger:
        :param time: current time
        :param ids: nodes ids (in network)
        :return:
        generate a valid packet, and insert it into the buffer.
        """
        if self.packet_generation_moment <= time:
            # print(f"generate packet in node {self.id}")
            while True:
                packet = Packet(
                    source=self.id,
                    packet_counter=self.packets_counter,
                    destination=cf.seed_simulator.randint(ids),
                    length=cf.MSG_LENGTH,
                    header_length=cf.HEADER_LENGTH,
                    generation_moment=time,
                    expiration_date=time + cf.EXPIRATION_LIMIT,
                )
                if packet.is_valid(time):
                    self.buffer += packet
                    break
            self.packet_generation_moment += cf.seed_simulator.uniform(
                cf.LOWER_BOUND_PACKET_GENERATION, cf.UPPER_BOUND_PACKET_GENERATION
            )
            logger.packets_logger[packet.id] = [
                False,
                False,
            ]  # initial drop and arrived status.
            self.packets_counter += 1
        # else:
        #     print(f"Not generate packet in node {self.id}")

    @only_active_nodes
    def calculate_utility(self, num_nodes):
        """
        :return:
        calculate utility four factors.
        --> equations 7 -> 18 in the article.
        """
        self.s1 = residual_energy(self.energy_source.energy)
        self.s2 = self.nb_neighbors / (num_nodes - 1)
        self.s3 = velocity_similarity(self.ray, self.s_table)
        self.s4 = min(
            cf.CLIPPING_THRESHOLD_S4,
            link_holding_time(self, self.s_table, self.p_table),
        )
        self.utility = sum(
            self.weights * np.array([self.s1, self.s2, self.s3, self.s4])
        )

    def contain(self, ch):
        """
        :param ch: node
        :return:
        True if the node is contained in all the tables of this node (we have its needed info)
        """
        return (
            (ch in self.p_table.keys())
            and (ch in self.w_table.keys())
            and (ch in self.ch_table.keys())
        )

    def maintain_cluster_membership(self, time):
        """
        :return:
        function to ensure cluster membership
        """
        # Case 0:
        # its a CH and has members.
        if self.role == -1:
            return -1

        # Case 1:
        # Has no connection to a cluster head.
        if len(self.ch_table) == 0:
            self.count_role += 1
            self.role = self.id
            return -1

        # Case 2:
        # Has a cluster head, but missing information about it.
        if self.role != self.id and not self.contain(self.role):
            self.count_role += 1
            self.ch_table = {
                k: v
                for k, v in sorted(
                    self.ch_table.items(), key=lambda item: item[1], reverse=True
                )
            }
            for ch in self.ch_table.keys():
                if self.contain(ch):
                    if (
                        calculate_distance_point(
                            self.p_table[ch][0][0],
                            self.p_table[ch][0][1],
                            self.pos_x,
                            self.pos_y,
                        )
                        < cf.COVERAGE_RADIUS
                    ):
                        return ch
            self.role = self.id
            self.time_out_of_cluster = time
            return -1  # doesnt send a join request.

        # Case 3:
        # It is not a cH, and doesn't have a CH or lost connection with it.
        if (
            self.role == self.id
            or calculate_distance_point(
                self.p_table[self.role][0][0],
                self.p_table[self.role][0][1],
                self.pos_x,
                self.pos_y,
            )
            > cf.COVERAGE_RADIUS
        ):
            self.count_role += 1
            self.ch_table = {
                k: v
                for k, v in sorted(
                    self.ch_table.items(), key=lambda item: item[1], reverse=True
                )
            }
            for ch in self.ch_table.keys():
                if ch in self.p_table.keys():
                    if (
                        calculate_distance_point(
                            self.p_table[ch][0][0],
                            self.p_table[ch][0][1],
                            self.pos_x,
                            self.pos_y,
                        )
                        < cf.COVERAGE_RADIUS
                    ):
                        return ch

            self.role = self.id
            return -1

        # Case 4:
        # No problem with the connection between the node and its CH.
        return self.role

    def connection_with_member(self, member):
        """
        :param member: node
        :return:
        True if the node (member) still close to this node (its cluster head)
        """
        return member in self.p_table and member in self.neighbors

    def check_hanging_nodes(self, time):
        """
        If a node stayed hanging without a cluster for a certain amount of time, it becomes a CH.
        """
        if ((time - self.time_out_of_cluster) > cf.NODES_HANGING_PERIOD) and (
            self.time_out_of_cluster != -1
        ):

            self.role = -1
            self.count_role += 1
            self.ch_time = time
            self.time_out_of_cluster = -1

    def maintain_members(self, time):
        """
            1- delete members who are no longer in the cluster.
            2- if there was no members --> it shouldn't be a CH.
        :return:
        """
        copy_m = copy.copy(self.members)
        for member in copy_m:
            if not self.connection_with_member(member):
                self.members.remove(member)
                if member in self.forwarding_members:
                    self.forwarding_members.remove(member)

        # node is CH, but has no members while it has neighbors.
        if len(self.members) == 0 and len(self.neighbors) > 0 and self.role == -1:
            if cf.seed_simulator.rand() < 0.7:
                self.role = self.id
                self.count_role += 1
                self.time_out_of_cluster = time

    def clear_old_messages_data(self, time, table_name):
        """
        :param time: current time.
        :param table_name: what to clear
        :return:
        clear the old data for the given tables, (this function is called after receiving new info)
        """
        time -= 1  # -1 because the time stamp for the sending not the receiving.
        if table_name == "hello":
            copy_p = {**self.p_table}

            for key, val in copy_p.items():
                time_stamp = val[1]
                if time_stamp < time:
                    self.p_table.pop(key)
                    self.s_table.pop(key)
                    self.role_table.pop(key)

        elif table_name == "utility":
            copy_w = {**self.w_table}
            for key, val in copy_w.items():
                time_stamp = val[1]
                if time_stamp < time:
                    self.w_table.pop(key)

        elif table_name == "ch":
            copy_ch = {**self.ch_table}
            for key, val in copy_ch.items():
                time_stamp = val[1]
                if time_stamp < time:
                    self.ch_table.pop(key)

    def if_ch(self, time):
        if len(self.w_table) == 0:
            return
        ws = list(map(lambda x: x[0], self.w_table.values()))
        self.role = -1 if max(ws) <= self.utility else self.role  # if this node is better than all its
        # neighbors.
        if self.role == -1:
            self.ch_time = time # when did this node became CH.
            self.time_out_of_cluster = -1 # initial value.
        else:
            self.time_out_of_cluster = time

    def define_role(self, time):
        """
        :param time: current time.
        :return:
        This function checks and defines the role of the node (ordinary node, CH, or gateway).
        A node becomes a gateway if it has a neighbor that is not part of its members (indicating it's connected to another cluster).
        """

        # Reset the gateway status and clear the gateway table at the beginning of the function
        self.is_gateway = False
        self.gateway_table = {}

        # Iterate through the w_table to check connections to nodes not in members (i.e., potential neighbors from other clusters)
        for neighbor_id, neighbor_info in self.w_table.items():
            if neighbor_id not in self.members and self.role != -1:  # Check if the neighbor is not a member of this node's cluster and it's not CH
                self.gateway_table[neighbor_id] = neighbor_info  # Add the neighbor to the gateway_table
                self.is_gateway = True  # Mark the node as a gateway

        # Determine if the node should remain or become a CH based on utility
        ws = list(map(lambda x: x[0], self.w_table.values()))
        if len(ws) == 0:
            return

        if max(ws) <= self.utility:
            if self.role != -1:  # If the node was previously a CH
                self.ch_time = time
                self.count_role += 1  # Change role to ordinary node
            self.role = -1
            self.time_out_of_cluster = -1
        else:  # There are better nodes, so this node will no longer be a CH and will have no members
            if self.role == -1:
                self.count_role += 1  # Change role to CH
                self.role = self.id
                self.time_out_of_cluster = time
                self.members.clear()

    def receive(self, message, measures_logger=None, time=None):
        """
        :param message: the received message.
        :param measures_logger: logger.
        :param time: current time
        :return:
        main function for receive --> choose the receiving function based on the message type.
        """
        logging.debug("node %d receiving." % self.id)
        m_type = "self.receive_" + message[0] + "_message"
        self.func = eval(m_type)
        if measures_logger is not None:
            returned_msg = self.func(message, measures_logger, time)
            return returned_msg
        else:
            self.func(message)
            return None

    # Each function of the receiving function is for a message type.
    def receive_hello_message(self, message):
        source = message[1]
        content = message[3]  # after type, source, dist
        self.p_table[source] = (content["position"], content["timestamp"])
        self.s_table[source] = (content["speed"], content["timestamp"])
        self.role_table[source] = (content["role"], content["timestamp"])

    def receive_utility_message(self, message):
        source = message[1]
        content = message[3]  # after type, source, dist
        self.w_table[source] = (content["utility"], content["timestamp"])

    def receive_ch_message(self, message):
        cluster_head_id = message[1]  # after type
        cluster_head_utility = message[3]["utility"]  # after type, source, dist.

        self.ch_table[cluster_head_id] = (cluster_head_utility, message[3]["timestamp"])

    def receive_join_request_message(self, message):
        member_id = message[1]  # after type
        forwarding_node = message[3]["forwarding_node"]
        link = link_holding_time(
            self,
            {member_id: self.s_table[member_id]},
            {member_id: self.p_table[member_id]},
        )
        if link >= cf.LINK_THRESHOLD and self.role == -1:
            self.new_members.append(member_id)
            if forwarding_node:
                self.forwarding_members.append(member_id)

    def receive_join_response_message(self, message):
        self.role = message[1]
        self.count_role += 1
        self.time_out_of_cluster = -1

    def receive_packet_message(self, message, measures_logger, time):
        """
        :param message: the received packet.
        :param measures_logger: logger
        :param time: current time.
        :return:
        calculate energy consumption, then either the packet destination is this node --> done.
        else: add it to buffer to be sent later if its still valid or drop it and ad to the logger.
        """
        content = message[3]
        destination = content["original_destination"]
        packet_id = content["id"]
        path = content['path']

        # energy model for receiver
        energy = cf.E_ELEC * content["size"]
        self.energy_source.consume(energy)

        if destination == self.id:  # message arrived.
            # print("arrived packet!")
            measures_logger.num_arrived += 1
            measures_logger.packets_logger[packet_id][1] = True  # arrived is True.
            measures_logger.e2e_delay += time - content["moment_of_generation"]
            self.successfully_received_messages[-1] += 1
            return True  # yes send an affirmation message that the packet arrived.

        else:
            # resend the packet, this node is not the destination.
            packet = Packet(
                source=self.id,
                packet_counter=0,
                destination=destination,
                length=content["size"],
                header_length=cf.HEADER_LENGTH,
                generation_moment=content["moment_of_generation"],
                expiration_date=content["expiration_date"],
                id_=packet_id,
                path=path
            )

            if packet.is_valid(time):
                if self.buffer.has_space():
                    self.buffer += packet
                else:
                    measures_logger.packets_logger[packet_id][0] = True  # drop is True.
                    measures_logger.num_dropped_packets_buffer_overflow += 1
                    #print("packet dropped! buffer overflow")
            else:
                measures_logger.packets_logger[packet_id][0] = True  # drop is True.
                measures_logger.num_dropped_packets_expiration += 1
                print("packet dropped! because its no longer valid")
            return False

    def receive_affirmation_message(self, message, measures_logger, time):
        content = message[3]
        path = content['path']

        # energy model for receiver
        energy = cf.E_ELEC * content["size"]
        self.energy_source.consume(energy)

        if len(path) == 0:  # message arrived.
            self.count_affirmations += 1
            return False
        else:
            return True  # send this affirmation to its next destination.
