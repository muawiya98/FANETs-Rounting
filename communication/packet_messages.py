from network.node import *
import config as cf
from tqdm import tqdm


class PacketMessages:
    @staticmethod
    def send_packets(network, time, event_handler, measures_logger):
        for node in network:

            for _ in range(cf.NUMBER_OF_SEND_PACKETS):
                if not node.alive:
                    break
                packet = node.buffer.get_top_packet()
                if packet is not None:
                    node.buffer -= 0  # delete the top of the buffer (id =0)

                    # if the packet was sent previously
                    packet_sent = node.packet_id_sended.get(packet.id, False)
                    if packet_sent:
                        measures_logger.packets_logger[packet.id][0] = True
                        continue
                    # get the list of possible destinations:
                    destination = node.find_next_hob(packet)
                    # print(f"packet.id {packet.id}, destination {destination}, packet.destination {packet.destination}")
                    # if the node stands alone with no neighbors:
                    if len(destination) == 1 and destination[0] == node.id:
                        measures_logger.packets_logger[packet.id][
                            0
                        ] = True  # the value Dropped = True for this packet.
                        # measures_logger.num_dropped_packets_no_connection += 1
                        # print("packet dropped! no connection")
                    else:
                        # send the packet to all neighbors.
                        node.packet_id_sended[packet.id] = True
                        packet.path.append(node.id)  # keep the path for the affirmation.
                        for des in destination:
                            # only send to CH or forwarding node because they can carry the message:
                            if network[des].role == -1 or network[des].forwarding_node:

                                node.messaging_obj(
                                    "packet",
                                    node.id,
                                    des,
                                    packet.id,
                                    packet.path,
                                    packet.destination,
                                    packet.length,
                                    packet.generation_moment,
                                    packet.expiration_date,
                                    time,
                                    event_handler,
                                )
                                node.consume_energy(
                                    calculate_distance(node, network[des]),
                                    packet.length,
                                )
                            if not node.alive:
                                break
                else:
                    break

    @staticmethod
    def send_packets_greedy(network, time, event_handler, measures_logger):
        for node in network:

            for _ in range(cf.NUMBER_OF_SEND_PACKETS):
                if not node.alive:
                    break
                packet = node.buffer.get_top_packet()
                if packet is not None:
                    node.buffer -= 0  # delete the top of the buffer (id =0)

                    # if the packet was sent previously
                    packet_sent = node.packet_id_sended.get(packet.id, False)
                    if packet_sent:
                        measures_logger.packets_logger[packet.id][0] = True
                        continue
                    # get the list of possible destinations:
                    destination = node.find_next_hob_greedy(packet,network)
                    # if the node stands alone with no neighbors:

                    if len(destination) == 1 and destination[0] == node.id:
                        measures_logger.packets_logger[packet.id][
                            0
                        ] = True  # the value Dropped = True for this packet.
                        # measures_logger.num_dropped_packets_no_connection += 1
                        # print("packet dropped! no connection")
                    else:
                        # send the packet to all neighbors.
                        node.packet_id_sended[packet.id] = True
                        packet.path.append(node.id)  # keep the path for the affirmation.
                        for des in destination:
                            # only send to CH or forwarding node because they can carry the message:
                            if network[des].role == -1 or network[des].forwarding_node:

                                node.messaging_obj(
                                    "packet",
                                    node.id,
                                    des,
                                    packet.id,
                                    packet.path,
                                    packet.destination,
                                    packet.length,
                                    packet.generation_moment,
                                    packet.expiration_date,
                                    time,
                                    event_handler,
                                )
                                node.consume_energy(
                                    calculate_distance(node, network[des]),
                                    packet.length,
                                )
                            if not node.alive:
                                break
                else:
                    break

    # @staticmethod
    # def send_packets_q_learning(network, time, event_handler, measures_logger):
    #     for node in network:
    #         for _ in range(cf.NUMBER_OF_SEND_PACKETS):
    #             if not node.alive:break
    #             packet = node.buffer.get_top_packet()
    #             if packet is None:break
    #             node.buffer -= 0  # delete the top of the buffer (id =0)
    #             packet_sent = node.packet_id_sended.get(packet.id, False)
    #             if packet_sent:
    #                 measures_logger.packets_logger[packet.id][0] = True
    #                 continue
    #             previous_neighbors = set(node.neighbors)
    #             destination = node.find_next_hob_q_learning(packet, network)
    #             if len(destination) == 1 and destination[0] == node.id:
    #                 measures_logger.packets_logger[packet.id][0] = True
    #                 node.update_q_learning([], False,packet,network)
    #             else:
    #                 node.packet_id_sended[packet.id] = True
    #                 packet.path.append(node.id)
    #                 acknowledge_received = False
    #                 for des in destination:
    #                     if network[des].role == -1 or network[des].forwarding_node:
    #                         send_time = time
    #                         node.messaging_obj("packet",node.id,
    #                                             des, packet.id, packet.path,
    #                                             packet.destination, packet.length,
    #                                             packet.generation_moment, packet.expiration_date, time, event_handler,)
    #                         node.consume_energy(calculate_distance(node, network[des]),
    #                                             packet.length,)
    #                         acknowledge_received = True
    #                     if not node.alive:break

    #                 current_neighbors = set(node.neighbors)
    #                 new_neighbors = current_neighbors - previous_neighbors
    #                 node.update_q_learning(list(new_neighbors), acknowledge_received,packet,network)

    @staticmethod
    def listen(network, time, event_handler, measures_logger):
        """
        function to read messages from the event handler and for every node to receive its messages.
        """
        affirmation_msg = []
        while not event_handler.is_empty():

            message = event_handler.get_top_message()

            # only receive if there are still connected.
            if (
                calculate_distance(network[message[1]], network[message[2]])
                <= cf.COVERAGE_RADIUS
                and network[message[2]].alive
            ):

                returned_msg = network[message[2]].receive(message, measures_logger, time)
                # if the packet arrived and an affirmation message should be sent:
                if returned_msg:
                    affirmation_msg.append([network[message[2]], message])
            else:
                # print(
                #     "packet message dropped, dis",
                #     calculate_distance(network[message[1]], network[message[2]]),
                # )
                measures_logger.num_dropped_packets_distance += 1

        for am in affirmation_msg:
            node = am[0]
            message = am[1]
            send_affirmation = node.receive_affirmation_message(message, measures_logger, time)
            # if the affirmation message didn't reach its goal, pass it over.
            if send_affirmation:

                destination = message[3]['path'].pop()
                node.messaging_obj("affirmation",
                                    node.id,
                                    destination,
                                    message[3]['id'],
                                    message[3]['path'],
                                    message[3]['expiration_date'],
                                    message[3]['size'],
                                    time,
                                    event_handler)

                node.consume_energy(calculate_distance(node, network[destination]),
                                    message[3]['size'])
