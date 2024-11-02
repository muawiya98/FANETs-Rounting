from network.node import *
import config as cf


class ControlMessages(object):
    def __init__(self):
        self.func = None
        self.states = {"hello": "utility", "utility": "hello"}
        self.informative_messages = ["hello", "utility", "ch"]
        # because we need the first message to be a hello message and we send the self.states[self.current]

        self.current_sent = "utility"

    def exchange_messages(
        self, network, time, event_handler, time_to_exchange_hellos, time_to_exchange_ch
    ):
        """
        :param network: nodes.
        :param time: current time.
        :param event_handler: object to handle messages.
        :param time_to_exchange_hellos: the right time to send hello and utility messages.
        :param time_to_exchange_ch: the right to re cluster.
        :return:
        listen to new messages, do cluster maintain, send join_response to new members , send ch messages for new
        candidate members
        """
        self.listening(network, time, event_handler, time_to_exchange_ch == 1)
        self.send_join_response_messages(network, time, event_handler)
        self.maintain_clusters(network, time)

        # only at the beginning --> send once and then use the given rate.
        if (time_to_exchange_ch == 0 and time != 0) or time == cf.INITIAL_RE_CLUSTERING:
            self.send_ch_messages(network, time, event_handler, "re_cluster")
        else:
            # only to alert new neighbors with the nearby CH.
            self.send_ch_messages(network, time, event_handler)

        if time_to_exchange_hellos:
            self.func = eval(
                "self.send_" + self.states[self.current_sent] + "_messages"
            )
            self.func(network, time, event_handler)

    @staticmethod
    def first_time(network):
        for node in network:
            if len(node.ch_table) > 0:
                return False
        return True

    """
    All the sending functions does the same thing, create a message and send it.
    """

    @staticmethod
    def send_join_request_messages(network, time, event_handler):
        """
        :return:
        Checks if the node still belongs to its cluster, and if the connection was lost, send a join request to another
        CH
        """
        m_type = "join_request"
        for node in network:
            if node.alive:
                chosen_ch = node.maintain_cluster_membership(time)

                if (
                    chosen_ch != -1 and chosen_ch != node.role
                ):  # if the node doesn't belong to a cluster.
                    node.messaging_obj(
                        m_type,
                        node.id,
                        chosen_ch,
                        time,
                        node.forwarding_node,
                        event_handler,
                    )

                    # node shouldn't follow this CH any more..
                    node.role = node.id
                    node.consume_energy(calculate_distance(node, network[chosen_ch]),
                                        cf.CONTROL_MSG_LENGTH)

    @staticmethod
    def send_join_response_messages(network, time, event_handler):
        """
        for all new members, send a response to them to join the cluster.
        """
        m_type = "join_response"
        for node in network:
            if node.alive:
                for member in node.new_members:
                    node.messaging_obj(m_type, node.id, member, time, event_handler)
                    node.consume_energy(calculate_distance(node, network[member]),
                                        cf.CONTROL_MSG_LENGTH)
                    node.members.append(member)
                node.new_members.clear()

    @staticmethod
    def maintain_clusters(network, time):
        """
        :return:
        maintain members in clusters and delete the members who left the cluster.
        """
        for node in network:
            if node.alive:
                # time to check validity
                if (time - node.ch_time) % cf.MAINTAIN_CH_VALIDITY == 0:
                    node.maintain_members(time)
                node.check_hanging_nodes(time)

    def send_hello_messages(self, network, time, event_handler):
        """
        send node info to its neighbors.
        """
        for node in network:
            for neighbor_id in node.neighbors:
                if node.alive:
                    node.messaging_obj(
                        "hello",
                        node.id,
                        neighbor_id,
                        [node.pos_x, node.pos_y],
                        node.ray,
                        node.role,
                        node.plane.name,
                        time,
                        event_handler,
                    )
                    node.consume_energy(calculate_distance(node, network[neighbor_id]),
                                        cf.CONTROL_MSG_LENGTH)
        self.current_sent = "hello"

    def send_utility_messages(self, network, time, event_handler):
        """
        node sends its utility value to its neighbors.
        """
        m_type = "utility"
        for node in network:
            node.calculate_utility(
                len(network)
            )  # based on the new received information.

        for node in network:
            for neighbor_id in node.neighbors:
                # only send if the node were alive , and this neighbor is in my plane.
                if node.alive and node.plane.name == node.planes_table[neighbor_id][0]:
                    node.messaging_obj(
                        m_type, node.id, neighbor_id, node.utility, time, event_handler
                    )
                    node.consume_energy(calculate_distance(node, network[neighbor_id]),
                                        cf.CONTROL_MSG_LENGTH)
        self.current_sent = m_type

    def send_ch_messages(self, network, time, event_handler, m_type=None):
        """
        :param network.
        :param time.
        :param event_handler.
        :param m_type: if the type was re clustering --> define each node role, or if it was none --> just send CH
         declaration messages to new neighbors.
        :return:
        if its re-cluster time, define nodes role, and define CHs and send CH messages.
        if its not re-cluster time, just send CH messages to new potential members.
        """
        if m_type == "re_cluster":
            if self.first_time(network):
                for node in network:  # to be deleted.
                    node.if_ch(time)  # based on the new received information.
            else:
                for node in network:
                    node.define_role(time)  # based on the new received information.`

        for node in network:
            if node.role == -1:  # head
                for neighbor_id in node.neighbors:
                    if m_type != "re_cluster" and neighbor_id in node.members:
                        continue
                    # only send if the node were alive , and this neighbor is in my plane.
                    if neighbor_id in node.planes_table.keys():
                        if (
                            node.alive
                            and node.plane.name == node.planes_table[neighbor_id][0]
                        ):
                            node.messaging_obj(
                                "ch",
                                node.id,
                                neighbor_id,
                                node.utility,
                                time,
                                event_handler,
                            )
                            node.consume_energy(calculate_distance(node, network[neighbor_id]),
                                                cf.CONTROL_MSG_LENGTH)

    def listening(self, network, time, event_handler, listen_re_clustering):
        """
        :param network.
        :param time.
        :param event_handler.
        :param listen_re_clustering:
        :return:
        this function listen to the messages who just arrived.
        """
        types = []
        while not event_handler.is_empty():

            message = event_handler.get_top_message()
            if message[0] in self.informative_messages and message[0] not in types:
                types.append(message[0])

            # only receive if there are still connected.
            if (
                calculate_distance(network[message[1]], network[message[2]])
                <= cf.COVERAGE_RADIUS and network[message[2]].alive
            ):
                if (
                    network[message[1]].role == message[2]
                    and message[0] == "join_request"
                ):
                    continue
                network[message[2]].receive(message)

        # for currently received messages, delete old knowledge.
        for t in types:
            for node in network:
                if t == "ch" and not listen_re_clustering:
                    continue
                node.clear_old_messages_data(time, t)
