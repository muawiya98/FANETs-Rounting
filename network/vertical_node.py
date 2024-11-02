from network.node import Node
import numpy as np
import math


class VerticalNode(Node):
    def __init__(self, plane, id_, movement_type="random"):
        super().__init__(id_, movement_type)
        self.plane = plane
        self.vertical_forwarding_node = False
        self.planes_table = {}

    def receive_hello_message(self, message):
        # only vertical forwarding info was added.

        source = message[1]
        content = message[3]  # after type, source, dist
        self.p_table[source] = (content["position"], content["timestamp"])
        self.s_table[source] = (content["speed"], content["timestamp"])
        self.role_table[source] = (content["role"], content["timestamp"])
        self.planes_table[source] = (content["plane"], content["timestamp"])

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

    def move(self, time):
        """
        :param time: current time
        :return:
        updates positions, and calculate the velocity and angle of movement.
        """
        new_x, new_y = self.movement_obj(time, self.pos_x, self.pos_y, self.plane.name)

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