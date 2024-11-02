from network.energy_sourse import PluggedIn
from network.node import Node
import  config as cf

class GCDNode(Node):
    def __init__(self, id_, movement_type="fixed"):
        """
        Initialize a GCDNode (Ground Control Device).
        A GCD is typically stationary, so the movement type is fixed by default.
        :param id_: id for the node, used to identify the GCD.
        :param movement_type: the movement type of the GCD, default is 'fixed'.
        """
        super().__init__(id_, movement_type)
        self.pos_x = cf.seed_simulator.uniform(0, cf.AREA_WIDTH)
        self.pos_y = cf.seed_simulator.uniform(0, cf.AREA_LENGTH)

        # Additional attributes specific to GCD
        self.is_control_station = True  # Indicates this node is a GCD
        self.controlled_uavs = []  # List of UAVs controlled by this GCD

        # GCDs may have different energy management or communication requirements
        self.energy_source = PluggedIn()  # Assuming GCDs are connected to a constant power source

    def move(self, time):
        """
        Overriding the move function. GCDs don't move in a typical setup.
        This function ensures GCDs remain stationary.
        """
        # GCDs do not move, so their position remains constant
        pass

    def assign_uav(self, uav_node):
        """
        Assign a UAV to be controlled by this GCD.
        :param uav_node: the UAV node to be controlled.
        """
        if uav_node not in self.controlled_uavs:
            self.controlled_uavs.append(uav_node)
            uav_node.control_station = self  # Set this GCD as the control station for the UAV