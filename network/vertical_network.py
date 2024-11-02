from network.network import Network
import config as cf
from network.plane import Plane
from network.vertical_node import VerticalNode
from utils.utils import calculate_distance


class VerticalNetwork(Network):
    """
    for vertical networks, planes and their mobilities should be considered.
    """
    def __init__(self):
        super().__init__()
        self.planes = []
        for index_plane in range(len(cf.LIST_INFO_PLANES)):
            self.planes.append(Plane(**cf.LIST_INFO_PLANES[index_plane]))

        self.nodes = []
        id = 0
        for index_plane in range(len(cf.LIST_NB_NODES_IN_PLANES)):
            for _ in range(cf.LIST_NB_NODES_IN_PLANES[index_plane]):
                self.nodes.append(VerticalNode(self.planes[index_plane], id))

                id += 1

    def closest_nodes(self, target_node):
        """
        get the nodes who fell in the range of coverage of the target node.
        also identify the forwarding nodes vertically and horizontally.
        ** a node is considered forwarding if one of its neighbors lies in a different cluster.
        *** a node is considered vertically forwarding if one of its neighbors lies in a different
         cluster and in a different plane.
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
                        if target_node.plane != node.plane:
                            target_node.vertical_forwarding_node = True
        target_node.nb_neighbors = len(target_node.neighbors)
