import math
import config as cf
import numpy as np


def find_nearest_cluster_for_node(nodes, target_node_id):
    """
    Find the nearest cluster head for the node with the given target_node_id.
    """
    target_node = next((node for node in nodes if node.id == target_node_id), None)
    if not target_node:
        # print(f"Node with ID {target_node_id} not found.")
        return None

    cluster_heads = get_cluster_heads(nodes)
    if not cluster_heads:
        print("No cluster heads available in the network.")
        return None

    nearest_ch = None
    min_distance = float('inf')

    for ch in cluster_heads:
        distance = calculate_distance(target_node, ch)
        if distance < min_distance:
            min_distance = distance
            nearest_ch = ch

    return nearest_ch

def get_cluster_heads(nodes):
    """
    Get all nodes that are cluster heads (CH).
    """
    return [node for node in nodes if node.role == -1]

def get_gateways(nodes):
    """
    Get all nodes that are cluster heads (CH).
    """
    return [node for node in nodes if node.is_gateway]

def calculate_distance(node1, node2):
    """Calculate the Euclidean distance between two nodes."""
    x1 = node1.pos_x
    y1 = node1.pos_y
    x2 = node2.pos_x
    y2 = node2.pos_y
    return calculate_distance_point(x1, y1, x2, y2)


def calculate_distance_point(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def only_active_nodes(func):
    """
    This is a decorator. It wraps all energy consuming methods to
    ensure that only active nodes execute this method. Also it automatically
    calls the battery.
    """

    def wrapper(self, *args, **kwargs):
        if self.alive:
            func(self, *args, **kwargs)
            return 1
        else:
            return 0

    return wrapper


def calculate_nb_clusters(avg_distance_to_base_station):
    """Calculate the optimal number of clusters for FCM."""
    term1 = math.sqrt(cf.NB_NODES) / (math.sqrt(2 * math.pi))
    term2 = cf.THRESHOLD_DIST
    term3 = cf.AREA_WIDTH / (avg_distance_to_base_station ** 2)
    return int(term1 * term2 * term3)


def residual_energy(energy_t):
    return energy_t/cf.INITIAL_ENERGY


def velocity_similarity(ray_i, neighbors_speed_rays):
    """
    :param ray_i: the speed ray of the original node.
    :param neighbors_speed_rays: the neighbors speeds and angles.
    :return:
    equations 9 --> 12 in the article.
    returns the velocity similarity between the node and its neighbors.
    """
    v_sim = []
    if len(neighbors_speed_rays) == 0:
        return 0
    for key, value in neighbors_speed_rays.items():
        ray = value[0]
        d = calculate_distance_point(ray_i[0], ray_i[1], ray[0], ray[1])
        v_sim.append(1/(1+d))

    return np.var(v_sim)


def link_holding_time(node, neighbors_speed_rays, neighbors_positions):
    """

    :param neighbors_positions:
    :param neighbors_speed_rays:
    :param node: original node
    :return:
    equations 13 --> 18 in the article.
    return the link holding time between the the node and its neighbors.
    """
    links = []
    for key in neighbors_speed_rays.keys():
        if key in neighbors_speed_rays.keys() and key in neighbors_positions.keys():
            a, b, c, d = calculate_LTHP_params(node.ray, [node.pos_x, node.pos_y], neighbors_speed_rays[key][0],
                                               neighbors_positions[key][0])
            xx = (a ** 2 + c ** 2)*(cf.COVERAGE_RADIUS ** 2)
            yy = (a*d - b*c)
            cc = (a*b + c*d)
            ccc = (a ** 2 + c ** 2)*(cf.COVERAGE_RADIUS ** 2) - ((a*d - b*c) ** 2)
            if ccc < 0:
                continue
            link = (-1*(a*b + c*d) + math.sqrt((a ** 2 + c ** 2)*(cf.COVERAGE_RADIUS ** 2) - ((a*d - b*c) ** 2)))\
                   / (a ** 2 + c ** 2)
            if link < 0:
                continue
            links.append(link)
    return np.mean(links)


def calculate_LTHP_params(node1_speed_ray, node1_position, node2_speed_ray, node2_position):
    """
    :return:
    helper function to calc the parameters to get the link holding.
    """
    a = node1_speed_ray[0] * math.cos(node1_speed_ray[1]) - node2_speed_ray[0] * math.cos(node2_speed_ray[1])
    b = node1_position[0] - node2_position[0]
    c = node1_speed_ray[0] * math.sin(node1_speed_ray[1]) - node2_speed_ray[0] * math.sin(node2_speed_ray[1])
    d = node1_position[1] - node2_position[1]
    return a, b, c, d


def calculate_per_time_unit(value_per_sec):
    return value_per_sec * cf.TIME_UNIT


def entropy(pi):
    """
    Defines the (discrete) distribution...
    pi is the probabilties of each value in vec i:
    pi = for every element in vec i: number_of_accurance/len(vec)
    """
    return - np.sum(pi * np.log2(pi))


def prop(vec):
    return list(map(lambda item: vec.count(item) / len(vec), np.unique(vec[:])))


def quantization(ls, mx, mn):
    quantized_values = []
    new_max = mx
    new_min = mn
    old_min = min(ls)
    old_max = max(ls)
    # print((old_max, old_min), ls,new_min)
    for old_value in ls:
        quantized_values.append(int(((old_value - old_min) * (new_max - new_min)) / max(0.00001, (old_max - old_min))) + new_min)

    return quantized_values


def determine_quadrant(x1, y1, x2, y2):
    """
    Determine which quadrant a neighbor is in relative to the current node (x1, y1).

    Returns:
        An integer representing the quadrant (1, 2, 3, or 4).
    """
    # Calculate the angle between the two points
    angle = math.atan2(y2 - y1, x2 - x1)
    # print(f"angle: {angle}: ")
    # print(f"y2 - y1, x2 - x1: {y2} - {y1}, {x2} - {x1}: ")
    # Map the angle to one of the four quadrants
    if 0 <= angle < math.pi / 2:
        return 0  # North-East
    elif math.pi / 2 <= angle <= math.pi:
        return 1  # North-West
    elif -math.pi <= angle < -math.pi / 2:
        return 2  # South-West
    elif -math.pi / 2 <= angle < 0:
        return 3


def get_farthest_neighbors(current_node, neighbor_nodes):
    """
    Cluster neighbors into four quadrants and filter to keep only the farthest node from the current node in each quadrant.

    current_node  : The current node object with 'x' and 'y' attributes.
    neighbor_nodes: A list of neighbor node objects, each with 'x' and 'y' attributes.

    Returns:
        A dictionary with the keys 1, 2, 3, 4 representing the four quadrants,
        and the values being the farthest node in that quadrant (if any).
    """
    # Dictionary to store the farthest neighbor in each quadrant
    farthest_neighbors = {0: None, 1: None, 2: None, 3: None}
    farthest_distances = {0: -1, 1: -1, 2: -1, 3: -1}

    for neighbor in neighbor_nodes:
        # Calculate the distance between the current node and the neighbor node
        distance = calculate_distance_point(current_node.pos_x, current_node.pos_y, neighbor.pos_x, neighbor.pos_y)
        # Determine which quadrant the neighbor is in
        quadrant = determine_quadrant(current_node.pos_x, current_node.pos_y, neighbor.pos_x, neighbor.pos_y)
        # print(f"neighbor.id : {neighbor.id}")

        # If this neighbor is farther than the current farthest in this quadrant, update it
        if distance > farthest_distances[quadrant]:
            farthest_distances[quadrant] = distance
            farthest_neighbors[quadrant] = neighbor # (neighbor, distance)
    # print(f"neighbor_nodes: {neighbor_nodes}")

    # print(f"farthest_neighbors: {farthest_neighbors}")
    return farthest_neighbors

def get_action_for_rl(node, nodes_list):
    neighbor_nodes = [n for n in nodes_list if n.id in node.neighbors]
    action=get_farthest_neighbors(node,neighbor_nodes)
    return action

def is_in_request_zone(node, source_x, source_y, last_known_x, last_known_y, speed, time_elapsed):
    """
    Determine if a node is within the request zone for the destination.
    The request zone is defined as a rectangle around the expected zone.

    node: The node being checked.
    source_x, source_y: The source node's location.
    last_known_x, last_known_y: The last known location of the destination node.
    speed: The maximum speed of the destination node.
    time_elapsed: The time since the last known location of the destination.

    Returns True if the node is within the request zone.
    """
    # Define the radius of the expected zone
    center_x, center_y, expected_radius = expected_zone(last_known_x, last_known_y, speed, time_elapsed)

    # Define the request zone as a rectangle
    x_min = min(source_x, center_x) - expected_radius
    x_max = max(source_x, center_x) + expected_radius
    y_min = min(source_y, center_y) - expected_radius
    y_max = max(source_y, center_y) + expected_radius


    # Check if the node is within this rectangular request zone
    return x_min <= node.pos_x <= x_max and y_min <= node.pos_y <= y_max

def expected_zone(last_known_x, last_known_y, speed, time_elapsed):
    """
    Calculate the expected zone of a destination node based on its last known location,
    speed, and the time elapsed since the last known position.

    The expected zone is a circular area with the radius determined by the speed and time elapsed.

    Returns:
        (center_x, center_y, radius) - the center of the expected zone and the radius.
    """
    radius = speed * time_elapsed
    return last_known_x, last_known_y, radius

def get_destination_node(nodes, destination_id):
    """
    Find and return the node with the given destination ID from a list of nodes.

    nodes: List of Node objects.
    destination_id: The ID of the destination node.

    Returns:
        The node object with the matching ID or None if not found.
    """
    for node in nodes:
        if node.id == destination_id:
            return node
    return None
