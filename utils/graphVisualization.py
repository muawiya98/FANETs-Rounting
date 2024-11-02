from random import choices
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, rgb_to_hsv, hsv_to_rgb, rgb2hex
import networkx as nx
import numpy as np
import config as cf
from network.node import Node
from utils.utils import calculate_per_time_unit

"""
the below functions is for either building the adjacency matrix to show clusters or neighbor
Or to plot them.
the plot functions: 
1- show_graph_with_labels: plot the graph for horizontal networks (and cam be used for vertical 
but with no colors to define each plane) with labels for the nodes role.
2- show_graph_heat_map: plot the amount of pressure on each node, the nodes with the darkest colors 
are the most busy.
3- show_graph_vertical_routing: plot vertical graph with a different color for each plane.
"""


def show_graph_with_labels(adjacency_matrix, fig_title, title, last, re_cluster):
    """
    show clusters
    adjacency_matrix : [adjacency, positions, role, lives, energy]
        adjacency shape (num_nodes, num_nodes) value 0 or 1 {Each CM connects to a CH}
        positions   is  [(x, y)] for every node
        role        is  []       for every node
        live        is  []       for every node
    fig_title : title of figure
    title : time of round
    last : true if timestep is last
    re_cluster : true if time of re cluster
    """
    plt.figure(fig_title)
    ax = plt.subplot()
    ax.clear()
    adjacency = adjacency_matrix[0]
    positions = adjacency_matrix[1]
    roles = adjacency_matrix[2]
    lives = adjacency_matrix[3]
    nodes = [node for node in range(len(adjacency[0]))]
    dictionary = {}
    for i in range(len(positions)):
        dictionary[i] = {"position": positions[i], "color": "#A0CBE2"}

    rows, cols = np.where(adjacency == 1)
    edges = zip(rows.tolist(), cols.tolist())

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.set_node_attributes(G, dictionary)
    labels = {n: "Ord" for n in G.nodes()}
    for i in range(len(roles)):
        if roles[i] == -1:
            G.nodes[i]["color"] = "#F7A1AE"
            labels[i] = "CH"
        elif roles[i] == i:
            G.nodes[i]["color"] = "#A0DD9A"
            labels[i] = "DM"

        if not lives[i]:
            G.nodes[i]["color"] = "#D9D9D9"
    pos = nx.get_node_attributes(G, "position")
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=500,
        node_color=[G.nodes[n]["color"] for n in G.nodes],
        width=2.0,
        labels=labels,
        edge_color="#262720",
    )
    plt.xlim([0, cf.AREA_WIDTH])
    plt.ylim([0, cf.AREA_LENGTH])
    plt.xlabel("X")
    plt.ylabel("Y")
    if re_cluster:
        plt.title("time " + str(calculate_per_time_unit(title)) + " Re cluster time!")
    else:
        plt.title("time " + str(calculate_per_time_unit(title)))
    limits = plt.axis("on")  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.pause(0.001)

    if last:
        plt.savefig("results/graph_" + fig_title + ".png")
        plt.savefig("results/graph_" + fig_title + ".svg")


def show_graph_heat_map(adjacency_matrix, fig_title, title, last, re_cluster):
    """
    show heat map for energy consumption
    adjacency_matrix : [adjacency, positions, role, lives, energy]
        adjacency shape (num_nodes, num_nodes) value 0 or 1 {Each CM connects to a CH}
        positions   is  [(x, y)] for every node
        role        is  []       for every node
        live        is  []       for every node
        energy      is  []       for every node
    fig_title : title of figure
    title : time of round
    last : true if timestep is last
    re_cluster : true if time of re cluster
    """
    plt.figure(fig_title)
    ax = plt.subplot()
    ax.clear()
    adjacency = adjacency_matrix[0]
    positions = adjacency_matrix[1]
    roles = adjacency_matrix[2]
    lives = adjacency_matrix[3]
    energy = adjacency_matrix[4]

    nodes = [node for node in range(len(adjacency[0]))]
    dictionary = {}
    for i in range(len(positions)):
        dictionary[i] = {"position": positions[i], "color": "#A0CBE2"}

    rows, cols = np.where(adjacency == 1)
    edges = zip(rows.tolist(), cols.tolist())

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.set_node_attributes(G, dictionary)
    labels = {n: "Ord" for n in G.nodes()}
    for i in range(len(roles)):
        color = rgb_to_hsv(hex2color("#FFFFFF"))
        color[1] = energy[i]
        G.nodes[i]["color"] = rgb2hex(hsv_to_rgb(color))
        if roles[i] == -1:
            labels[i] = "CH"
        elif roles[i] == i:
            labels[i] = "DM"

        if not lives[i]:
            G.nodes[i]["color"] = "#D9D9D9"
    pos = nx.get_node_attributes(G, "position")
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=500,
        node_color=[G.nodes[n]["color"] for n in G.nodes],
        width=2.0,
        labels=labels,
        edge_color="#262720",
    )
    plt.xlim([0, cf.AREA_WIDTH])
    plt.ylim([0, cf.AREA_LENGTH])
    plt.xlabel("X")
    plt.ylabel("Y")
    if re_cluster:
        plt.title("time " + str(calculate_per_time_unit(title)) + " Re cluster time!")
    else:
        plt.title("time " + str(calculate_per_time_unit(title)))
    limits = plt.axis("on")  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.pause(0.001)

    if last:
        plt.savefig("results/graph_" + fig_title + ".png")
        plt.savefig("results/graph_" + fig_title + ".svg")


def show_graph_vertical_routing(adjacency_matrix, fig_title, title, last, re_cluster):
    """
    to show nodes in every plane
    adjacency_matrix : [adjacency, positions, role, lives, energy]
        adjacency shape (num_nodes, num_nodes) value 0 or 1 {Each CM connects to a CH}
        positions   is  [(x, y)] for every node
        role        is  []       for every node
        live        is  []       for every node
        planes      is  []       for every node
    fig_title : title of figure
    title : time of round
    last : true if timestep is last
    re_cluster : true if time of re cluster
    """
    plt.figure(fig_title)
    ax = plt.subplot()
    ax.clear()
    adjacency = adjacency_matrix[0]
    positions = adjacency_matrix[1]
    roles = adjacency_matrix[2]
    lives = adjacency_matrix[3]
    planes = adjacency_matrix[4]

    nodes = [node for node in range(len(adjacency[0]))]

    colors = {}
    for i, index in enumerate(set(planes)):
        colors[index] = cf.PLANES_COLORS[i]

    dictionary = {}
    for i in range(len(positions)):
        dictionary[i] = {"position": positions[i], "color": "#A0CBE2"}

    rows, cols = np.where(adjacency == 1)
    edges = zip(rows.tolist(), cols.tolist())

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.set_node_attributes(G, dictionary)
    for i in range(len(roles)):
        G.nodes[i]["color"] = colors[planes[i]]
        if not lives[i]:
            G.nodes[i]["color"] = "#D9D9D9"

    pos = nx.get_node_attributes(G, "position")
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=500,
        node_color=[G.nodes[n]["color"] for n in G.nodes],
        width=2.0,
        edge_color="#262720",
    )
    plt.xlim([0, cf.AREA_WIDTH])
    plt.ylim([0, cf.AREA_LENGTH])
    plt.xlabel("X")
    plt.ylabel("Y")
    if re_cluster:
        plt.title("time " + str(calculate_per_time_unit(title)) + " Re cluster time!")
    else:
        plt.title("time " + str(calculate_per_time_unit(title)))
    limits = plt.axis("on")  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.pause(0.001)

    if last:
        plt.savefig("results/graph_" + fig_title + ".png")
        plt.savefig("results/graph_" + fig_title + ".svg")


def build_adjacency_neighbors(nodes, title, last, re_cluster):
    adjacency = np.zeros((len(nodes), len(nodes)))
    positions = [(node.pos_x, node.pos_y) for node in nodes if node.alive]
    role = [node.role for node in nodes]
    lives = [node.alive for node in nodes]

    for node in nodes:
        if node.alive:
            for ne in node.neighbors:
                adjacency[ne][node.id] = 1
                adjacency[node.id][ne] = 1
    adjacency_pos_role = [adjacency, positions, role, lives]
    # print(adjacency)
    show_graph_with_radius(nodes, adjacency_pos_role, "neighbors", title, last, re_cluster)


def build_adjacency_clusters(nodes, title, last, re_cluster, type=None):
    """
    title : time of round
    last : true if timestep is last
    re_cluster : true if time of re cluster
    type : "vertical", "cluster","heat map"
    """
    adjacency = np.zeros((len(nodes), len(nodes)))
    adjacency_neighbors = np.zeros((len(nodes), len(nodes)))

    positions = [(node.pos_x, node.pos_y) for node in nodes]
    lives = [node.alive for node in nodes]
    role = [node.role for node in nodes]

    for node in nodes:
        if node.role != -1 and node.alive:
            adjacency[node.id][node.role] = 1
        if node.alive:
            for ne in node.neighbors:
                adjacency_neighbors[ne][node.id] = 1
                adjacency_neighbors[node.id][ne] = 1

    if type == "heat map":
        energy = [node.delta_energy_in_timestep for node in nodes]
        min_energy = min(energy)
        len_quantiztion_energy = max(energy) - min_energy
        if len_quantiztion_energy != 0:
            energy = [
                ((node_energy - min_energy) / len_quantiztion_energy)
                for node_energy in energy
            ]
        else:
            energy = [0.1 for _ in energy]
        adjacency_pos_role = [adjacency, positions, role, lives, energy]
        show_graph_heat_map(adjacency_pos_role, "heat_map", title, last, re_cluster)

    elif type == "vertical":
        planes = [node.plane for node in nodes]
        adjacency_pos_role = [adjacency, positions, role, lives, planes]
        show_graph_vertical_routing(
            adjacency_pos_role, "vertical_routing", title, last, re_cluster
        )

    elif type == "cluster":
        adjacency_pos_role = [adjacency, positions, role, lives]
        show_graph_with_labels(adjacency_pos_role, "clusters", title, last, re_cluster)

    elif type == "cluster-radius":
        adjacency_pos_role = [adjacency, positions, role, lives, adjacency_neighbors]
        show_graph_with_radius(nodes, adjacency_pos_role, "clusters", title, last, re_cluster)


def show_graph_with_radius(real_nodes, adjacency_matrix, fig_title, title, last, re_cluster):
    """
    Show clusters with additional neighbor edges.

    adjacency_matrix : [adjacency, positions, role, lives, neighbors]
        adjacency shape (num_nodes, num_nodes) value 0 or 1 {Each CM connects to a CH}
        positions   is  [(x, y)] for every node
        role        is  []       for every node
        live        is  []       for every node
        neighbors   is  (num_nodes, num_nodes) with values 0 or 1 indicating neighbor connections

    fig_title : title of figure
    title : time of round
    last : true if timestep is last
    re_cluster : true if time of re cluster
    """
    plt.figure(fig_title, figsize=(cf.AREA_WIDTH / 100, cf.AREA_LENGTH / 100))
    ax = plt.subplot()
    ax.clear()
    adjacency = adjacency_matrix[0]
    positions = adjacency_matrix[1]
    roles = adjacency_matrix[2]
    lives = adjacency_matrix[3]
    neighbors = adjacency_matrix[4]
    nodes = [node for node in range(len(adjacency[0]))]
    dictionary = {}
    for i in range(len(positions)):
        dictionary[i] = {"position": positions[i], "color": "#A0CBE2"}

    rows, cols = np.where(adjacency == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.set_node_attributes(G, dictionary)
    labels = {n: "Ord" for n in G.nodes()}
    for i in range(len(roles)):
        if roles[i] == -1:
            G.nodes[i]["color"] = "#F7A1AE"
            labels[i] = "CH"
        elif roles[i] == i:
            G.nodes[i]["color"] = "#A0DD9A"
            labels[i] = "DM"

        if not lives[i]:
            G.nodes[i]["color"] = "#D9D9D9"
    pos = nx.get_node_attributes(G, "position")
    # Draw neighbor edges in a different color
    neighbor_rows, neighbor_cols = np.where(neighbors == 1)
    neighbor_edges = list(zip(neighbor_rows.tolist(), neighbor_cols.tolist()))
    nx.draw_networkx_edges(G, pos, edgelist=neighbor_edges, ax=ax, edge_color='green')
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=500,
        node_color=[G.nodes[n]["color"] for n in G.nodes],
        width=2.0,
        labels=labels,
        edge_color="#FF0000",
        style="dotted"

    )

    # Draw communication radius for each node
    for i in range(len(positions)):
        circle = plt.Circle(positions[i], cf.COVERAGE_RADIUS, fill=False, linestyle=':', color='blue', alpha=.25)
        ax.add_patch(circle)
        energy_percentage = real_nodes[i].energy_source.energy / real_nodes[i].energy_source.max_energy
        color = plt.cm.RdYlGn(energy_percentage)
        plt.scatter(positions[i][0], positions[i][1], color=color, s=100)
        # Show buffer fill level
        buffer_fill = len(real_nodes[i].buffer.buffer) / cf.BUFFER_SIZE
        plt.text(positions[i][0], positions[i][1] - 10, f"UAV: {i}", fontsize=6, ha='center', va='top')
        plt.text(positions[i][0], positions[i][1] - 25, f"Buf: {buffer_fill:.2f}", fontsize=6, ha='center', va='top')

        # Show energy level
        plt.text(positions[i][0], positions[i][1] - 40, f"E: {energy_percentage:.2f}", fontsize=6, ha='center',
                 va='top')

    plt.xlim([0, cf.AREA_WIDTH])
    plt.ylim([0, cf.AREA_LENGTH])
    plt.xlabel("X")
    plt.ylabel("Y")
    # Add horizontal and vertical lines to create subgrid visual effect
    x_step = int(cf.AREA_WIDTH / cf.SUB_GRID_SIZE)
    y_step = int(cf.AREA_LENGTH / cf.SUB_GRID_SIZE)

    # Draw vertical lines for the subgrid
    for i in range(1, x_step):
        ax.axvline(x=i * cf.SUB_GRID_SIZE, linestyle=':', linewidth=1)

    # Draw horizontal lines for the subgrid
    for i in range(1, y_step):
        ax.axhline(y=i * cf.SUB_GRID_SIZE, linestyle=':', linewidth=1)
    if re_cluster:
        plt.title("time " + str(calculate_per_time_unit(title)) + " Re cluster time!")
    else:
        plt.title("time " + str(calculate_per_time_unit(title)))
    limits = plt.axis("on")  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.pause(0.001)

    if last:
        plt.savefig("results/graph_" + fig_title + ".png")
        plt.savefig("results/graph_" + fig_title + ".svg")
