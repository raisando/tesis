import networkx as nx
import matplotlib.pyplot as plt
import random
from classes import *



def generate_random_graph(n_nodes):
    N = Network()

    # Add nodes
    for i in range(1, n_nodes + 1):
        N.add_node(i)

    # Ensure each node (except the last one) has at least one outgoing edge
    for i in range(1, n_nodes):
        cost = random.randint(1, 10)
        target = random.randint(i + 1, n_nodes)
        N.connect_nodes(i, target, cost)

    # Ensure each node (except the first one) has at least one incoming edge
    for i in range(2, n_nodes + 1):
        cost = random.randint(1, 10)
        source = random.randint(1, i - 1)
        if not N.has_edge(source, i):
            N.connect_nodes(source, i, cost)

    # Add additional random edges
    additional_edges = random.randint(1, n_nodes * (n_nodes - 1) // 2)
    for _ in range(additional_edges):
        u, v = random.sample(range(1, n_nodes + 1), 2)
        if u != v and not N.has_edge(u, v):
            cost = random.randint(1, 10)
            N.connect_nodes(u, v, cost)

    if N.has_edge(1,n_nodes):
        N.disconnect_nodes(1,n_nodes)

    return N

def generate_random_graph2(nodes_per_layer, layers = 1):
    N = Network(nodes_per_layer,layers,f_mean,f_std)

    # Define start and end nodes for the entire network
    start_node = 1
    end_node = layers * (nodes_per_layer - 1) + 1

    # Add nodes
    for i in range(start_node, end_node + 1):
        N.add_node(i)

    N.reset()

    return N
