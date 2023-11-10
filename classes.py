import numpy as np

class Node:
    def __init__(self,node_id):
        self.id = node_id
        self.connections = []

    def connect(self,node):
        self.connections.append(node.node_id)

class Edge:
    def __init__(self, from_node, to_node, capacity, cost):
        self.from_node = from_node
        self.to_node = to_node
        self.capacity = capacity
        self.cost = cost
        self.interdicted = False

class Network:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.cost_matrix = np.array([[]])
        self.capacity_matrix = np.array([[]])


    def add_node(self, node_id):
        #adds node to dict
        self.nodes[node_id] = Node(node_id)

        #gets size
        size = len(self.nodes)
        new_matrix = np.zeros((size, size))

        #cost matrix
        new_matrix[:size-1, :size-1] = self.cost_matrix
        self.cost_matrix = new_matrix

        #capacity matrix
        new_matrix[:size-1, :size-1] = self.capacity_matrix
        self.capacity_matrix = new_matrix


    def connect_nodes(self,from_node_id,to_node_id,capacity,cost):
        self.edges.append(
            Edge(from_node_id,
                 to_node_id,
                 capacity,
                 cost)
            )

        self.capacity_matrix[from_node_id-1, to_node_id-1] = capacity
        self.capacity_matrix[to_node_id-1, from_node_id-1] = capacity

        self.cost_matrix[from_node_id-1, to_node_id-1] = cost
        self.cost_matrix[to_node_id-1, from_node_id-1] = cost

    def interdict_edge(self,edge):
        edge.interdicted = True

    def calculate_evader_cost(self):
        #
        pass
    
    def show(self):
        print(self.nodes.keys())
        print(self.edges)
        print(self.capacity_matrix)

class Interdictor:
    def __init__(self,budget):
        self.budget = budget

    def interdict(self,network):
        #for edge in lista
        #network.interdict_edge(edge)
        pass
