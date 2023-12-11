import numpy as np
import heapq
import networkx as nx
import random


# The Node class represents a node in a graph and allows for connecting nodes
# together.
class Node:
    def __init__(self,node_id):
        """
        Constructor for a class that initializes a node with an id and an empty list of connections.

        :param node_id: Represents the unique identifier for the node
        """
        self.id          = node_id
        self.connections = []

    def connect(self,node):
        self.connections.append(node.node_id)

# The Edge class represents an edge between two nodes in a graph, with a cost
# associated with it.
class Edge:
    def __init__(self, from_node, to_node, cost):
        """
        Constructor for a class that represents a connection between two nodes, with a specified cost.

        :param from_node: The node from which the edge originates
        :param to_node: The destination node in a graph or network. It is the node that the edge connects to
        :param cost: The cost or weight associated with the edge between two nodes.
        """
        self.from_node   = from_node
        self.to_node     = to_node
        self.cost        = cost
        self.interdicted = False

# The `Network` class represents a network of nodes and edges, with methods for
# adding nodes, connecting nodes with edges, interdicting edges, calculating the
# cost for an evader to travel from one node to another, and displaying the
# network.
class Network:
    def __init__(self,nodes_per_layer,layers):
        """
        Constructor for a class that initializes attributesfor a graph data structure.
        """
        self.nodes       = {}
        self.edges       = {}
        self.cost_matrix = np.array([[]])
        self.pos = None
        self.nodes_per_layer = nodes_per_layer
        self.layers = layers
        self.total_nodes = layers * (nodes_per_layer-1) + 1

    def add_node(self, node_id):
        """
        This function adds a new node to a dictionary and updates the cost
        matrix.

        :param node_id: The `node_id` parameter is the identifier for the nodethat you want to add.
        """
        #adds node to dict
        self.nodes[node_id] = Node(node_id)
        #gets size
        size = len(self.nodes)
        new_matrix = np.zeros((size, size))
        #cost matrix
        new_matrix[:size-1, :size-1] = self.cost_matrix
        self.cost_matrix = new_matrix

    def connect_nodes(self,from_node_id,to_node_id,cost):
        """
        This function adds an edge between two nodes and updates the cost matrix with the given cost.

        :param from_node_id: The ID of the node from which the edge starts
        :param to_node_id: The ID of the node thatthe edge is connecting to
        :param cost: The cost parameter represents the cost or weight associated with the edge connecting the from_node_id and to_node_id.
        """
        self.edges[from_node_id,to_node_id] = Edge(from_node_id,to_node_id,cost)
        self.cost_matrix[from_node_id-1, to_node_id-1] = cost
        #self.cost_matrix[to_node_id-1, from_node_id-1] = cost

    def disconnect_nodes(self,from_node_id,to_node_id):
        del self.edges[from_node_id,to_node_id]
        self.cost_matrix[from_node_id-1, to_node_id-1] = 0


    def has_edge(self,from_node_id,to_node_id):
        """
        This method checks if there is an edge between two nodes in a graph.

        :param from_node_id: The ID of the node from which the edge starts
        :param to_node_id: The `to_node_id` parameter represents the ID of the node that we want to check if there is an edge from the `from_node_id` to it

        :return: a boolean value indicating whether there is an edge between the twogiven nodes.
        """
        return self.cost_matrix[from_node_id-1, to_node_id-1] > 0

    def interdict_edge(self,edge):
        r = edge[0]
        c = edge[1]
        if self.has_edge(r,c):
            self.cost_matrix[r-1,c-1] = float('inf')
            self.edges[r,c].interdicted = True
            self.edges[r,c].cost = float('inf')

    def interdict(self,to_interdict=[]):
        if len(to_interdict) == 0:
            K = len(self.nodes)//2
            for i in range(K):
                edge = random.choice(list(self.edges.keys()))
                self.interdict_edge(edge)

        else:
            for edge in to_interdict:
                self.interdict_edge(edge)

    def calculate_evader_cost(self, start_node, end_node):
        """
        This function calculates the shortest path and cost from a start node to an end node in a graph using Dijkstra's algorithm.

        :param start_node: Represents the node from which the calculation of the evader cost will begin. It is the starting point of the path for which the cost is being calculated

        :param end_node: Represents the node where the evader wants to reach. It is the destination node for which the cost of reaching from the `start_node` needs to be calculated
        :return: a tuple containing the path and the distance from the start node to the end node. The path is a list of nodes that represents the shortest path from the start node to the end node. The distance is the total cost of the path.
        """
        cost_matrix              = self.cost_matrix
        np.fill_diagonal(cost_matrix, 0)  # No cost for staying at the same node

        num_nodes                = len(self.nodes)
        visited                  = [False] * num_nodes
        distance                 = [float('inf')] * num_nodes
        predecessor              = [-1] * num_nodes
        distance[start_node - 1] = 0
        pq                       = [(0, start_node - 1)] # distance,node

        while pq: #not empty
            current_distance, current_node = heapq.heappop(pq)
            visited[current_node]          = True

            if current_node == end_node - 1: #finished
                break
            for neighbor in range(num_nodes): #else:
                if not visited[neighbor] and cost_matrix[current_node, neighbor] > 0: #each not visited neighbor
                    new_distance = current_distance + cost_matrix[current_node, neighbor]

                    if new_distance < distance[neighbor]: #new path is shorter
                        distance[neighbor]    = new_distance #update
                        predecessor[neighbor] = current_node #for backtracking
                        heapq.heappush(pq, (new_distance, neighbor))
                        continue

        # Reconstruct the path
        path    = []
        current = end_node - 1
        while current != -1:
            path.append(current + 1)
            current = predecessor[current]
        path.reverse()
        return path, distance[end_node - 1]



    def show(self):
        """
        This method creates a directed graph using the nodes and edges provided,
        and then visualizes the graph using networkx library in Python.
        """
        G = nx.DiGraph()  # Use DiGraph for directed graph
        for node in self.nodes:
            G.add_node(node)

        for edge in self.edges.values():
            G.add_edge(edge.from_node, edge.to_node, cost=edge.cost)

        # Draw the network
        if self.pos is None:
            self.pos = nx.spring_layout(G)

        pos = self.pos
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                arrows=True, arrowstyle='-|>', arrowsize=10)
        edge_labels = {(u, v): d['cost'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    def show2(self):
        """
        Visualize the graph with a custom layout for multiple layers.
        """
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)

        for edge in self.edges.values():
            G.add_edge(edge.from_node, edge.to_node, cost=edge.cost)

        # Custom layout for multiple layers
        pos = self.custom_layout()

        # Draw the network
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                arrows=True, arrowstyle='-|>', arrowsize=10)
        edge_labels = {(u, v): d['cost'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)



    def custom_layout(self):
        """
        Create a custom layout for the nodes for multiple layers, with shared nodes between layers.
        """
        pos = {}
        layer_width = 1.0 / self.layers
        node_height = 1.0 / (self.nodes_per_layer - 2)

        for node in self.nodes:
            # Calculate which layer the node belongs to
            layer = (node - 1) // (self.nodes_per_layer - 1)
            x_offset = layer * layer_width

            if node == 1 or node == len(self.nodes):
                x_pos = x_offset #if node == 1 else x_offset + layer_width
                pos[node] = (x_pos, 0.5)
            else:
                # Calculate position within the layer
                within_layer_node_index = (node - 1) % (self.nodes_per_layer - 1)

                # Adjust y-position for shared nodes
                if within_layer_node_index == 0:
                    y_pos = 0.5
                    x_pos = x_offset
                else:
                    y_pos = within_layer_node_index * node_height
                    x_pos = x_offset + layer_width / 2

                pos[node] = (x_pos, y_pos)

        return pos


    def test(self):
        for node in self.nodes:
            print(node)
