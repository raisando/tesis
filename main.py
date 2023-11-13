import networkx as nx
import matplotlib.pyplot as plt

from classes import *
capacity = 10
cost = 5

N = Network()

N.add_node(1)
N.add_node(2)
N.add_node(3)
N.add_node(4)
N.connect_nodes(1,2,capacity,cost)
N.connect_nodes(1,3,capacity,cost)
N.connect_nodes(2,3,capacity,cost)
N.connect_nodes(3,4,capacity,cost)

N.show()
