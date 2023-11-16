from classes import *
import matplotlib.pyplot as plt
import networkx as nx
import random

cost = 5
seed = 196796533

N = Network()

N.add_node(1)
N.add_node(2)
N.add_node(3)
N.add_node(4)

N.connect_nodes(1,2,random.randint(1,10,seed=seed))
N.connect_nodes(1,3,random.randint(1,10,seed=seed))
N.connect_nodes(2,3,random.randint(1,10,seed=seed))
N.connect_nodes(3,4,random.randint(1,10,seed=seed))

N.show()
