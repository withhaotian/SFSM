import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text import ProgressBar
from configs.config import Config
from env.server import Edge_Server
from utils.tools import print_edge_networks

bar = ProgressBar()

random.seed(Config.seed)
np.random.seed(Config.seed)

def generate_udg_edge_network(n_nodes, n_max_connections, all_connect=False):
    """
    Generate the edge computing scenario, i.e., generate a undirected adn connected graph of edge servers,
    """
    
    # The maximum number of connections per node cannot exceed the total number of nodes
    assert 0 < n_max_connections < n_nodes

    # step 1: generate a connected graph
    # initialize
    G = np.zeros((n_nodes, n_nodes))

    if not all_connect:
        # According to the maximum number of connections, initialize graph
        connection_set = [set() for _ in range(n_nodes)]
        for i in range(n_nodes):
            n_nodes_already_connected = len(connection_set[i])
            n_nodes_all_connections = np.random.randint(1, n_max_connections + 1)
            while n_nodes_already_connected < n_nodes_all_connections:
                new_node_number = random.randint(0, n_nodes - 1)

                # The new connection cannot be connected to itself
                if new_node_number == i:
                    continue

                # If another connection has reached its maximum value, 
                # the generated connection is invalid and needs to be regenerated
                if len(connection_set[new_node_number]) >= n_max_connections:
                    continue

                # Valid connection
                connection_set[i].add(new_node_number)
                connection_set[new_node_number].add(i)
                n_nodes_already_connected = len(connection_set[i])

        for i in range(n_nodes):
                for j in connection_set[i]:
                    G[i, j], G[j, i] = 1, 1
    else:
    # All nodes connect to others
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    G[i, j], G[j, i] = 1, 1

    # step 2: set the bandwidth
    W = np.ones((n_nodes, n_nodes)) * -1
    for i in range(n_nodes):
        j = 0
        while j < i:
            if G[i, j] == 1:
                b = random.randint(Config.band_lower, Config.band_upper)
                W[i, j], W[j, i] = b, b
            j = j + 1

    # step 3: set the processing power
    ESs = []
    for i in range(n_nodes):
        ESs.append(Edge_Server(i, Config.cap_conf, random.randint(Config.proc_lower, Config.proc_upper), 0))

    return G, W, ESs

if __name__ == '__main__':
    G, W, ESs = generate_udg_edge_network(10, 7, True)
    print_edge_networks(G, W)