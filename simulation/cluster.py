# /simulation/cluster.py
import random
from core.node import Node
from core.supernode import SuperNode
from config import config

def initialize_clusters(model, target_class):
    clusters = []

    for i in range(config.clusters):
        nodes = [Node(node_id=(i * config.nodes_per_cluster + j), model=model, target_class=target_class)
                 for j in range(config.nodes_per_cluster)]

        # Assign neighbors randomly within the cluster
        for node in nodes:
            possible_neighbors = [n for n in nodes if n != node]
            node.buffer = random.sample(possible_neighbors, min(config.neighbors_per_node, len(possible_neighbors)))

        supernode = SuperNode(supernode_id=i, nodes=nodes)
        clusters.append((supernode, nodes))

    return clusters
