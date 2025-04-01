# /simulation/cluster.py
from core.node import Node
from core.supernode import SuperNode
from config import config

def initialize_clusters(model, target_class):
    clusters = []
    for i in range(config.clusters):
        nodes = [Node(node_id=j, model=model, target_class=target_class) for j in range(config.nodes_per_cluster)]
        supernode = SuperNode(supernode_id=i, neighbors=[])
        clusters.append((supernode, nodes))
    return clusters
