# /simulation/run_simulation.py
from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
import config

def run_simulation(model, target_class):
    clusters = initialize_clusters(model, target_class)
    candc = CommandAndControl()
    
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)
    
    # Run evolution until termination condition is met
    while not candc.terminated:
        for _, nodes in clusters:
            for node in nodes:
                node.evolve()
        candc.check_termination()
