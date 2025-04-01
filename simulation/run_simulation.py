# /simulation/run_simulation.py
from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
from config import config

def run_simulation(model, target_class):
    print("🚀 Initializing clusters and nodes...")
    
    clusters = initialize_clusters(model, target_class)
    candc = CommandAndControl()
    
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)
    
    print("🔄 Starting evolution process...")
    
    # Run the evolution until a solution is found
    for gen in range(config.max_generations):
        for _, nodes in clusters:
            for node in nodes:
                node.evolve()
        
        if gen % config.supernode_sync_interval == 0:
            for supernode, _ in clusters:
                supernode.sync_with_peers()

        if candc.check_termination():
            print("✅ Termination condition met. System stopping.")
            break

    print("✅ Evolution completed successfully.")
