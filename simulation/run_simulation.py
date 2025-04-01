# /simulation/run_simulation.py
import matplotlib.pyplot as plt
from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
from config import config

def plot_combined_progress(clusters):
    """Plot combined confidence progress for multiple nodes."""
    plt.figure(figsize=(10, 6))
    
    for supernode, nodes in clusters:
        for node in nodes[:3]:  # Plot progress for the first 3 nodes per cluster
            plt.plot(node.confidence_progress, label=f"Node {node.node_id}")
    
    plt.xlabel("Generation")
    plt.ylabel("Confidence")
    plt.title("Confidence Progression of Selected Nodes")
    plt.legend()
    plt.savefig("output_images/combined_confidence_progress.png")
    plt.close()

def print_summary(clusters):
    """Print summary of best solutions for all nodes."""
    print("\n Summary of Final Best Solutions:")
    for supernode, nodes in clusters:
        for node in nodes:
            if node.best_solution is not None:
                confidence = evaluate_fitness(node.best_solution, node.model, node.target_class)
                print(f"Node {node.node_id} - Final Confidence: {confidence:.4f}")

def run_simulation(model, target_class):
    print("Initializing clusters and nodes...")
    
    clusters = initialize_clusters(model, target_class)
    candc = CommandAndControl()
    
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)
    
    print("Starting evolution process...")
    
    # Run the evolution until a solution is found
    for gen in range(config.max_generations):
        for _, nodes in clusters:
            for node in nodes:
                node.evolve()
        
        if gen % config.supernode_sync_interval == 0:
            for supernode, _ in clusters:
                supernode.sync_with_peers()

        if candc.check_termination():
            print("Termination condition met. System stopping.")
            break

    print("Evolution completed successfully.")
    
    # Plot combined confidence progression for selected nodes
    plot_combined_progress(clusters)

    # Print summary of final best solutions
    print_summary(clusters)
