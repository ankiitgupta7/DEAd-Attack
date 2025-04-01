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
    print("🔄 Starting evolution process...")

    clusters = initialize_clusters(model, target_class)
    candc = CommandAndControl()

    # Register all nodes with C&C
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)

    # MAIN LOOP
    while not candc.terminated:
        for _, nodes in clusters:
            for node in nodes:
                node.evolve()

                # 👉 Immediately check if some node reached threshold
                candc.check_termination()
                if candc.terminated:
                    print("🔎 A node has reached the threshold. Stopping now.")
                    break  # breaks out of node loop

            if candc.terminated:
                break  # breaks out of cluster loop
        # Potentially re-check here if you want
        # candc.check_termination()

    print("✅ Simulation ended (terminated =", candc.terminated, ")")
