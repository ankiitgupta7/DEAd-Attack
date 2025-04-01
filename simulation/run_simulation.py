# /simulation/run_simulation.py

import matplotlib.pyplot as plt
from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
from config import config
from utils.evaluation import evaluate_fitness

def plot_combined_progress(clusters):
    """Plot combined confidence progress for multiple nodes."""
    plt.figure(figsize=(10, 6))
    
    for supernode, nodes in clusters:
        for node in nodes[:3]:  # Plot progress for the first 3 nodes per cluster
            plt.plot(node.confidence_progress, label=node.global_id)
    
    plt.xlabel("Generation")
    plt.ylabel("Confidence")
    plt.title("Confidence Progression of Selected Nodes")
    plt.legend()
    plt.savefig("output_images/combined_confidence_progress.png")
    plt.close()

def print_summary(clusters):
    """Print summary of best solutions for all nodes."""
    print("\n📊 Summary of Final Best Solutions:")
    for supernode, nodes in clusters:
        for node in nodes:
            if node.best_solution is not None:
                confidence = evaluate_fitness(node.best_solution, node.model, node.target_class)
                print(f"✔️ {node.global_id} - Final Confidence: {confidence:.4f}")
            else:
                print(f"❌ {node.global_id} - No valid solution found.")

def run_simulation(model, target_class):
    print("🔄 Starting evolution process...")

    # Initialize nodes and clusters
    clusters = initialize_clusters(model, target_class)
    candc = CommandAndControl()

    # Link supernodes as peers
    supernodes = [supernode for supernode, _ in clusters]
    for supernode in supernodes:
        supernode.set_peers(supernodes)

    # Register all nodes with C&C
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)

    round = 0

    # Main simulation loop
    while not candc.terminated:
        print(f"\n🌀 Round {round}")
        for _, nodes in clusters:
            for node in nodes:
                node.evolve()
                candc.check_termination()
                if candc.terminated:
                    print("🎯 A node has reached the threshold. Stopping now.")
                    break
            if candc.terminated:
                break

        # Supernode communication
        if round % config.supernode_sync_interval == 0:
            print(f"\n🌐 Supernode syncing at round {round}")
            for supernode, _ in clusters:
                supernode.sync_with_peers()

        round += 1

    print("\n✅ Simulation ended (terminated =", candc.terminated, ")")
    plot_combined_progress(clusters)
    print_summary(clusters)
