# /simulation/run_simulation.py
import matplotlib.pyplot as plt
from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
from config import config
from utils.evaluation import evaluate_fitness
from config.paths import get_experiment_root
import os

def plot_combined_progress(clusters):
    plt.figure(figsize=(10, 6))
    for supernode, nodes in clusters:
        for node in nodes[:3]:  # first 3 nodes for clarity
            plt.plot(node.confidence_progress, label=node.global_id)
    
    plt.xlabel("Generation")
    plt.ylabel("Confidence")
    plt.title("Confidence Progress of Selected Nodes")
    plt.legend()

    experiment_dir = get_experiment_root()
    os.makedirs(experiment_dir, exist_ok=True)
    outfile = os.path.join(experiment_dir, "combined_confidence_progress.png")
    plt.savefig(outfile)
    plt.close()

def print_summary(clusters):
    from utils.evaluation import evaluate_fitness

    summary_lines = []
    summary_lines.append("\n📊 Summary of Final Best Solutions:\n")
    for supernode, nodes in clusters:
        for node in nodes:
            if node.best_solution is not None:
                confidence = evaluate_fitness(node.best_solution, node.model, node.target_class)
                summary_lines.append(f"✔️ {node.global_id} - Final Confidence: {confidence:.4f}\n")
            else:
                summary_lines.append(f"❌ {node.global_id} - No valid solution found.\n")

    experiment_dir = get_experiment_root()
    summary_file = os.path.join(experiment_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.writelines(summary_lines)

    # Also print to console
    print("".join(summary_lines))

def run_simulation(model, target_class):
    print("🔄 Starting evolution process...")

    clusters = initialize_clusters(model, target_class)
    candc = CommandAndControl()

    # Link supernodes as peers
    supernodes = [supernode for supernode, _ in clusters]
    for supernode in supernodes:
        supernode.set_peers(supernodes)

    # Register all nodes
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)

    round_num = 0
    while not candc.terminated:
        print(f"\n🌀 Round {round_num}")
        for _, nodes in clusters:
            for node in nodes:
                node.evolve(round_num=round_num)
                candc.check_termination()
                if candc.terminated:
                    print("🎯 A node has reached the threshold. Stopping now.")
                    break
            if candc.terminated:
                break

        # Supernode communication
        if round_num % config.supernode_sync_interval == 0:
            print(f"\n🌐 Supernode syncing at round {round_num}")
            for supernode, _ in clusters:
                supernode.sync_with_peers()

        round_num += 1

    print("\n✅ Simulation ended (terminated =", candc.terminated, ")")
    plot_combined_progress(clusters)
    print_summary(clusters)
