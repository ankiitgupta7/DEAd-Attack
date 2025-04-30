import matplotlib.pyplot as plt
import csv
import os
import math
import networkx as nx
from tqdm import tqdm
from itertools import count

from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
from config import config
from utils.evaluation import evaluate_fitness
from config.paths import get_experiment_root


def visualize_topology(clusters, out_path="topology.png"):
    G = nx.Graph()
    positions = {}
    node_colors = {}
    n_clusters = len(clusters)
    R = 10  # outer ring radius

    # Supernodes
    supernode_labels = []
    for i, (supernode, nodes) in enumerate(clusters):
        s_label = f"S{supernode.supernode_id}"
        supernode_labels.append(s_label)

        angle = 2 * math.pi * i / n_clusters
        x = R * math.cos(angle)
        y = R * math.sin(angle)

        G.add_node(s_label)
        positions[s_label] = (x, y)
        node_colors[s_label] = "red"

    # Supernode connections
    for i in range(n_clusters):
        s1 = supernode_labels[i]
        s2 = supernode_labels[(i + 1) % n_clusters]
        G.add_edge(s1, s2, color="magenta")

    cluster_radius = 3
    for i, (supernode, nodes) in enumerate(clusters):
        s_label = f"S{supernode.supernode_id}"
        sx, sy = positions[s_label]
        n_nodes = len(nodes)

        for j, node in enumerate(nodes):
            angle = 2 * math.pi * j / n_nodes
            rx = sx + cluster_radius * math.cos(angle)
            ry = sy + cluster_radius * math.sin(angle)

            G.add_node(node.global_id)
            positions[node.global_id] = (rx, ry)
            node_colors[node.global_id] = "skyblue"

            for neighbor in node.buffer:
                if not G.has_edge(node.global_id, neighbor.global_id):
                    G.add_edge(node.global_id, neighbor.global_id, color="black")

            G.add_edge(s_label, node.global_id, color="gray")

    edge_colors = [G[u][v].get("color", "gray") for u, v in G.edges()]
    node_color_list = [node_colors[n] for n in G.nodes()]
    labels = {n: n for n in G.nodes()}

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, labels=labels, with_labels=True,
            node_color=node_color_list, edge_color=edge_colors,
            node_size=1000, font_size=8)
    plt.title("Distributed System Topology: True Clusters & Supernodes")
    plt.axis("off")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Topology saved to {out_path}")


def plot_combined_progress(clusters):
    plt.figure(figsize=(10, 6))
    for supernode, nodes in clusters:
        for node in nodes[:3]:  # First 3 for simplicity
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
    summary_lines = ["\n📊 Summary of Final Best Solutions:\n"]
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

    print("".join(summary_lines))


def run_simulation(model, target_class):
    clusters = initialize_clusters(model, target_class)
    visualize_topology(clusters, out_path=os.path.join(get_experiment_root(), "topology.png"))
    candc = CommandAndControl()

    supernodes = [s for s, _ in clusters]
    for sn in supernodes:
        sn.set_peers(supernodes)

    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)

    # Create CSV to track all node confidences
    experiment_dir = get_experiment_root()
    os.makedirs(experiment_dir, exist_ok=True)
    csv_path = os.path.join(experiment_dir, f"confidence_log.csv")

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["node", "round", "cumulative_generation", "confidence"])

    cumulative_gens = {node.global_id: 0 for _, nodes in clusters for node in nodes}

    for round_num in tqdm(count(), desc="🌱 Rounds", position=0):
        if candc.terminated:
            break

        all_nodes = [node for _, nodes in clusters for node in nodes]
        with tqdm(all_nodes, desc=f"⚙️ Round {round_num}", position=1, leave=False) as bar:
            for node in bar:
                node.evolve(round_num)
                candc.check_termination()
                if candc.terminated:
                    break

        # Log CSV data for this round
        for _, nodes in clusters:
            for node in nodes:
                for i, conf in enumerate(node.confidence_progress):
                    writer.writerow([
                        node.global_id,
                        round_num,
                        cumulative_gens[node.global_id] + i,
                        round(conf, 6)
                    ])
                    # print(f"Node {node.global_id} | Round {round_num} | Gen {cumulative_gens[node.global_id] + i} | Confidence: {round(conf, 6)}")
                    # print(f"✔️ Logged {len(node.confidence_progress)} entries for {node.global_id} (Round {round_num})")

                    csv_file.flush()
                cumulative_gens[node.global_id] += len(node.confidence_progress)
                node.confidence_progress = []

        # Show best confidence live
        best_conf = max(
            evaluate_fitness(node.best_solution, node.model, node.target_class)
            for _, nodes in clusters for node in nodes if node.best_solution is not None
        )
        tqdm.write(f"Round {round_num} complete | Best Confidence: {best_conf:.4f}")

        if round_num % config.supernode_sync_interval == 0:
            for supernode, _ in clusters:
                supernode.sync_with_peers()

    csv_file.close()
    plot_combined_progress(clusters)
    print_summary(clusters)
