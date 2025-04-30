# /simulation/run_simulation.py
import matplotlib.pyplot as plt
from simulation.cluster import initialize_clusters
from core.candc import CommandAndControl
from config import config
from utils.evaluation import evaluate_fitness
from config.paths import get_experiment_root
import os
import math
import networkx as nx
from tqdm import tqdm
from itertools import count


import math
import os
import networkx as nx
import matplotlib.pyplot as plt

def visualize_topology(clusters, out_path="topology.png"):
    """
    Visualization that doesn't 'hard-code' a node as supernode:
    - For cluster i, the supernode is shown as a separate red node labeled S{i}
    - The cluster's normal nodes are in skyblue
    - Edges for neighbor connections are black
    - Supernodes form a ring in magenta edges
    - No references to best node or node[0]
    """

    G = nx.Graph()
    positions = {}
    node_colors = {}

    n_clusters = len(clusters)

    # --- 1) Place supernodes in a big ring (radius=10) ---
    R = 10
    supernode_labels = []
    for i, (supernode, nodes) in enumerate(clusters):
        s_label = f"S{supernode.supernode_id}"  # e.g. S0, S1, ...
        supernode_labels.append(s_label)

        angle = 2 * math.pi * i / n_clusters
        x = R * math.cos(angle)
        y = R * math.sin(angle)

        # Add to the graph
        G.add_node(s_label)
        positions[s_label] = (x, y)
        # We'll color supernodes red
        node_colors[s_label] = "red"

    # --- 2) Connect supernodes in a ring with magenta edges ---
    for i in range(n_clusters):
        s1 = supernode_labels[i]
        s2 = supernode_labels[(i + 1) % n_clusters]
        G.add_edge(s1, s2, color="magenta")

    # --- 3) Place cluster nodes around each supernode (small ring radius=3) ---
    cluster_radius = 3

    for i, (supernode, nodes) in enumerate(clusters):
        s_label = f"S{supernode.supernode_id}"
        sx, sy = positions[s_label]
        n_nodes = len(nodes)

        for j, node in enumerate(nodes):
            # place each cluster node around the supernode
            angle = 2 * math.pi * j / n_nodes
            rx = sx + cluster_radius * math.cos(angle)
            ry = sy + cluster_radius * math.sin(angle)

            # Insert into graph
            G.add_node(node.global_id)
            positions[node.global_id] = (rx, ry)
            node_colors[node.global_id] = "skyblue"  # normal cluster node

            # Add edges for neighbor connections (black)
            for neighbor in node.buffer:
                if not G.has_edge(node.global_id, neighbor.global_id):
                    G.add_edge(node.global_id, neighbor.global_id, color="black")

        # If you DO want an edge from supernode to each cluster node, do:
        for node in nodes:
            G.add_edge(s_label, node.global_id, color="gray")

    # --- 4) Prepare to draw
    edge_colors = [G[u][v].get("color", "gray") for u, v in G.edges()]
    node_color_list = [node_colors[n] for n in G.nodes()]

    labels = {n: n for n in G.nodes()}

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos=positions,
        labels=labels,
        with_labels=True,
        node_color=node_color_list,
        edge_color=edge_colors,
        node_size=1000,
        font_size=8
    )
    plt.title("Distributed System Topology: True Clusters & Supernodes")
    plt.axis("off")

    # Ensure directory exists if out_path has a subdir
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

def run_simulation(model, target_class, replicate_id):
    print("🔄 Starting evolution process...")

    clusters = initialize_clusters(model, target_class)
    visualize_topology(clusters, out_path="topology.png")
    candc = CommandAndControl()

    # Link supernodes
    supernodes = [supernode for supernode, _ in clusters]
    for supernode in supernodes:
        supernode.set_peers(supernodes)

    # Register nodes
    for _, nodes in clusters:
        for node in nodes:
            candc.assign_node(node)

    # ✅ Outer tqdm for rounds
    for round_num in tqdm(count(), desc="🌱 Rounds", position=0, leave=True):
        if candc.terminated:
            break

        # ✅ Inner tqdm for nodes within round
        all_nodes = [node for _, nodes in clusters for node in nodes]
        with tqdm(all_nodes, desc=f"⚙️  Evolving Nodes (Round {round_num})", position=1, leave=False) as node_bar:
            for node in node_bar:
                node.evolve(round_num=round_num)
                candc.check_termination()
                if candc.terminated:
                    node_bar.set_description("🎯 Termination triggered")
                    break

        # 📊 Track & show confidence live in outer tqdm
        best_conf = 0.0
        for _, nodes in clusters:
            for node in nodes:
                if node.best_solution is not None:
                    conf = evaluate_fitness(node.best_solution, node.model, node.target_class)
                    best_conf = max(best_conf, conf)

        tqdm.write(f"Round {round_num} done. Best confidence so far: {best_conf:.4f}")
        tqdm._instances.clear()  # fixes overlapping bars sometimes

        # 🔁 Supernode sync
        if round_num % config.supernode_sync_interval == 0:
            for supernode, _ in clusters:
                supernode.sync_with_peers()

    # Wrap up
    plot_combined_progress(clusters)
    print_summary(clusters)

    
