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

def visualize_topology(clusters):
    """
    clusters: A list of tuples (supernode, [node1, node2, ...]),
              where each node has:
                - global_id (e.g. C2-N3)
                - buffer (neighbors)
    supernode: has supernode_id
    """

    G = nx.Graph()

    # 1) Place supernodes in a ring with radius 10
    n_supernodes = len(clusters)
    supernode_positions = {}
    R = 10  # ring radius for supernodes

    for i, (supernode, _) in enumerate(clusters):
        # supernode label like S0, S1, ...
        s_label = f"S{supernode.supernode_id}"
        angle = 2 * math.pi * i / n_supernodes
        x = R * math.cos(angle)
        y = R * math.sin(angle)
        G.add_node(s_label)
        supernode_positions[s_label] = (x, y)

    # 2) Connect supernodes in a ring with magenta edges
    for i in range(n_supernodes):
        s1_label = f"S{clusters[i][0].supernode_id}"
        s2_label = f"S{clusters[(i+1) % n_supernodes][0].supernode_id}"
        G.add_edge(s1_label, s2_label, color="magenta")

    # 3) Place cluster nodes in smaller rings around each supernode, no direct edge to supernode
    positions = {}
    node_colors = {}
    cluster_radius = 3  # ring radius around supernode

    for i, (supernode, nodes) in enumerate(clusters):
        s_label = f"S{supernode.supernode_id}"
        # center of this cluster's ring is supernode's position
        sx, sy = supernode_positions[s_label]
        n_nodes = len(nodes)

        for j, node in enumerate(nodes):
            # angle-based offset from the supernode center
            angle = 2 * math.pi * j / n_nodes
            rx = sx + cluster_radius * math.cos(angle)
            ry = sy + cluster_radius * math.sin(angle)

            G.add_node(node.global_id)
            positions[node.global_id] = (rx, ry)
            node_colors[node.global_id] = "skyblue"

            # 4) Add edges for neighbor connections
            for neighbor in node.buffer:
                if not G.has_node(neighbor.global_id):
                    G.add_node(neighbor.global_id)
                # Avoid duplicating edges
                if not G.has_edge(node.global_id, neighbor.global_id):
                    G.add_edge(node.global_id, neighbor.global_id, color="black")

    # Combine positions for supernodes
    for s_label, pos in supernode_positions.items():
        positions[s_label] = pos
        node_colors[s_label] = "red"

    # 5) Prepare to draw edges with color
    edge_colors = []
    for (u, v, attrs) in G.edges(data=True):
        edge_colors.append(attrs.get("color", "gray"))

    # 6) Build node color array
    node_color_map = []
    for n in G.nodes():
        node_color_map.append(node_colors.get(n, "skyblue"))  # default skyblue

    # 7) Draw
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos=positions,
        with_labels=True,
        node_color=node_color_map,
        edge_color=edge_colors,
        node_size=1000,
        font_size=8
    )
    plt.title("Distributed System Topology: Clusters / Supernodes / Neighbors")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

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

    # Visualize the initial topology
    visualize_topology(clusters)

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
