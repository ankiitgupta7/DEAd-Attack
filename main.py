import argparse
from config import config
from models.load_model import load_trained_model
from simulation.run_simulation import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Distributed Evolution Simulation")

    # Override experiment structure
    parser.add_argument("--clusters", type=int, default=config.clusters, help="Number of clusters")
    parser.add_argument("--nodes_per_cluster", type=int, default=config.nodes_per_cluster, help="Number of nodes per cluster")

    # Dataset and model
    parser.add_argument("--dataset_name", type=str, default=config.dataset_name, help="Dataset to use (digits, mnist, fashion_mnist)")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Model to use (SVM, RF, MLP, etc.)")

    # replicate index
    parser.add_argument("--replicate", type=int, default=0, help="replicate ID (used to track logs/output)")

    # Optional target class override (if needed)
    parser.add_argument("--target_class", type=int, default=0, help="Class to evolve toward")

    args = parser.parse_args()

    # Apply overrides to shared config
    config.clusters = args.clusters
    config.nodes_per_cluster = args.nodes_per_cluster
    config.dataset_name = args.dataset_name
    config.model_name = args.model_name
    config.replicate = args.replicate  # if used elsewhere

    print(f"🚀 Starting replicate {args.replicate} | Setup: {args.clusters}x{args.nodes_per_cluster} | Dataset: {args.dataset_name} | Model: {args.model_name}")

    # Load model and run simulation
    model = load_trained_model()
    run_simulation(model, target_class=args.target_class, replicate_id=args.replicate)
