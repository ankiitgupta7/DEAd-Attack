# /config/paths.py

import os
from config import config

def get_experiment_root():
    """
    E.g. 'results/fashion_mnist_SVM/rep3'
    """
    base = f"{config.dataset_name}_{config.model_name}"
    target = f"target{config.target_class}"
    nodes = f"{config.clusters}x{config.nodes_per_cluster}"
    return os.path.join(f"results{nodes}", base, target, f"rep{config.replicate}")
