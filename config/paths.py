# /config/paths.py

import os
from config import config

def get_experiment_root():
    """
    E.g. 'results/fashion_mnist_SVM/rep3'
    """
    base = f"{config.dataset_name}_{config.model_name}"
    return os.path.join("results", base, f"rep{config.replicate}")
