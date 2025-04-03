# /config/config.py

# System parameters
clusters = 1
nodes_per_cluster = 30
neighbors_per_node = 3

# Evolution parameters
# population_size = 50
mutation_rate = 0.5
max_generations = 100
target_confidence = 0.25

# Communication parameters
buffer_size = 10
communication_interval = 5
supernode_sync_interval = 1

# Fault tolerance
failure_rate = 0.02
recovery_time = 3

# Dataset & Model Info
dataset_name = "mnist"  # or #"digits", "mnist", "fashion_mnist", etc.
model_name = "SVM"

# After training, these will be set automatically
model_file = None
image_height = None
image_width = None
pixel_max = None
