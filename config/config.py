# /config/config.py

# System parameters
clusters = 5
nodes_per_cluster = 6
neighbors_per_node = 3

# Evolution parameters
population_size = 50
mutation_rate = 0.5
max_generations = 50
target_confidence = 0.15

# Communication parameters
buffer_size = 10
communication_interval = 5
supernode_sync_interval = 10

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
