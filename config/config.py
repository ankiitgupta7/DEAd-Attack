# /config/config.py

# System parameters
clusters = 5               # Number of clusters (super-nodes)
nodes_per_cluster = 6     # Nodes per cluster
neighbors_per_node = 3     # Number of neighbors per node

# Evolution parameters
population_size = 50       # Population size per node
mutation_rate = 0.5       # Mutation rate for pixel modifications
max_generations = 200       # Max generations before termination
target_confidence = 0.99 # Confidence threshold to terminate

# Communication parameters
buffer_size = 10           # Buffer size for neighbor solutions
communication_interval = 5 # Interval to push solutions to neighbors
supernode_sync_interval = 10 # Super-node solution sync interval

# Fault tolerance
failure_rate = 0.02        # Probability of node failure
recovery_time = 3          # Time to reassign tasks upon failure
