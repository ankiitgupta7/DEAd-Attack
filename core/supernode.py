# /core/supernode.py

class SuperNode:
    def __init__(self, supernode_id, nodes):
        self.supernode_id = supernode_id
        self.nodes = nodes  # List of Node instances

    def sync_with_peers(self):
        """Placeholder for syncing with other supernodes (future extension)."""
        pass

    def broadcast_best_solution(self):
        """Optional: Share best solution across nodes in this cluster."""
        best_node = max(self.nodes, key=lambda n: n.best_fitness)
        for node in self.nodes:
            if node != best_node and best_node.best_fitness > node.best_fitness:
                node.population = best_node.best_solution
                node.best_fitness = best_node.best_fitness
                node.best_solution = best_node.best_solution
                print(f"📢 SuperNode {self.supernode_id}: Broadcast better solution to Node {node.node_id}")
