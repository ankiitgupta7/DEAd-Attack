# /core/supernode.py

class SuperNode:
    def __init__(self, supernode_id, nodes):
        self.supernode_id = supernode_id
        self.nodes = nodes  # List[Node]
        self.peers = []     # List[SuperNode] – to be set later

    def set_peers(self, all_supernodes):
        self.peers = [sn for sn in all_supernodes if sn != self]

    def get_best_node(self):
        return max(self.nodes, key=lambda n: n.best_fitness)

    def sync_with_peers(self):
        """Exchange best solutions with peer supernodes and possibly update own nodes."""
        my_best = self.get_best_node()

        for peer in self.peers:
            peer_best = peer.get_best_node()

            if peer_best.best_fitness > my_best.best_fitness:
                # Push peer_best to all my nodes if better
                for node in self.nodes:
                    if peer_best.best_fitness > node.best_fitness:
                        node.population = peer_best.best_solution
                        node.best_fitness = peer_best.best_fitness
                        node.best_solution = peer_best.best_solution
                        print(f"🌐 SuperNode {self.supernode_id} pulled better solution from SuperNode {peer.supernode_id} for Node {node.global_id}")

    def broadcast_best_solution(self):
        """(Optional) Broadcast best solution to all local nodes."""
        best_node = self.get_best_node()
        for node in self.nodes:
            if node != best_node and best_node.best_fitness > node.best_fitness:
                node.population = best_node.best_solution
                node.best_fitness = best_node.best_fitness
                node.best_solution = best_node.best_solution
                print(f"📢 SuperNode {self.supernode_id}: Broadcast best solution to {node.global_id}")
