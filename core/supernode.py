# /core/supernode.py
import numpy as np
from config import config
from utils.evaluation import evaluate_fitness

class SuperNode:
    def __init__(self, supernode_id, neighbors):
        self.supernode_id = supernode_id
        self.neighbors = neighbors
        self.best_solution = None

    def sync_with_peers(self):
        """Exchange solutions with neighboring supernodes."""
        for peer in self.neighbors:
            print(f"🔄 SuperNode {self.supernode_id} is synchronizing solutions with peer SuperNode {peer.supernode_id}")
            if peer.best_solution is not None:
                peer_confidence = evaluate_fitness(peer.best_solution, None, None)
                if self.best_solution is None or peer_confidence > evaluate_fitness(self.best_solution, None, None):
                    self.best_solution = peer.best_solution
                    print(f"📡 SuperNode {self.supernode_id} accepted a better solution from SuperNode {peer.supernode_id} with confidence {peer_confidence:.4f}")
