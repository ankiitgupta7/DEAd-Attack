# /core/supernode.py
class SuperNode:
    def __init__(self, supernode_id, neighbors):
        self.supernode_id = supernode_id
        self.neighbors = neighbors
        self.best_solution = None
    
    def exchange_solutions(self):
        # Super-node pushes high-performing solutions to its neighbors
        for neighbor in self.neighbors:
            neighbor.update_solution(self.best_solution)
