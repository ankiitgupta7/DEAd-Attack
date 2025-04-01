# /core/candc.py
from utils.evaluation import evaluate_fitness
from config import config


class CommandAndControl:
    def __init__(self):
        self.nodes = []
        self.terminated = False
    
    def assign_node(self, node):
        self.nodes.append(node)
    
    def check_termination(self):
        for node in self.nodes:
            if node.best_solution is not None and evaluate_fitness(node.best_solution, node.model, node.target_class) >= config.target_confidence:
                self.terminated = True
                print("Termination condition met. System stopping.")
                return self.terminated
