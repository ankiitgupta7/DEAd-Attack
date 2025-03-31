# /core/node.py
import numpy as np
from utils.mutation import mutate
from utils.evaluation import evaluate_fitness
import config

class Node:
    def __init__(self, node_id, model, target_class):
        self.node_id = node_id
        self.model = model
        self.target_class = target_class
        self.population = self.initialize_population()
        self.buffer = []
        self.best_solution = None
    
    def initialize_population(self):
        # Generate a random 8x8 grayscale image with pixel values [0-16]
        return np.random.randint(0, 17, (8, 8))
    
    def evolve(self):
        for gen in range(config.max_generations):
            mutated_image = mutate(self.population, config.mutation_rate)
            confidence = evaluate_fitness(mutated_image, self.model, self.target_class)
            
            if confidence > evaluate_fitness(self.population, self.model, self.target_class):
                self.population = mutated_image
            
            # Check if confidence threshold is met
            if confidence >= config.target_confidence:
                self.best_solution = mutated_image
                print(f"Node {self.node_id} found a solution with confidence {confidence}")
                return mutated_image
