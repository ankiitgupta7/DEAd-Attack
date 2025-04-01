# /core/node.py
import numpy as np
from utils.mutation import mutate
from utils.evaluation import evaluate_fitness
from config import config

class Node:
    def __init__(self, node_id, model, target_class):
        self.node_id = node_id
        self.model = model
        self.target_class = target_class
        self.population = self.initialize_population()
        self.buffer = []
        self.best_solution = None
        print(f"✅ Config Loaded: max_generations = {config.max_generations}, mutation_rate = {config.mutation_rate}")

    
    def initialize_population(self):
        # Generate a random 8x8 grayscale image with pixel values [0-16]
        return np.random.randint(0, 17, (8, 8))
    
    def evolve(self):
        for gen in range(config.max_generations):
            print(f"Node {self.node_id} - Generation {gen}: Evolving...")
            
            mutated_image = mutate(self.population, config.mutation_rate)
            current_confidence = evaluate_fitness(self.population, self.model, self.target_class)
            mutated_confidence = evaluate_fitness(mutated_image, self.model, self.target_class)
            
            # Print confidence for debugging
            print(f"  Current Confidence: {current_confidence:.4f}")
            print(f"  Mutated Confidence: {mutated_confidence:.4f}")
            
            # Selection: Keep the mutated solution if it's better
            if mutated_confidence > current_confidence:
                self.population = mutated_image
                print(f"  Node {self.node_id}: New solution accepted with confidence {mutated_confidence:.4f}")
            
            # Terminate if the desired confidence is met
            if mutated_confidence >= config.target_confidence:
                self.best_solution = mutated_image
                print(f"✅ Node {self.node_id} found a solution with confidence {mutated_confidence:.4f}")
                return mutated_image
        
        print(f"❌ Node {self.node_id} did not meet the confidence threshold after {config.max_generations} generations.")
