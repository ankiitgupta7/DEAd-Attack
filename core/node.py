# /core/node.py
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.mutation import mutate
from utils.evaluation import evaluate_fitness
from config import config

class Node:
    def __init__(self, node_id, model, target_class):
        self.node_id = node_id
        self.model = model
        self.target_class = target_class
        self.population = self.initialize_population()
        self.buffer = []  # Stores neighbor solutions
        self.best_solution = None
        self.confidence_progress = []  # To track confidence progression

    def initialize_population(self):
        # Generate a random 8x8 grayscale image with pixel values [0-16]
        return np.random.randint(0, 17, (8, 8))
    
    def save_image(self, image, gen, confidence):
        """Save the 8x8 evolved image for visualization."""
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        plt.imshow(image, cmap='gray')
        plt.title(f"Node {self.node_id} - Gen {gen} - Conf: {confidence:.4f}")
        plt.axis('off')
        file_path = os.path.join(output_dir, f"node_{self.node_id}_gen_{gen}.png")
        plt.savefig(file_path)
        plt.close()

    def plot_confidence_progress(self):
        """Plot confidence progression over generations."""
        plt.plot(self.confidence_progress, label=f"Node {self.node_id}")
        plt.xlabel("Generation")
        plt.ylabel("Confidence")
        plt.title(f"Confidence Progression for Node {self.node_id}")
        plt.legend()
        plt.savefig(f"output_images/node_{self.node_id}_confidence.png")
        plt.close()

    def evolve(self):
        for gen in range(config.max_generations):
            # Mutate and evaluate population
            mutated_image = mutate(self.population, config.mutation_rate)
            current_confidence = evaluate_fitness(self.population, self.model, self.target_class)
            mutated_confidence = evaluate_fitness(mutated_image, self.model, self.target_class)

            # Log confidence progression
            self.confidence_progress.append(current_confidence)

            # Save evolved image at intervals (e.g., every 5 generations)
            if gen % 5 == 0 or mutated_confidence >= config.target_confidence:
                self.save_image(mutated_image, gen, mutated_confidence)

            # Selection: Keep the mutated solution if it's better
            if mutated_confidence > current_confidence:
                self.population = mutated_image
                print(f"  Node {self.node_id}: New solution accepted with confidence {mutated_confidence:.4f}")

            # Save final best image if confidence is met
            if mutated_confidence >= config.target_confidence:
                self.best_solution = mutated_image
                self.save_image(mutated_image, gen, mutated_confidence)
                print(f"✅ Node {self.node_id} found a solution with confidence {mutated_confidence:.4f}")
                self.plot_confidence_progress()
                return

        # Save final evolved image if no solution was found
        self.save_image(self.population, config.max_generations, current_confidence)
        self.plot_confidence_progress()

    def communicate_with_neighbors(self):
        """Exchange solutions with neighbors periodically."""
        for neighbor in self.buffer:
            print(f"🔁 Node {self.node_id} is communicating with neighbors, sending its solution...")
            neighbor_solution = neighbor.best_solution
            if neighbor_solution is not None:
                neighbor_confidence = evaluate_fitness(neighbor_solution, self.model, self.target_class)
                current_confidence = evaluate_fitness(self.population, self.model, self.target_class)
                if neighbor_confidence > current_confidence:
                    self.population = neighbor_solution
                    print(f"📬 Node {self.node_id} received a better solution from a neighbor with confidence {neighbor_confidence:.4f}")
