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
        self.buffer = []  # Stores neighbor solutions (List[Node])
        self.best_solution = None
        self.best_fitness = 0.0
        self.confidence_progress = []  # Track confidence progression

    def initialize_population(self):
        # Generate a random 8x8 grayscale image with pixel values [0-16]
        return np.random.randint(0, 17, (8, 8))

    def communicate_with_neighbors(self):
        """Exchange solutions with neighbors every 'communication_interval' generations."""
        for neighbor in self.buffer:
            if neighbor.best_solution is not None:
                neighbor_conf = evaluate_fitness(neighbor.best_solution, self.model, self.target_class)
                my_conf = evaluate_fitness(self.population, self.model, self.target_class)
                if neighbor_conf > my_conf:
                    self.population = neighbor.best_solution
                    self.best_fitness = neighbor_conf
                    self.best_solution = neighbor.best_solution
                    print(f"Node {self.node_id} adopted a better solution from neighbor with confidence {neighbor_conf:.4f}")

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
        """Main evolutionary loop."""
        for gen in range(config.max_generations):
            # Current & mutated solution
            current_confidence = evaluate_fitness(self.population, self.model, self.target_class)
            mutated_image = mutate(self.population, config.mutation_rate)
            mutated_confidence = evaluate_fitness(mutated_image, self.model, self.target_class)

            # Track confidence progression
            self.confidence_progress.append(current_confidence)

            # Possibly save an image
            if gen % 5 == 0 or mutated_confidence >= config.target_confidence:
                self.save_image(mutated_image, gen, mutated_confidence)

            # Selection: keep mutated if it's better
            if mutated_confidence > current_confidence:
                self.population = mutated_image
                # print(f"Node {self.node_id}: New solution accepted with confidence {mutated_confidence:.4f}")

            # Always track best fitness
            if mutated_confidence > self.best_fitness:
                self.best_fitness = mutated_confidence
                self.best_solution = mutated_image

            # Check termination
            if mutated_confidence >= config.target_confidence:
                print(f"Node {self.node_id} found a solution with confidence {mutated_confidence:.4f}")
                self.save_image(mutated_image, gen, mutated_confidence)
                self.plot_confidence_progress()
                return

            # ▶️ Communicate with neighbors at interval (if more than 0 generations)
            if gen > 0 and (gen % config.communication_interval == 0):
                # print(f"Node {self.node_id} is communicating with neighbors at generation {gen}.")
                self.communicate_with_neighbors()

        # After finishing all generations
        print(f"Node {self.node_id} did not meet threshold after {config.max_generations} generations.")
        self.save_image(self.population, config.max_generations, current_confidence)
        self.plot_confidence_progress()
