# /core/node.py

import numpy as np
import matplotlib.pyplot as plt
import os
from utils.mutation import mutate
from utils.evaluation import evaluate_fitness
from config import config

class Node:
    def __init__(self, cluster_id, local_node_id, model, target_class):
        self.cluster_id = cluster_id
        self.local_node_id = local_node_id
        self.global_id = f"C{cluster_id}-N{local_node_id}"

        self.model = model
        self.target_class = target_class
        self.population = self.initialize_population()
        self.buffer = []
        self.best_solution = None
        self.best_fitness = 0.0
        self.confidence_progress = []

    def initialize_population(self):
        return np.random.randint(0, 17, (8, 8))

    def save_image(self, image, gen, confidence, round_num=None):
        output_dir = f"output_images/{self.global_id}/R{round_num}" if round_num is not None else f"output_images/{self.global_id}"
        os.makedirs(output_dir, exist_ok=True)
        tag = f"R{round_num}_" if round_num is not None else ""
        filename = f"{tag}{self.global_id}_gen_{gen}.png"
        
        plt.imshow(image, cmap='gray')
        plt.title(f"{self.global_id} - Gen {gen} - Conf: {confidence:.4f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def plot_confidence_progress(self, round_num=None):
        tag = f"R{round_num}_" if round_num is not None else ""
        plt.plot(self.confidence_progress, label=f"{self.global_id}")
        plt.xlabel("Generation")
        plt.ylabel("Confidence")
        plt.title(f"Confidence Progress - {self.global_id}")
        plt.legend()
        filename = f"{tag}{self.global_id}_confidence.png"
        plt.savefig(os.path.join("output_images", filename))
        plt.close()


    def communicate_with_neighbors(self):
        for neighbor in self.buffer:
            if neighbor.best_solution is not None:
                neighbor_conf = evaluate_fitness(neighbor.best_solution, self.model, self.target_class)
                my_conf = evaluate_fitness(self.population, self.model, self.target_class)
                if neighbor_conf > my_conf:
                    self.population = neighbor.best_solution
                    self.best_fitness = neighbor_conf
                    self.best_solution = neighbor.best_solution
                    print(f"📬 {self.global_id} adopted better solution from {neighbor.global_id} (conf={neighbor_conf:.4f})")

    def evolve(self, round_num=None):
        for gen in range(config.max_generations):
            current_confidence = evaluate_fitness(self.population, self.model, self.target_class)
            mutated_image = mutate(self.population, config.mutation_rate)
            mutated_confidence = evaluate_fitness(mutated_image, self.model, self.target_class)

            self.confidence_progress.append(current_confidence)

            if gen % 5 == 0 or mutated_confidence >= config.target_confidence:
                self.save_image(mutated_image, gen, mutated_confidence, round_num)

            if mutated_confidence > current_confidence:
                self.population = mutated_image
#                print(f"✅ {self.global_id}: New solution accepted with confidence {mutated_confidence:.4f}")

            if mutated_confidence > self.best_fitness:
                self.best_fitness = mutated_confidence
                self.best_solution = mutated_image

            if mutated_confidence >= config.target_confidence:
                print(f"🎯 {self.global_id} found a solution with confidence {mutated_confidence:.4f}")
                self.save_image(mutated_image, gen, mutated_confidence, round_num)
                self.plot_confidence_progress(round_num)
                return

            if gen > 0 and gen % config.communication_interval == 0:
                self.communicate_with_neighbors()

        # print(f"❌ {self.global_id} did not meet the threshold after {config.max_generations} generations.")
        self.save_image(self.population, config.max_generations, current_confidence, round_num)
        self.plot_confidence_progress(round_num)
