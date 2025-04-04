import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import trange


# --- Config ---
image_shape = (28, 28)
pixel_max = 255
mutation_rate = 0.01
max_generations = 20000000
target_confidence = 0.9999
model_path = "model_SVM_mnist.pkl"
target_class = 0  # evolve toward digit "0"

# --- Load trained SVM model ---
model = joblib.load(model_path)

# --- Fitness Function ---
def evaluate_fitness(image, model, target_class):
    image_flat = image.flatten().reshape(1, -1)
    return model.predict_proba(image_flat)[0][target_class]

# --- Mutation Function ---
def mutate(image, mutation_rate):
    mutated = np.copy(image)
    num_pixels = int(mutation_rate * image.size)
    for _ in range(num_pixels):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        mutated[x, y] = np.clip(mutated[x, y] + np.random.randint(-20, 21), 0, pixel_max)
    return mutated

# --- Save Image ---
def save_image(image, confidence, generation):
    os.makedirs("evolved_output", exist_ok=True)
    plt.imshow(image, cmap='gray')
    plt.title(f"Gen {generation} - Conf: {confidence:.4f}")
    plt.axis('off')
    plt.savefig(f"evolved_output/gen_{generation:03d}.png")
    plt.close()

# --- Evolution Loop ---
image = np.random.randint(0, pixel_max + 1, image_shape, dtype=np.uint8)
fitness_progress = []

pbar = trange(max_generations, desc="Evolving Image", leave=True)
for gen in pbar:
    current_conf = evaluate_fitness(image, model, target_class)
    fitness_progress.append(current_conf)

    mutated = mutate(image, mutation_rate)
    mutated_conf = evaluate_fitness(mutated, model, target_class)

    pbar.set_description(f"Gen {gen} | Conf: {current_conf:.4f}")

    if gen % 10 == 0 or mutated_conf >= target_confidence:
        save_image(mutated, mutated_conf, gen)

    if mutated_conf > current_conf:
        image = mutated
        # pbar.write(f"✅ Gen {gen}: New solution accepted (conf={mutated_conf:.4f})")

    if mutated_conf >= target_confidence:
        pbar.write(f"🎯 Target confidence reached at generation {gen}: {mutated_conf:.4f}")
        break

# --- Plot final confidence curve ---
plt.plot(fitness_progress)
plt.xlabel("Generation")
plt.ylabel("Confidence")
plt.title("Confidence over Evolution")
plt.savefig("evolved_output/confidence_plot.png")
plt.close()
