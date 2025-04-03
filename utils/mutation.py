# /utils/mutation.py
import numpy as np
from config import config

def mutate(image, mutation_rate):
    mutated_image = np.copy(image)
    # For an H×W image, total pixels = image.size
    num_mutations = int(mutation_rate * image.size)

    for _ in range(num_mutations):
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])
        # random in [-1, 1]
        delta = np.random.randint(-1, 2)
        mutated_image[x, y] = np.clip(mutated_image[x, y] + delta, 0, config.pixel_max)

    return mutated_image
