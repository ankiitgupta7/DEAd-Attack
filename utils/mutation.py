# /utils/mutation.py
import numpy as np

def mutate(image, mutation_rate):
    mutated_image = np.copy(image)
    num_mutations = int(mutation_rate * 64)  # 64 pixels in 8x8 image
    for _ in range(num_mutations):
        x, y = np.random.randint(0, 8), np.random.randint(0, 8)
        mutated_image[x, y] = np.clip(mutated_image[x, y] + np.random.randint(-1, 2), 0, 16)
    return mutated_image
