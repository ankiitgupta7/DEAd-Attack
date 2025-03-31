# /utils/evaluation.py
import numpy as np

def evaluate_fitness(image, model, target_class):
    image_flat = image.flatten().reshape(1, -1)
    confidence = model.predict_proba(image_flat)[0][target_class]
    return confidence
