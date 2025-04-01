# /utils/evaluation.py
import numpy as np

def evaluate_fitness(image, model, target_class):
    image_flat = image.flatten().reshape(1, -1)
    decision_function = model.decision_function(image_flat)[0]
    
    # Confidence for target class
    confidence = decision_function[target_class]
    
    # Scale confidence to be positive for fitness evaluation
    return 1 / (1 + np.exp(-confidence))
