# /utils/evaluation.py
import numpy as np

def evaluate_fitness(image, model, target_class):
    image_flat = image.flatten().reshape(1, -1)
    
    # Confidence for target class
    confidence = model.predict_proba(image_flat)[0][target_class]

    
    # Scale confidence to be positive for fitness evaluation
    return confidence
