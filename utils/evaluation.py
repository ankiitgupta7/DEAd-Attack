import numpy as np
import pandas as pd

def evaluate_fitness(image, model, target_class):
    image_flat = image.flatten().reshape(1, -1)

    # If model was trained with column names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
        df = pd.DataFrame(image_flat, columns=feature_names)
        confidence = model.predict_proba(df)[0][target_class]
    else:
        # fallback
        confidence = model.predict_proba(image_flat)[0][target_class]

    return confidence
