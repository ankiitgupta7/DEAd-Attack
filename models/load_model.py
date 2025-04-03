from config import config
import joblib
import os

def load_trained_model():
    if config.model_file:
        model_path = config.model_file
    else:
        # fallback: guess model filename from config
        model_path = f"model_{config.model_name}_{config.dataset_name}.pkl"

    # Set config dynamically based on dataset in the filename
    if "mnist" in model_path:
        config.image_height = 28
        config.image_width = 28
        config.pixel_max = 255
    elif "digits" in model_path:
        config.image_height = 8
        config.image_width = 8
        config.pixel_max = 16

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    return model
