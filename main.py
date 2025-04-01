# /main.py
from models.load_model import load_trained_model
from simulation.run_simulation import run_simulation
from config import config


if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model()
    
    # Define target class to attack
    target_class = 8  # Example: Trick classifier into predicting '8' instead of another digit
    
    # Run the simulation
    run_simulation(model, target_class)
