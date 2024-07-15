import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from evaluate.util import *

if __name__ == "__main__":
    load_dotenv()
    
    experiment_name = "experiment_test"
    evaluate_experiment(experiment_name, num_samples=3)