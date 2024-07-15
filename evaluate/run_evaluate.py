import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from evaluate.util import *

if __name__ == "__main__":    
    experiment_name = "experiment_test"
    evaluate_experiment(experiment_name, num_samples=3)