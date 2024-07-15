import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evaluate.util import run_experiment
    

if __name__ == "__main__":
    run_config = {
        "experiment_name": "experiment_test",
        "input_config": {
            "reference_path": "./evaluate/reference/data.json"
        },
        "rag_config": {
            "model": {
                "Reader": {"selected": ""},
                "Chunker": {"selected": ""},
                "Embedder": {"selected": ""},
                "Retriever": {"selected": "knowledge-base"},
                "Generator": {"selected": "gpt4"},
                "Revisor": {"selected": "gpt"},
            },
            "pipeline": {
                "revise_query": False,
                "hyde": False,
            }
        }
    }
    run_experiment(run_config, num_samples=3)