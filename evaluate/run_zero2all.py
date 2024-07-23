import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from evaluate.util import cold_start, run_experiment, evaluate_experiment


enable_cold_start = False
enable_experiment = True
enable_evaluation = False

experiment_name = "experiment_revise_hyde_kb_4"
eval_name = "experiment_revise_hyde_kb_gpt-4-turbo_4"
ref_json_name = "data_small.json"

num_questions = 3
num_samples = None

run_config = {
    "experiment_name": experiment_name,
    "input_config": {
        "reference_path": f"./evaluate/reference/{ref_json_name}"
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
            "revise_query": True,
            "hyde": True,
        }
    }
}

if __name__ == "__main__":
    if enable_cold_start:
        print("Generating questions...")
        cold_start(num_questions=num_questions, file_name=ref_json_name , reference_generator="gpt-4-turbo", verbose=False)
        print("Finished generating questions\n\n")

    if enable_experiment:
        print("Running experiment...")
        run_experiment(run_config, num_samples=num_samples)
        print("Finished running experiment\n\n")

    if enable_evaluation:
        print("Evaluating experiment...")
        evaluate_experiment(experiment_name, eval_name=eval_name, num_samples=num_samples)
        print("Finished evaluating experiment")
