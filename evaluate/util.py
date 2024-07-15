import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from wasabi import msg
import json, time
from tqdm import tqdm
import numpy as np
import re

from rag.api import api as rag_api
from evaluate.prepare.prompts import evaluate_answer_prompt, generate_questions_prompt
from evaluate.prepare.context import context

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def parse_pair(pair_string):
    question_match = re.search(r"<question>(.*?)</question>", pair_string, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", pair_string, re.DOTALL)

    question_match = question_match.group(1).strip() if question_match else ""
    answer_match = answer_match.group(1).strip() if answer_match else ""

    if not question_match:
        msg.warn("No question found in the pair")
    if not answer_match:
        msg.warn("No answer found in the pair")

    return bool(question_match and answer_match), question_match, answer_match

def cold_start(num_questions=5, file_name="data.json", reference_generator="gpt-4-turbo", verbose=False):
    load_dotenv()
    prompt = generate_questions_prompt

    synthetic_data = {
        "data": [],
        "num_data": 0
    }

    total_cnt = 0
    success_cnt = 0
    responses = []
    for ctx in context:
        llm = ChatOpenAI(model=reference_generator)
        chain = prompt | llm | StrOutputParser()
        msg.info(f"Generating questions...")
        start_time = time.time()

        response = ""
        for res in chain.stream({
            "num_questions": num_questions,
            "base_context": ctx["base"]["text"],
            "additional_context": ctx["additional"]["text"]
        }):
            response += res
            if verbose:
                print(res, end="", flush=True)
        if verbose:
            print()
        end_time = time.time()
        msg.good(f"Generated {num_questions} questions in {end_time - start_time:.2f} seconds")
        responses.append(response)


        entries = re.compile("<pair>(.*?)</pair>", re.DOTALL).findall(response)
        total_cnt += len(entries)
        for entry in entries:
            success, question, answer = parse_pair(entry)

            if success:
                success_cnt += 1
                synthetic_data['data'].append({
                    "context": ctx,
                    "question": question,
                    "answer": answer
                })
    synthetic_data['num_data'] = len(synthetic_data['data'])

    msg.good(f"Generated {success_cnt} questions out of {num_questions * len(context)} total questions")

    # check if the json already exists
    file_name_without_extension = file_name.split(".")[0] if len(file_name.split(".")) > 1 else file_name
    extension = file_name.split(".")[1] if len(file_name.split(".")) > 1 else "json"
    raw_file_name = file_name_without_extension + "_raw.txt"
    if os.path.exists(f"./evaluate/reference/{file_name}"):
        tm = int(time.time())
        file_name = file_name_without_extension + f"_{tm}.{extension}"
        raw_file_name = file_name_without_extension + f"_{tm}_raw.txt"

    with open(f"./evaluate/reference/{file_name}", "w") as f:
        json.dump(synthetic_data, f, indent=4)

    with open(f"./evaluate/reference/{raw_file_name}", "w") as f:
        f.write("\n---------------------------------\n".join(responses))

def experiment_single_query(run_config, query, history=[]):
    rag_api.init(run_config["rag_config"])
    generated_response = ""
    base_docs = None
    additional_docs = None
    for response in rag_api.query_stream(query, history):
        generated_response += response
    base_docs = [
        {
            "doc_id": doc_id,
            "chunks_ids": [chunk_id for chunk_id in rag_api.recent_base_docs[doc_id].chunks]
        } for doc_id in rag_api.recent_base_docs
    ]
    additional_docs = [
        {
            "doc_id": doc_id,
            "chunks_ids": [chunk_id for chunk_id in rag_api.recent_additional_docs[doc_id].chunks]
        } for doc_id in rag_api.recent_additional_docs
    ]

    result = {
        "input": {
            "question": query,
            "history": history
        },
        "output": {
            "context": {
                "base": base_docs,
                "additional": additional_docs
            },
            "response": generated_response
        }
    }
    return result

def run_experiment(run_config, num_samples=None):
    experiment_result = {
        "config": run_config,
        "result": []
    }

    reference_path = run_config["input_config"]["reference_path"]
    with open(reference_path, "r") as f:
        reference = json.load(f)
    reference_data = reference["data"][:num_samples]

    queries = [data["question"] for data in reference_data]
    for query in tqdm(queries, total=len(queries)):
        result = experiment_single_query(run_config, query)
        experiment_result["result"].append(result)
    
    # save the reuslt
    experiment_name = run_config["experiment_name"]
    with open(f"./evaluate/experiment/{experiment_name}.json", "w") as f:
        json.dump(experiment_result, f, indent=4)
    
    return experiment_result

def get_retrieval_score_docs(exp_docs, ref_docs):
    exp_doc_ids = set([doc["doc_id"] for doc in exp_docs])
    ref_doc_ids = set([doc["doc_id"] for doc in ref_docs])
    # TODO evaluating chunks? -> unable, chunk id format is different between indices
    return (len(exp_doc_ids.intersection(ref_doc_ids)) / len(ref_doc_ids)) * 5


def get_retrieval_score(exp_ctx, ref_ctx):
    # TODO Many candidate metrics
    base_score = get_retrieval_score_docs(exp_ctx["base"], ref_ctx["base"]["source"])
    additional_score = get_retrieval_score_docs(exp_ctx["additional"], ref_ctx["additional"]["source"])
    return (base_score + additional_score) / 2

def evaluate_experiment(
        experiment_name,
        eval_name=None,
        num_samples=None,
        evaluator="gpt-4-turbo",
    ):
    experiment_path = f"./evaluate/experiment/{experiment_name}.json"
    eval_name = f"{experiment_name}_{evaluator}" if eval_name is None else eval_name
    if os.path.exists(f"./evaluate/evaluation/{eval_name}.json"):
        msg.warn(f"Evaluation {eval_name} already exists")
        return
    
    with open(experiment_path, "r") as f:
        experiment = json.load(f)
    experiment_result = experiment["result"][:num_samples]

    reference_path = experiment["config"]["input_config"]["reference_path"]
    with open(reference_path, "r") as f:
        reference = json.load(f)
    reference_data = reference["data"][:num_samples]
    assert len(experiment_result) == len(reference_data)

    # Scoring
    results = []
    for i in tqdm(range(len(experiment_result)), total=len(experiment_result)):
        exp = experiment_result[i]
        ref = reference_data[i]

        assert exp["input"]["question"] == ref["question"]
        question = exp["input"]["question"]

        exp_ans, exp_ctx = exp["output"]["response"], exp["output"]["context"]
        ref_ans, ref_ctx = ref["answer"], ref["context"]

        # Evaluate the retrieval
        retrieval_score = get_retrieval_score(exp_ctx, ref_ctx)

        # Evaluate the answer
        llm = ChatOpenAI(model=evaluator)
        prmopt = evaluate_answer_prompt
        chain = prmopt | llm | StrOutputParser()

        response = ""
        for res in chain.stream({
            "query": question,
            "reference_answer": ref_ans,
            "generated_answer": exp_ans
        }):
            response += res
            # print(res, end="", flush=True)
        # print()
        answer_score, reasoning = response.split("\n", 1) if "\n" in response else (0, "")

        evaluate_result = {
            "question": question,
            "reference_answer": ref_ans,
            "generated_answer": exp_ans,
            "reference_context": {
                "base": ref_ctx["base"]["source"],
                "additional": ref_ctx["additional"]["source"]
            },
            "generated_context": {
                "base": exp_ctx["base"],
                "additional": exp_ctx["additional"]
            },
            "answer_score": float(answer_score),
            "retrieval_score": float(retrieval_score),
            "reasoning": reasoning
        }
        results.append(evaluate_result)

    evaluation = {
        "config": {**experiment["config"], "evaluator": evaluator},
        "score": {
            "answer": np.mean([result["answer_score"] for result in results]),
            "retrieval": np.mean([result["retrieval_score"] for result in results])
        },
        "result": results,
    }

    # save the evaluation
    with open(f"./evaluate/evaluation/{eval_name}.json", "w") as f:
        json.dump(evaluation, f, indent=4)
    return evaluation


if __name__ == "__main__":
    st = """<pair>
<question>What is the requirement for the availability of source code for products seeking OCP Accepted™ Product Recognition?</question>
<answer>
### Answer
For products seeking OCP Accepted™ Product Recognition, the source code and binary blobs for BMC (if applicable) must be submitted and made available on the specified GitHub repository.

### Changes
There are no changes regarding the BMC source availability requirements in the additional context provided.

### References
- Datacenter NVDatacenter NVMe SSD Specification v2.0r21.pdf
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf
</answer>
</pair>"""
    q, a = parse_pair(st)
    print(q, a)