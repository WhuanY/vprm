import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import AzureOpenAI, OpenAI
from rich.logging import RichHandler
from tqdm import tqdm

# from evaluation.prompts.ext_ans import demo_prompt
from prompts.ext_ans import demo_prompt, demo_prompt_w_cot
# from models import gpt
from utilities import read_json, save_json


class CustomizedRemoteGPTModel:
    def __init__(self, client, model):
        self.client = client
        self.model = model # gpt-4o-mini

    def get_response(self, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        return response.choices[0].message.content

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(model, response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""
    
    extraction = None
    answer_match = "<answer>" in response and "</answer>" in response
    if answer_match:
        extraction = response.split('<answer>')[-1].split('</answer>')[0].strip()
    
    if question_type == 'multi_choice':
        if response in choices: # choices是选项内容，而不是ABCDEEF，这个老六
            return response
        if extraction is not None and extraction in choices:
            return extraction
        if extraction is not None:
            for c in "ABCDE":
                if extraction == c or extraction == c.lower():
                    return c
                if extraction == f"({c})":
                    return c

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

        if extraction is not None:
            try:
                extraction = int(extraction)
                return str(extraction)
            except Exception as e:
                pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass
        if extraction is not None:
            try:
                extraction = str(float(extraction))
                return extraction
            except Exception as e:
                pass

    # quick extraction
    if quick_extract:
        logging.info("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        if answer_match:
            full_prompt = create_test_prompt(demo_prompt_w_cot, query, response)
        extraction = model.get_response(user_prompt=full_prompt)
        return extraction
    except Exception as e:
        logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logging.info(e)

    return ""


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The max number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

    # parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    # parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    # parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    # parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))
    parser.add_argument('--customized_remote_openai_api_endpoint', type=str, default=os.getenv("CUSTOMIZED_REMOTE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--customized_remote_openai_api_key', type=str, default=os.getenv("CUSTOMIZED_REMOTE_OPENAI_API_KEY"))
    parser.add_argument('--num_threads', type=int, default=1, help='Number of concurrent threads for sending requests')

    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Extract Answers - Start")
    args = parse_args()

    # args
    label = args.response_label

    assert (
        args.customized_remote_openai_api_endpoint is not None
    ), "Env var CUSTOMIZED_REMOTE_OPEN_API_ENDPOINT is not set but is required for OpenAI client."
    assert (
        args.customized_remote_openai_api_key is not None
    ), "Env var CUSTOMIZED_REMOTE_OPEN_API_KEY is not set but is required for OpenAI client."

    # assert (
    #     args.azure_openai_api_endpoint is not None
    # ), "Env var AZURE_OPENAI_API_ENDPOINT is not set but is required for OpenAI client."
    # assert (
    #     args.azure_openai_api_key is not None
    # ), "Env var AZURE_OPENAI_API_KEY is not set but is required for OpenAI client."
    # assert (
    #     args.azure_openai_api_version is not None
    # ), "Env var AZURE_OPENAI_API_VERSION is not set but is required for OpenAI client."
    # assert (
    #     args.azure_openai_model is not None
    # ), "Env var AZURE_OPENAI_MODEL is not set but is required for OpenAI client."

    # client = AzureOpenAI(
    #     azure_endpoint=args.azure_openai_api_endpoint,
    #     api_key=args.azure_openai_api_key,
    #     api_version=args.azure_openai_api_version,
    # )
    client = OpenAI(
        base_url=args.customized_remote_openai_api_endpoint,
        api_key=args.customized_remote_openai_api_key,
    )
    # model = gpt.GPT_Model(client=client, model=args.azure_openai_model)
    model = CustomizedRemoteGPTModel(client=client, model="gpt-4o-mini")

    logging.info(f"Reading {args.results_file_path}...")
    results = read_json(args.results_file_path)

    full_pids = list(results.keys())

    skip_pids = []
    for pid, problem in results.items():
        extraction = problem.get('extraction')
        if extraction is not None and verify_extraction(extraction):
            skip_pids.append(problem['pid'])

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
            )
        test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.info(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")
    logging.info(f"Using {args.num_threads} thread(s) for concurrent request processing")

    # Thread-safe lock for results dictionary and save counter
    results_lock = threading.Lock()
    processed_count = [0]  # Using list to allow modification in nested function

    def process_problem(pid, index):
        """Process a single problem and update results thread-safely"""
        problem = results[pid].copy()

        assert label in problem
        response = problem[label]
        extraction = extract_answer(model, response, problem, args.quick_extract)

        # Thread-safe update of results
        with results_lock:
            results[pid]['extraction'] = extraction
            processed_count[0] += 1
            current_count = processed_count[0]
            
            # Thread-safe saving
            if (current_count % args.save_every == 0) or (current_count == len(test_pids)):
                try:
                    save_json(results, args.results_file_path)
                    logging.info(f"Saved results to {args.results_file_path} (processed {current_count}/{len(test_pids)})")
                except Exception as e:
                    logging.info(f"Error in saving {args.results_file_path}")
                    logging.info(e)
        
        return pid

    # Process problems concurrently using ThreadPoolExecutor
    if args.num_threads > 1:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all tasks
            future_to_problem = {
                executor.submit(process_problem, pid, i): pid 
                for i, pid in enumerate(test_pids)
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_problem), total=len(test_pids), desc="Processing problems"):
                pid = future_to_problem[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Exception occurred while processing {pid}: {e}")
    else:
        # Single-threaded mode (original behavior)
        for i, pid in enumerate(tqdm(test_pids, desc="Processing problems")):
            process_problem(pid, i)

    logging.info("MathVista: Extract Answers - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
