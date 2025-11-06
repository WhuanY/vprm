import argparse
import io
import logging
import os
import sys
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from datasets import load_dataset
from openai import AzureOpenAI, OpenAI
from rich.logging import RichHandler
from tqdm import tqdm

# from evaluation.build_query import create_query_data
from build_query import create_query_data
from utilities import read_json, save_json

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from PIL import Image

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--data_file_path', type=str, default=None)
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    # Local Model
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model (required for local VLLM inference)')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=20, help='Batch size for inference (default: 20)')
    parser.add_argument("--tp", type=int, default=1, help='Tensor parallel size (default: 1)')
    parser.add_argument("--system_prompt", type=str, default="", help='System prompt for the model')
    # Remote model
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='llm engine',
        choices=['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'],
    )
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    # query
    parser.add_argument('--query_file', type=str, default=None)
    parser.add_argument('--caption_file', type=str, default='../data/texts/captions_bard.json')
    parser.add_argument('--ocr_file', type=str, default='../data/texts/ocrs_easyocr.json')
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type', choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', action='store_true', help='use caption data')
    parser.add_argument('--use_ocr', action='store_true', help='use ocr data')
    parser.add_argument('--pre_prompt', type=str, default="")
    parser.add_argument('--after_prompt',type=str, default="")
    # other settings
    parser.add_argument('--rerun', action='store_true', help='rerun answer extraction for all problems')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))
    args = parser.parse_args()
    return args

# def load_local_dataset(format, data_files):
#     assert format == "parquet", "Only parquet format is supported for local dataset"

def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()

    # load data
    if args.dataset_name: # remote
        logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}...")
        data_list = load_dataset(args.dataset_name, split=args.test_split_name)
    elif args.data_file_path: # local
        data_list = load_dataset("parquet", data_files={"testmini": args.data_file_path}, split="testmini")
    # Convert Hugging Face data into dictionary to match local data format
    # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
    logging.info("Dataset Loaded")
    data = {item['pid']: item for item in data_list}


    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            logging.info(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        logging.info("Creating new query...")

        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                logging.info(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    logging.info("Caption data loaded.")
                except Exception as e:
                    logging.info("Caption data not found!! Please Check.")

        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                logging.info(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    logging.info("OCR data loaded.")
                except Exception as e:
                    logging.info("OCR data not found!! Please Check.")

        query_data = create_query_data(data, caption_data, ocr_data, args)

    # If we were given a custom model path, load that model directly (no API server needed)
    if args.model_path:
        logging.info(f"Using local VLLM model from {args.model_path}...")
        # Initialize processor, tokenizer, llm, and sampling_params
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tp,
            limit_mm_per_prompt={"image": 10, "video": 2},
            gpu_memory_utilization=0.7,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
        )
        model = None  # Will use llm directly for batch processing
        logging.info("Local VLLM model loaded successfully!")
    else:
        model_name = args.azure_openai_model if args.azure_openai_model else args.model
        logging.info(f"Loading {model_name}...")

        if model_name == 'bard':
            from models import bard

            if args.key == '':
                logging.info("Loading key from environment variable")
                key = os.environ['_BARD_API_KEY']
            else:
                key = args.key
            model = bard.Bard_Model(key)
        elif "gpt" in model_name:
            from models import gpt

            key = args.azure_openai_api_key if args.azure_openai_api_key else args.key
            if key == '':
                key = os.getenv("OPENAI_API_KEY")

            assert (
                args.azure_openai_api_endpoint is not None
            ), "Env var AZURE_OPENAI_API_ENDPOINT is not set but is required for OpenAI client."
            assert (
                args.azure_openai_api_key is not None
            ), "Env var AZURE_OPENAI_API_KEY is not set but is required for OpenAI client."
            assert (
                args.azure_openai_api_version is not None
            ), "Env var AZURE_OPENAI_API_VERSION is not set but is required for OpenAI client."
            assert (
                args.azure_openai_model is not None
            ), "Env var AZURE_OPENAI_MODEL is not set but is required for OpenAI client."

            client = AzureOpenAI(
                azure_endpoint=args.azure_openai_api_endpoint,
                api_key=args.azure_openai_api_key,
                api_version=args.azure_openai_api_version,
            )

            model = gpt.GPT_Model(client=client, model=model_name)

        elif "claude" in model_name:
            from models import claude

            if args.key == '':
                logging.info("Loading token from environment variable")
                key = os.environ.get("ANTHROPIC_API_KEY")
            else:
                key = args.key
            model = claude.Claude_Model(model_name, key)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    logging.info(f"Model loaded.")

    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    skip_pids = []
    if not args.rerun:
        for problem_id in full_pids:
            # logging.info(f"Checking {pid}...")
            if problem_id in results and 'response' in results[problem_id]:
                response = results[problem_id]['response']
                if verify_response(response):
                    # logging.info(f"Valid response found for {pid}.")
                    skip_pids.append(problem_id)

    if len(skip_pids) > 0:
        logging.info(
            f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
        )

    test_pids = [pid for pid in full_pids if pid not in skip_pids]
    print("len(test_pids) before", len(test_pids))


    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        print("len(test_pids) before", len(test_pids))

        logging.warning(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    # Process problems based on model type
    if args.model_path:
        # Batch processing with local VLLM (similar to BLINK)
        logging.info(f"Using batch processing with batch_size={args.batch_size}")
        
        def _process_single_item_helper(item_data):
            """Process a single item to prepare it for batch inference"""
            problem_id, problem, query, processor, args = item_data
            
            # Get decoded image (make a copy to avoid modifying original)
            problem_copy = problem.copy()
            problem_decoded_image = problem_copy.get('decoded_image')
            
            # Build messages following MME-RealWorld-Lite pattern
            has_images = problem_decoded_image is not None
            
            if has_images:
                # Add <image> marker to prompt if not present
                if '<image>' not in query:
                    query_with_markers = '<image>\n' + query
                else:
                    query_with_markers = query
                
                # Split by <image> to interleave text and images
                text_parts = query_with_markers.split("<image>")
                content = []
                
                # Build content: image first, then text
                if text_parts[0].strip():
                    content.append({"type": "text", "text": text_parts[0].strip()})
                content.append({"type": "image", "image": problem_decoded_image})
                
                # Add remaining text after image
                if len(text_parts) > 1 and text_parts[-1].strip():
                    content.append({"type": "text", "text": text_parts[-1].strip()})
            else:
                # Pure text case
                content = [{"type": "text", "text": query}]

            messages = [{"role": "user", "content": content}]
            if args.system_prompt:
                messages.insert(0, {"role": "system", "content": args.system_prompt})
            
            # Process with processor
            processed_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Prepare input for VLLM
            if has_images:
                image_data, _ = process_vision_info(messages)
                result = {"prompt": processed_prompt, "multi_modal_data": {"image": image_data}}
            else:
                result = {"prompt": processed_prompt}
            
            # Store metadata for later use
            result["_problem_id"] = problem_id
            result["_problem"] = problem
            result["_query"] = query
            
            return problem_id, result
        
        # Prepare all inputs
        logging.info("Preparing inputs for batch processing...")
        prepared_results = []
        for problem_id in tqdm(test_pids, desc="Preparing inputs"):
            problem = data[problem_id].copy()
            query = query_data[problem_id]
            # Note: problem['decoded_image'] is a PIL Image, which will be used directly
            result = _process_single_item_helper((problem_id, problem, query, processor, args))
            prepared_results.append(result)
        
        # Sort by problem_id to maintain order
        prepared_results.sort(key=lambda x: x[0])
        
        # Extract inputs and metadata
        inputs = [result[1] for result in prepared_results]
        problem_list = [result[1]["_problem"] for result in prepared_results]
        query_list = [result[1]["_query"] for result in prepared_results]
        problem_id_list = [result[1]["_problem_id"] for result in prepared_results]
        
        # Remove metadata from inputs before passing to VLLM
        for inp in inputs:
            inp.pop("_problem_id", None)
            inp.pop("_problem", None)
            inp.pop("_query", None)
        
        # Batch inference
        logging.info(f"Running batch inference with batch_size={args.batch_size}...")
        all_responses = []
        for idx in tqdm(range(0, len(inputs), args.batch_size), 
                       desc="Inferencing",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"):
            batch_inputs = inputs[idx : idx + args.batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_params)
            
            # Extract responses
            for i in range(len(batch_outputs)):
                response_text = batch_outputs[i].outputs[0].text
                all_responses.append(response_text)
        
        # Post-process: create problem results (maintain output format for downstream evaluation)
        logging.info("Post-processing results...")
        for i, (problem_id, problem, query, response) in enumerate(tqdm(
            zip(problem_id_list, problem_list, query_list, all_responses),
            total=len(problem_id_list),
            desc="Post-processing"
        )):
            problem_result = problem.copy()
            # Remove decoded_image for JSON serialization (downstream evaluation doesn't need it)
            problem_result.pop('decoded_image', None)
            problem_result['query'] = query
            
            if args.shot_type == 'solution':
                problem_result['response'] = response
            else:
                output, error = evaluate_code(response)
                problem_result['response'] = response
                problem_result['execution'] = output
                problem_result['error'] = str(error)
            
            results[problem_id] = problem_result
            
            # Save periodically
            if (i + 1) % args.save_every == 0 or (i + 1) == len(problem_id_list):
                try:
                    save_json(results, output_file_path)
                    logging.info(f"Saved results to {output_file_path} (processed {i + 1}/{len(problem_id_list)})")
                except Exception as e:
                    logging.info(f"Error in saving {output_file_path}")
                    logging.info(e)
    else:
        # Remote model processing (original behavior with threading)
        num_threads = getattr(args, 'num_threads', 1)
        logging.info(f"Using {num_threads} thread(s) for concurrent request processing")
        
        # Thread-safe lock for results dictionary and save counter
        results_lock = threading.Lock()
        processed_count = [0]  # Using list to allow modification in nested function

        def process_problem(problem_id, index):
            """Process a single problem and update results thread-safely"""
            problem: dict = data[problem_id].copy()

            # Remove decoded Image for JSON deserialization
            problem_decoded_image = problem['decoded_image']
            problem.pop('decoded_image')

            query = query_data[problem_id]

            logging.debug("--------------------------------------------------------------")
            logging.debug(f"Generating response for problem: {problem_id}...")
            
            problem_result = None
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = model.get_response(user_prompt=query, decoded_image=problem_decoded_image)
                    problem_result = problem.copy()
                    problem_result['query'] = query
                    if args.shot_type == 'solution':
                        problem_result['response'] = response
                    else:
                        output, error = evaluate_code(response)
                        problem_result['response'] = response
                        problem_result['execution'] = output
                        problem_result['error'] = str(error)
                    logging.debug(f"Query: \n{query}")
                    logging.debug(f"Response: \n{response}")
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e)
                    is_internal_error = (
                        "500" in error_str or 
                        "Internal Server Error" in error_str or
                        "Error code: 500" in error_str
                    )
                    
                    if is_internal_error and attempt < max_retries - 1:
                        # Retry for internal server errors
                        wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                        logging.warning(
                            f"Internal server error (500) for problem {problem_id}, "
                            f"attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable error or max retries reached
                        logging.error(f"Error in extracting answer for {problem_id}")
                        logging.error(e)
                        problem_result = problem.copy()
                        problem_result['error'] = str(e)
                        break

            # Thread-safe update of results
            with results_lock:
                results[problem_id] = problem_result
                processed_count[0] += 1
                current_count = processed_count[0]
                
                # Thread-safe saving
                if (current_count % args.save_every == 0) or (current_count == len(test_pids)):
                    try:
                        save_json(results, output_file_path)
                        logging.info(f"Saved results to {output_file_path} (processed {current_count}/{len(test_pids)})")
                    except Exception as e:
                        logging.info(f"Error in saving {output_file_path}")
                        logging.info(e)
            
            return problem_id

        # Process problems concurrently using ThreadPoolExecutor
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks
                future_to_problem = {
                    executor.submit(process_problem, problem_id, i): problem_id 
                    for i, problem_id in enumerate(test_pids)
                }
                
                # Process completed tasks with progress bar
                for future in tqdm(as_completed(future_to_problem), total=len(test_pids), desc="Processing problems"):
                    problem_id = future_to_problem[future]
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Exception occurred while processing {problem_id}: {e}")
        else:
            # Single-threaded mode (original behavior)
            for i, problem_id in enumerate(tqdm(test_pids, desc="Processing problems")):
                process_problem(problem_id, i)

    logging.info("MathVista: Generating Responses - Finish")


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
