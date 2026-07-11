import json
import os
import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from tqdm import tqdm

current_path_list = os.getcwd().split("/")[:-2]
current_path = "/".join(current_path_list)
print(f"current_path: {current_path}\n", flush=True)
sys.path.append(current_path)

from wtb.constant import PROJECT_ROOT, RESULT_PATH, PROMPT_PATH, TEST_IDS_TO_GENERATE_PATH
from wtb.utils import load_file, sort_key
from wtb.model_handler.handler_map import HANDLER_MAP


RETRY_LIMIT = 3
# 60s for the timer to complete. But often we find that even with 60 there is a conflict. So 65 is a safe no.
RETRY_DELAY = 65  # Delay in seconds


def get_involved_test_entries(run_ids):
    all_test_entries_involved = []
    if run_ids:
        with open(TEST_IDS_TO_GENERATE_PATH) as f:
            test_ids = json.load(f)
        if len(test_ids) != 0:
            all_test_entries_involved.extend(
                [
                    entry
                    for entry in load_file(PROMPT_PATH)
                    if entry["id"] in test_ids
                ]
            )
    else:
        all_test_entries_involved.extend(load_file(PROMPT_PATH))

    return all_test_entries_involved


def collect_test_cases(args, model_name, all_test_entries_involved):
    model_name_dir = model_name.replace("/", "_")
    model_result_dir = args.result_dir / model_name_dir

    existing_result = []
    result_file_path = model_result_dir / os.path.basename(PROMPT_PATH).replace(".jsonl", "_result.jsonl")
    if result_file_path.exists():
        # Not allowing overwrite, we will load the existing results
        if not args.allow_overwrite:
            existing_result.extend(load_file(result_file_path))
        # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
        elif not args.run_ids:
            result_file_path.unlink()
        # Allow overwrite and running specific test ids, we will do nothing here
        else:
            pass

    existing_ids = [entry["id"] for entry in existing_result]

    test_cases_to_generate = [
        test_case
        for test_case in all_test_entries_involved
        if test_case["id"] not in existing_ids
    ]

    return sorted(test_cases_to_generate, key=sort_key)


def build_handler(model_name, temperature):
    handler = HANDLER_MAP[model_name](model_name, temperature)
    return handler


def generate_results(args, model_name, test_cases_total):
    handler = build_handler(model_name, args.temperature)

    futures = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        with tqdm(
            total=len(test_cases_total), desc=f"Generating results for {model_name}"
        ) as pbar:

            for test_case in test_cases_total:
                future = executor.submit(
                    multi_threaded_inference,
                    handler,
                    model_name,
                    test_case
                )
                futures.append(future)

            for future in futures:
                # This will wait for the task to complete, so that we are always writing in order
                result = future.result()
                handler.write(
                    result, result_dir=args.result_dir, update_mode=args.run_ids
                )  # Only when we run specific test ids, we will need update_mode=True to keep entries in the same order
                pbar.update()


def multi_threaded_inference(handler, model_name, test_case):
    retry_count = 0

    while True:
        try:
            result = handler.inference(deepcopy(test_case))
            break  # Success, exit the loop
        except Exception as e:
            # OpenAI has openai.RateLimitError while Anthropic has anthropic.RateLimitError. It would be more robust in the long run.
            if retry_count < RETRY_LIMIT and (
                "rate limit reached" in str(e).lower()
                or (hasattr(e, "status_code") and e.status_code in {429, 503, 500})
            ):
                print(
                    f"Rate limit reached. Sleeping for 65 seconds. Retry {retry_count + 1}/{RETRY_LIMIT}"
                )
                time.sleep(RETRY_DELAY)
                retry_count += 1
            else:
                # This is usually the case when the model getting stuck on one particular test case.
                # For example, timeout error or FC model returning invalid JSON response.
                # Since temperature is already set to 0.001, retrying the same test case will not help.
                # So we continue the generation process and record the error message as the model response
                print("-" * 100)
                print(
                    "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
                )
                print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
                traceback.print_exc()
                print("-" * 100)

                return {
                    "id": test_case["id"],
                    "result": f"Error during inference: {str(e)}"
                }

    result_to_write = {
        "id": test_case["id"],
        "model_name": model_name,
        "result": result
    }

    return result_to_write


def main(args):
    if type(args.model) != list:
        args.model = [args.model]

    all_test_entries_involved = get_involved_test_entries(args.run_ids)

    print(f"Generating results for {args.model}")
    if args.run_ids:
        print("Running specific test cases.")
    else:
        print("Running full test cases.")

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    for model_name in args.model:
        test_cases_total = collect_test_cases(
            args,
            model_name,
            all_test_entries_involved
        )

        if len(test_cases_total) == 0:
            print(
                f"All selected test cases have been previously generated for {model_name}. No new test cases to generate."
            )
        else:
            generate_results(args, model_name, test_cases_total)


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="deepseek-chat", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", action="store_true", default=False)

    args = parser.parse_args()
    return args
