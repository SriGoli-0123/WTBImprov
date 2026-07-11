import json
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Identify failed/missing test cases in WildToolBench results and write their IDs to test_case_ids_to_generate.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="Model name as used in evaluation")
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    # Assuming this script is run from wildtoolbench_workspace_vllm
    base_dir = script_dir
    result_dir = base_dir / "WildToolBench" / "wild-tool-bench" / "result"
    prompt_file = base_dir / "WildToolBench" / "wild-tool-bench" / "data" / "Wild-Tool-Bench.jsonl"
    test_ids_path = base_dir / "WildToolBench" / "wild-tool-bench" / "test_case_ids_to_generate.json"

    model_dir_name = args.model.replace("/", "_")
    result_file = result_dir / model_dir_name / "Wild-Tool-Bench_result.jsonl"

    if not prompt_file.exists():
        print(f"Error: Prompt file not found at {prompt_file}")
        return

    # 1. Load all target test case IDs from Wild-Tool-Bench.jsonl
    all_target_ids = []
    with open(prompt_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                all_target_ids.append(entry["id"])
            except Exception as e:
                print(f"Error parsing prompt line: {e}")

    total_target = len(all_target_ids)
    print(f"Loaded {total_target} total target test cases from prompt file.")

    # 2. Check existing results
    success_ids = set()
    failed_ids = set()
    total_existing = 0

    if result_file.exists():
        with open(result_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    total_existing += 1
                    entry_id = entry["id"]
                    result = entry.get("result")
                    
                    # If result is a string, it means it contains "Error during inference: ..."
                    is_failed = False
                    if isinstance(result, str) and ("error" in result.lower() or "timeout" in result.lower()):
                        is_failed = True
                    elif not isinstance(result, list):
                        is_failed = True
                        
                    if is_failed:
                        failed_ids.add(entry_id)
                    else:
                        success_ids.add(entry_id)
                except Exception as e:
                    print(f"Error parsing result line: {e}")
    else:
        print(f"Result file does not exist yet at {result_file}")

    # 3. Find missing IDs
    missing_ids = [entry_id for entry_id in all_target_ids if entry_id not in success_ids]

    print(f"=== WildToolBench Result Analysis for {args.model} ===")
    print(f"Total target test cases: {total_target}")
    print(f"Successfully generated so far: {len(success_ids)}")
    print(f"Failed/Error entries in result file: {len(failed_ids)}")
    print(f"Total cases to generate/regenerate (failed or missing): {len(missing_ids)}")

    if missing_ids:
        print(f"\nWriting {len(missing_ids)} IDs to {test_ids_path}...")
        with open(test_ids_path, "w") as f:
            json.dump(missing_ids, f, indent=2)
        print("Done! You can now run the evaluation script with --run-ids --allow-overwrite to regenerate these cases.")
    else:
        print("\nAll entries generated successfully! No failed or missing cases to run.")

if __name__ == "__main__":
    main()
