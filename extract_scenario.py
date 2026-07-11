import argparse
import json
from pathlib import Path

def extract():
    parser = argparse.ArgumentParser(description="Extract and inspect WildToolBench evaluation scenarios and failure cases.")
    parser.add_argument("--model", type=str, default="Qwen_Qwen2.5-7B-Instruct", help="Model folder name (e.g., Qwen_Qwen2.5-7B-Instruct, llama3.1_latest)")
    parser.add_argument("--id", type=str, help="Specific test case ID (e.g., wild_tool_bench_0)")
    parser.add_argument("--error-only", action="store_true", help="Only show turns that resulted in an error")
    parser.add_argument("--error-type", type=str, help="Filter by error substring (e.g., 'mismatch', 'type inconsistent', 'not in")
    parser.add_argument("--keyword", type=str, help="Search for a keyword in the user prompt")
    args = parser.parse_args()

    score_dir = Path("score")
    model_dir = score_dir / args.model
    score_file = model_dir / "Wild-Tool-Bench_score.jsonl"
    dataset_file = Path("data/Wild-Tool-Bench.jsonl")

    if not score_file.exists():
        print(f"[-] Error: Results file for model '{args.model}' not found at {score_file}")
        print("    Make sure the evaluation run has generated some results first.")
        print(f"    Current search directory: {score_file.resolve()}")
        return

    # Load dataset for context
    dataset = {}
    if dataset_file.exists():
        with open(dataset_file) as f:
            for line in f:
                entry = json.loads(line)
                dataset[entry["id"]] = entry

    # Process score file
    results_found = 0
    with open(score_file) as f:
        for line in f:
            score_entry = json.loads(line)
            id_ = score_entry["id"]
            
            # Filter by ID
            if args.id and id_ != args.id:
                continue

            entry_ctx = dataset.get(id_, {})
            tasks = entry_ctx.get("english_tasks", [])
            turn_subtypes = ["First Turn"] + entry_ctx.get("english_turn_subtypes", [])
            task_types = entry_ctx.get("english_task_types", [])
            expected_answers = entry_ctx.get("english_answer_list", [])

            # Temp print blocks for matched turns
            matched_turns = []
            results = score_entry.get("results", [])
            for turn_idx, res in enumerate(results):
                label = res.get("label", "N/A")
                is_error = label == "error"

                # Filter by error-only
                if args.error_only and not is_error:
                    continue

                task_text = tasks[turn_idx] if turn_idx < len(tasks) else "N/A"
                
                # Filter by keyword
                if args.keyword and args.keyword.lower() not in task_text.lower():
                    continue

                # Find errors in inference logs
                step_logs = []
                inf_log = res.get("inference_log", {})
                error_match = False
                err_msg = ""
                
                for step_key, step_val in inf_log.items():
                    if not step_key.startswith("step_"):
                        continue
                    output = step_val.get("inference_output", {})
                    
                    if output.get("current_action_name_label") == "error":
                        err_msg = f"Action Name Error: {output.get('error_reason')}"
                        if not args.error_type or args.error_type.lower() in err_msg.lower():
                            error_match = True
                    elif output.get("current_action_arguments_label") == "error":
                        err_msg = f"Arguments Error: {output.get('current_action_arguments_check_result')}"
                        if not args.error_type or args.error_type.lower() in err_msg.lower():
                            error_match = True
                    
                    tool_calls_str = []
                    for tc in output.get("tool_calls", []) or []:
                        fn = tc.get("function", {})
                        tool_calls_str.append(f"{fn.get('name')}({fn.get('arguments')})")
                    
                    step_logs.append({
                        "step": step_key,
                        "called": tool_calls_str,
                        "text": output.get("content", ""),
                        "err": err_msg
                    })

                # Filter by error type
                if args.error_type and not error_match:
                    continue

                matched_turns.append({
                    "turn_idx": turn_idx,
                    "task": task_text,
                    "task_type": task_types[turn_idx] if turn_idx < len(task_types) else "N/A",
                    "subtype": turn_subtypes[turn_idx] if turn_idx < len(turn_subtypes) else "N/A",
                    "status": label,
                    "is_optimal": res.get("is_optimal", False),
                    "steps": step_logs,
                    "expected": expected_answers[turn_idx] if turn_idx < len(expected_answers) else []
                })

            if matched_turns:
                results_found += 1
                print(f"# Test Case: {id_}")
                print(f"**Environment Info:** `{entry_ctx.get('english_env_info', 'N/A')}`\n")
                for turn in matched_turns:
                    print(f"### Turn {turn['turn_idx']}: {turn['task_type']} | {turn['subtype']}")
                    print(f"> **User Request:** \"{turn['task']}\"\n")
                    print(f"* **Status:** `{turn['status']}` (Optimal: `{turn['is_optimal']}`)\n")
                    
                    print("**Expected Actions:**")
                    for e_idx, exp in enumerate(turn["expected"]):
                        act = exp.get("action", {})
                        print(f"  * `{act.get('name')}({act.get('arguments')})`")
                    
                    print("\n**Model Execution Steps:**")
                    for step in turn["steps"]:
                        print(f"  * **`{step['step']}`**:")
                        if step["called"]:
                            for c in step["called"]:
                                print(f"    * Called: `{c}`")
                        else:
                            print(f"    * Text: *\"{step['text'][:200]}...\"*")
                        if step["err"]:
                            print(f"    * ⚠️ **Error Detail:** `{step['err']}`")
                    print("\n" + "---" + "\n")

    if results_found == 0:
        print("[-] No scenarios matched the search filters.")

if __name__ == "__main__":
    extract()
