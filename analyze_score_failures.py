import json
from pathlib import Path
from collections import Counter

# Paths
base_dir = Path(__file__).parent.resolve()
prompt_file = base_dir / "WildToolBench" / "wild-tool-bench" / "data" / "Wild-Tool-Bench.jsonl"
score_file = base_dir / "WildToolBench" / "wild-tool-bench" / "score" / "Qwen_Qwen3-14B" / "Wild-Tool-Bench_score.jsonl"

if not score_file.exists():
    print(f"Error: Score file not found at {score_file}")
    exit(1)

# Load prompts
prompts = {}
with open(prompt_file, "r") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            prompts[entry["id"]] = entry

# Analyze score results
action_name_failures = 0
arguments_failures = 0
both_failures = 0
success_turns = 0
total_turns = 0

failure_by_task = Counter()
failure_reasons = Counter()

examples_name_error = []
examples_arg_error = []

with open(score_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        entry_id = entry["id"]
        prompt = prompts.get(entry_id)
        task_types = prompt.get("english_task_types", []) if prompt else []
        
        results = entry.get("results", [])
        for i, turn in enumerate(results):
            total_turns += 1
            label = turn.get("label")
            action_name_label = turn.get("action_name_label")
            action_arguments_label = turn.get("action_arguments_label")
            task_type = task_types[i] if i < len(task_types) else "unknown"
            
            if label == "correct":
                success_turns += 1
            else:
                failure_by_task[task_type] += 1
                
                # Determine fail mode
                is_name_err = (action_name_label == "error")
                is_arg_err = (action_arguments_label == "error")
                
                if is_name_err and is_arg_err:
                    both_failures += 1
                elif is_name_err:
                    action_name_failures += 1
                    if len(examples_name_error) < 3:
                        examples_name_error.append((entry_id, i, turn))
                elif is_arg_err:
                    arguments_failures += 1
                    if len(examples_arg_error) < 3:
                        examples_arg_error.append((entry_id, i, turn))
                
                # Check error details in steps
                inf_log = turn.get("inference_log", {})
                for step_key, step_val in sorted(inf_log.items()):
                    if not step_key.startswith("step_"):
                        continue
                    output = step_val.get("inference_output", {})
                    if output.get("current_action_name_label") == "error":
                        reason = output.get("error_reason", "unknown name error")
                        failure_reasons[f"Name Error: {reason}"] += 1
                    if output.get("current_action_arguments_label") == "error":
                        check_res = output.get("current_action_arguments_check_result", [])
                        failure_reasons[f"Arg Error: {str(check_res)}"] += 1

print(f"Total turns analyzed: {total_turns}")
print(f"Success turns: {success_turns}")
print(f"Action Name only failures: {action_name_failures}")
print(f"Arguments only failures: {arguments_failures}")
print(f"Both failures: {both_failures}")
print()
print("Failures by Task Type:")
for task, count in failure_by_task.most_common():
    print(f"  {task}: {count}")

print()
print("Top failure reasons in steps:")
for reason, count in failure_reasons.most_common(15):
    print(f"  {count}x - {reason}")

print()
print("=== Examples of Action Name Error ===")
for entry_id, turn_idx, turn in examples_name_error:
    print("-" * 50)
    print(f"ID: {entry_id}, Turn: {turn_idx}")
    inf_log = turn.get("inference_log", {})
    for step_key, step_val in sorted(inf_log.items()):
        if not step_key.startswith("step_"):
            continue
        output = step_val.get("inference_output", {})
        print(f"  {step_key} Content: {repr(output.get('content'))}")
        print(f"  {step_key} Tool Calls: {output.get('tool_calls')}")
        print(f"  {step_key} Label: {output.get('current_action_name_label')}")
        print(f"  {step_key} Reason: {output.get('error_reason')}")

print()
print("=== Examples of Arguments Error ===")
for entry_id, turn_idx, turn in examples_arg_error:
    print("-" * 50)
    print(f"ID: {entry_id}, Turn: {turn_idx}")
    inf_log = turn.get("inference_log", {})
    for step_key, step_val in sorted(inf_log.items()):
        if not step_key.startswith("step_"):
            continue
        output = step_val.get("inference_output", {})
        print(f"  {step_key} Content: {repr(output.get('content'))}")
        print(f"  {step_key} Tool Calls: {output.get('tool_calls')}")
        print(f"  {step_key} Check Result: {output.get('current_action_arguments_check_result')}")
