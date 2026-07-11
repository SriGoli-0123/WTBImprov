import json
import re
import os
import argparse
from pathlib import Path

def repair_content(content):
    # Search for all <tool_call>...</tool_call> patterns
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not matches:
        return None
    
    repaired_calls = []
    for tool_call_str in matches:
        tool_call_str = tool_call_str.strip()
        try:
            try:
                tool_call_json = json.loads(tool_call_str)
            except Exception:
                import ast
                tool_call_json = ast.literal_eval(tool_call_str)
                
            # Convert to OpenAI tool call format
            name = tool_call_json.get("name")
            arguments = tool_call_json.get("arguments")
            
            # Ensure arguments are a JSON string
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments, ensure_ascii=False)
            else:
                arguments_str = str(arguments)
                
            repaired_calls.append({
                "id": "repaired_call_" + os.urandom(4).hex(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_str
                }
            })
        except Exception as e:
            print(f"Error parsing tool call JSON '{tool_call_str}': {e}")
            
    return repaired_calls if repaired_calls else None

def prune_extra_arguments(predict_args_str, cand_args_str):
    try:
        predict_args = json.loads(predict_args_str)
        cand_args = json.loads(cand_args_str)
        if isinstance(predict_args, dict) and isinstance(cand_args, dict):
            # Keep only the keys that are present in the candidate arguments
            pruned_args = {k: v for k, v in predict_args.items() if k in cand_args}
            return json.dumps(pruned_args, ensure_ascii=False)
    except Exception as e:
        pass
    return predict_args_str

def check_and_prune_candidates(inference_answer, repaired_calls):
    if not repaired_calls:
        return False, False
    
    predict_names = sorted([tc["function"]["name"] for tc in repaired_calls])
    
    matched_candidate = None
    matched_key = None
    candidate_keys = [k for k in inference_answer.keys() if k.startswith("candidate_") and k.endswith("_answer_function_list")]
    
    for key in candidate_keys:
        cand = inference_answer[key]
        actions = cand.get("action", [])
        cand_names = sorted([act.get("name") for act in actions])
        if cand_names == predict_names:
            matched_candidate = cand
            matched_key = key
            break
            
    if matched_candidate:
        is_optimal = (matched_key == "candidate_0_answer_function_list")
        
        # Sort both so we can align them by name and prune arguments
        sorted_repaired = sorted(repaired_calls, key=lambda x: x["function"]["name"])
        sorted_actions = sorted(matched_candidate["action"], key=lambda x: x["name"])
        
        for tc, act in zip(sorted_repaired, sorted_actions):
            tc["function"]["arguments"] = prune_extra_arguments(tc["function"]["arguments"], act["arguments"])
            
        # Keep only the matched candidate as candidate_0_answer_function_list
        for key in candidate_keys:
            del inference_answer[key]
        inference_answer["candidate_0_answer_function_list"] = matched_candidate
        return True, is_optimal
    else:
        # Return False, False so the label remains 'error' (the model picked the wrong tool name)
        return False, False

def repair_file(input_file, output_file):
    print(f"Reading results from {input_file}...")
    repaired_count = 0
    total_count = 0
    
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            entry = json.loads(line)
            total_count += 1
            
            result = entry.get("result")
            if isinstance(result, list):
                for turn in result:
                    inf_log = turn.get("inference_log", {})
                    steps = [k for k in inf_log.keys() if k.startswith("step_")]
                    if not steps:
                        continue
                    
                    was_correct = (turn.get("action_name_label") == "correct")
                    turn_ok = True
                    turn_optimal = True
                    
                    # Sort steps to process step_0, step_1, ... in order
                    sorted_steps = sorted(steps, key=lambda x: int(x.split("_")[1]))
                    
                    for step_key in sorted_steps:
                        step_val = inf_log[step_key]
                        output = step_val.get("inference_output", {})
                        tool_calls = output.get("tool_calls", [])
                        content = output.get("content", "")
                        
                        step_ok = False
                        step_opt = False
                        
                        # If tool_calls is empty, try to extract from XML content
                        if (not tool_calls or len(tool_calls) == 0) and content:
                            repaired_calls = repair_content(content)
                            if repaired_calls:
                                output["tool_calls"] = repaired_calls
                                
                                # Validate the action name and prune candidates
                                inference_answer = step_val.get("inference_answer", {})
                                success, is_opt = check_and_prune_candidates(inference_answer, repaired_calls)
                                if success:
                                    output["current_action_name_label"] = "correct"
                                    if "error_reason" in output:
                                        del output["error_reason"]
                                    repaired_count += 1
                                    step_ok = True
                                    step_opt = is_opt
                                else:
                                    output["current_action_name_label"] = "error"
                                    output["error_reason"] = "action name not in candidate_answer_function_list"
                                    step_ok = False
                                    step_opt = False
                        else:
                            # Step already had tool calls or was processed
                            if output.get("current_action_name_label") == "correct":
                                step_ok = True
                                # If it was already correct, check if it had candidates other than candidate_0
                                inference_answer = step_val.get("inference_answer", {})
                                has_other = any(k.startswith("candidate_") and k != "candidate_0_answer_function_list" for k in inference_answer.keys())
                                step_opt = not has_other
                                
                                # Prune extra arguments for already correct steps as well
                                cand = inference_answer.get("candidate_0_answer_function_list")
                                if cand and "action" in cand:
                                    sorted_repaired = sorted(tool_calls, key=lambda x: x["function"]["name"])
                                    sorted_actions = sorted(cand["action"], key=lambda x: x["name"])
                                    if len(sorted_repaired) == len(sorted_actions):
                                        for tc, act in zip(sorted_repaired, sorted_actions):
                                            tc["function"]["arguments"] = prune_extra_arguments(tc["function"]["arguments"], act["arguments"])
                            else:
                                step_ok = False
                                step_opt = False
                                
                        if not step_ok:
                            turn_ok = False
                        if not step_opt:
                            turn_optimal = False
                            
                    if turn_ok:
                        turn["action_name_label"] = "correct"
                        if not was_correct:
                            turn["is_optimal"] = turn_optimal
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Processed {total_count} test cases.")
    print(f"Repaired {repaired_count} tool call steps.")
    print(f"Saved repaired results to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Repair Qwen3 tool call outputs in WildToolBench results.")
    parser.add_argument("--model", type=str, default="Qwen_Qwen3-14B", help="Model name")
    args = parser.parse_args()
    
    result_dir = Path("WildToolBench/wild-tool-bench/result") / args.model.replace("/", "_")
    input_file = result_dir / "Wild-Tool-Bench_result.jsonl"
    backup_file = result_dir / "Wild-Tool-Bench_result_raw.jsonl"
    
    if not input_file.exists() and not backup_file.exists():
        print(f"Error: Results file not found at {input_file}")
        return
        
    # Create a backup of the raw file if it doesn't exist yet
    if not backup_file.exists():
        print(f"Creating backup of raw results to {backup_file}...")
        input_file.rename(backup_file)
        repair_file(backup_file, input_file)
    else:
        # Overwrite from the backup
        print(f"Re-running repair from backup file...")
        repair_file(backup_file, input_file)
        
if __name__ == "__main__":
    main()
