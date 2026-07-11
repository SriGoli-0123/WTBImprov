import argparse
import json
import os

from tqdm import tqdm
from wtb.constant import PROJECT_ROOT, PROMPT_PATH, RESULT_PATH, SCORE_PATH
from wtb.utils import load_file, write_list_of_dicts_to_file, write_dicts_to_file
from wtb.checker_utils import ToolArgsChecker


def params_checker(result):
    tool_args_checker = ToolArgsChecker()
    action_arguments_label = "correct"
    inference_log = result["inference_log"]
    for key in inference_log.keys():
        if not key.startswith("step_"):
            continue

        # Current parameter check error, subsequent prediction results can be discarded.
        if action_arguments_label == "error":
            result["action_name_label"] = "correct"
            del inference_log[key]
            break

        step_data = inference_log[key]
        inference_input = step_data["inference_input"]
        inference_output = step_data["inference_output"]

        current_action_name_label = inference_output["current_action_name_label"]
        if current_action_name_label == "error":
            break

        inference_answer = step_data["inference_answer"]
        # At this stage, action_name is guaranteed to be correct, so the size of candidate_answer_function_list will always be 1.
        candidate_answer_function = inference_answer["candidate_0_answer_function_list"]
        assert "candidate_1_answer_function_list" not in inference_answer

        tools = inference_input["tools"]
        predict_tool_calls = inference_output["tool_calls"]
        answer_actions = candidate_answer_function["action"]

        if answer_actions[0]["name"] in ["prepare_to_answer", "ask_user_for_required_parameters"]:
            continue

        assert len(predict_tool_calls) == len(answer_actions)

        predict_tool_calls = sorted(predict_tool_calls, key=lambda x: (x["function"]["name"], x["function"]["arguments"]))
        predict_actions = [item["function"] for item in predict_tool_calls]
        inference_output["tool_calls"] = predict_tool_calls

        answer_actions = sorted(answer_actions, key=lambda x: (x["name"], x["arguments"]))
        candidate_answer_function["action"] = answer_actions

        current_action_arguments_label = "correct"
        arguments_check_result = []
        for predict_action, answer_action in zip(predict_actions, answer_actions):
            predict_name = predict_action["name"]
            predict_arguments = predict_action["arguments"]

            answer_name = answer_action["name"]
            answer_arguments = answer_action["arguments"]

            assert predict_name == answer_name

            check_result = tool_args_checker.check(tools, predict_name, predict_arguments, answer_arguments)
            arguments_check_result.append(check_result)
            if check_result != "correct":
                current_action_arguments_label = "error"
                action_arguments_label = "error"

        step_data["inference_output"]["current_action_arguments_label"] = current_action_arguments_label
        if current_action_arguments_label == "error":
            step_data["inference_output"]["current_action_arguments_check_result"] = arguments_check_result

    action_name_label = result["action_name_label"]
    if action_name_label == "correct":
        items = list(result.items())
        items.insert(1, ("action_arguments_label", action_arguments_label))
        result.clear()
        result.update(items)
        if action_arguments_label == "error" and result["is_optimal"] is True:
            result["is_optimal"] = False

    return action_name_label, action_arguments_label


def add_accuracy_field(variable_name, info_dict):
    """
    遍历字典，计算 accuracy 并将其插入到每个子字典的第一个位置。
    原地修改 info_dict 中的子对象。
    """
    for key, stats in info_dict.items():
        correct = stats.get("correct_count", 0)
        total = stats.get("total_count", 0)

        # 1. 计算 Accuracy (防止除以零)
        acc = correct / total if total > 0 else 0.0

        # 2. 构造有序的新内容
        # 先放入 accuracy，确保它是第一个 key
        new_content = {"accuracy": acc}
        # 再放入原有的数据
        new_content.update(stats)

        # 3. 原地修改子字典 (保持引用地址不变)
        stats.clear()
        stats.update(new_content)
    print(f"{variable_name}:")
    print(json.dumps(info_dict, ensure_ascii=False, indent=4))
    print("\n" + "=" * 100 + "\n")


def add_rate_field(variable_name, info_dict):
    """
    遍历字典，计算 accuracy 并将其插入到每个子字典的第一个位置。
    原地修改 info_dict 中的子对象。
    """
    for key, stats in info_dict.items():
        correct = stats.get("complete_step", 0)
        total = stats.get("total_step", 0)

        # 1. 计算 Accuracy (防止除以零)
        acc = correct / total if total > 0 else 0.0

        # 2. 构造有序的新内容
        # 先放入 accuracy，确保它是第一个 key
        new_content = {"rate": acc}
        # 再放入原有的数据
        new_content.update(stats)

        # 3. 原地修改子字典 (保持引用地址不变)
        stats.clear()
        stats.update(new_content)
    print(f"{variable_name}:")
    print(json.dumps(info_dict, ensure_ascii=False, indent=4) + "\n")


def calc_accuracy(model_name, all_test_entries, score_results):
    total_info = {
        "task": {"correct_count": 0, "total_count": 0},
        "session": {"correct_count": 0, "total_count": 0}
    }
    task_type_info = {
        "Single-Tool": {"correct_count": 0, "total_count": 0},
        "Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Parallel Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Sequential Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Mixed Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Clarify": {"correct_count": 0, "total_count": 0},
        "Chat": {"correct_count": 0, "total_count": 0}
    }
    layer_info = {
        "0": {"correct_count": 0, "total_count": 0},
        "1": {"correct_count": 0, "total_count": 0},
        "2": {"correct_count": 0, "total_count": 0},
        "3": {"correct_count": 0, "total_count": 0}
    }
    turn_subtype_info = {
        "First Turn": {"correct_count": 0, "total_count": 0},
        "Subsequent Turn": {"correct_count": 0, "total_count": 0},
        "Coreferential Reference": {"correct_count": 0, "total_count": 0},
        "Partial Information": {"correct_count": 0, "total_count": 0},
        "Long-Range Dependency": {"correct_count": 0, "total_count": 0}
    }
    progress_info = {
        "Total": {"complete_step": 0, "total_step": 0},
        "Sequential Multi-Tool": {"complete_step": 0, "total_step": 0},
        "Mixed Multi-Tool": {"complete_step": 0, "total_step": 0}
    }
    optimal_info = {
        "Total": {"correct_count": 0, "total_count": 0},
        "Parallel Multi-Tool": {"correct_count": 0, "total_count": 0},
        "Mixed Multi-Tool": {"correct_count": 0, "total_count": 0}
    }
    for score_result in score_results:
        id_ = score_result["id"]
        results = score_result["results"]
        parts = id_.rsplit("_", 1)
        index = int(parts[1])
        test_entry = all_test_entries[index]
        english_task_types = test_entry["english_task_types"]
        english_turn_subtypes = test_entry["english_turn_subtypes"]
        answer_list = test_entry["answer_list"]

        total_info["session"]["total_count"] += 1
        session_correct = True
        for i, result in enumerate(results):
            label = result["label"]
            is_optimal = result["is_optimal"]
            task_type = english_task_types[i]
            if i == 0:
                turn_subtype = "First Turn"
            else:
                turn_subtype = english_turn_subtypes[i - 1]
                turn_subtype_info["Subsequent Turn"]["total_count"] += 1

            total_info["task"]["total_count"] += 1
            if "Multi-Tool" in task_type:
                task_type_info["Multi-Tool"]["total_count"] += 1
            task_type_info[task_type]["total_count"] += 1
            layer_info[str(i)]["total_count"] += 1
            turn_subtype_info[turn_subtype]["total_count"] += 1

            if label == "correct":
                total_info["task"]["correct_count"] += 1
                if "Multi-Tool" in task_type:
                    task_type_info["Multi-Tool"]["correct_count"] += 1
                task_type_info[task_type]["correct_count"] += 1
                layer_info[str(i)]["correct_count"] += 1
                turn_subtype_info[turn_subtype]["correct_count"] += 1
                if i > 0:
                    turn_subtype_info["Subsequent Turn"]["correct_count"] += 1
            else:
                session_correct = False

            if task_type in ["Parallel Multi-Tool", "Mixed Multi-Tool"]:
                optimal_info["Total"]["total_count"] += 1
                optimal_info[task_type]["total_count"] += 1
                if is_optimal:
                    optimal_info["Total"]["correct_count"] += 1
                    optimal_info[task_type]["correct_count"] += 1

            if task_type in ["Sequential Multi-Tool", "Mixed Multi-Tool"]:
                inference_log = result["inference_log"]
                complete_step = len([k for k in inference_log.keys() if k.startswith("step")])
                answer = answer_list[i]
                total_step = len(answer)
                progress_info["Total"]["total_step"] += total_step
                progress_info["Total"]["complete_step"] += complete_step
                progress_info[task_type]["total_step"] += total_step
                progress_info[task_type]["complete_step"] += complete_step

        if session_correct:
            total_info["session"]["correct_count"] += 1

    add_accuracy_field("total_info", total_info)
    add_accuracy_field("task_type_info", task_type_info)
    add_accuracy_field("layer_info", layer_info)
    add_accuracy_field("turn_subtype_info", turn_subtype_info)
    add_accuracy_field("optimal_info", optimal_info)
    add_rate_field("progress_info", progress_info)
    metric_info = {
        "model_name": model_name,
        "total_info": total_info,
        "task_type_info": task_type_info,
        "layer_info": layer_info,
        "turn_subtype_info": turn_subtype_info,
        "optimal_info": optimal_info,
        "progress_info": progress_info
    }
    return metric_info


def runner(model_names, result_dir, score_dir):
    # Get a list of all entries in the folder
    entries = result_dir.iterdir()

    # Filter out the subdirectories
    subdirs = [entry for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in tqdm(subdirs, desc="Number of models evaluated"):
        model_name = subdir.relative_to(result_dir).name

        if model_names is not None and model_name not in model_names:
            continue

        print(f"Model: {model_name}")

        score_results = []
        model_result_jsonl = subdir / os.path.basename(PROMPT_PATH).replace(".jsonl", "_result.jsonl")
        model_results = load_file(model_result_jsonl, sort_by_id=True)
        for model_result in model_results:
            id_ = model_result["id"]
            results = model_result["result"]
            if isinstance(results, str):
                print(f"⚠️ Warning: Skipping errored result for {id_}: {results}")
                continue
            for result in results:
                action_name_label, action_arguments_label = params_checker(result)
                if action_name_label == "error" or action_arguments_label == "error":
                    label = "error"
                else:
                    label = "correct"
                items = list(result.items())
                items.insert(0, ("label", label))
                result.clear()
                result.update(items)

            score_results.append({"id": id_, "results": results})

        output_file_name = os.path.basename(PROMPT_PATH).replace(".jsonl", "_score.jsonl")
        output_file_dir = score_dir / model_name
        write_list_of_dicts_to_file(output_file_name, score_results, output_file_dir)

        all_test_entries = load_file(PROMPT_PATH)
        metric_info = calc_accuracy(model_name, all_test_entries, score_results)
        metric_file_name = os.path.basename(PROMPT_PATH).replace(".jsonl", "_metric.json")
        write_dicts_to_file(metric_file_name, metric_info, output_file_dir)


def main(model, result_dir, score_dir):
    if result_dir is None:
        result_dir = RESULT_PATH
    else:
        result_dir = (PROJECT_ROOT / result_dir).resolve()

    if score_dir is None:
        score_dir = SCORE_PATH
    else:
        score_dir = (PROJECT_ROOT / score_dir).resolve()

    runner(model, result_dir, score_dir)


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="deepseek-chat", nargs="+")

    # Parameters for the model that you want to eval.
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--score-dir", default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    main(
        args.model,
        args.result_dir,
        args.score_dir,
    )
