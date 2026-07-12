import json
import os

from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from overrides import final

from wtb.tool_call_graph import ToolCallGraph
from wtb.utils import sort_key, load_file, generate_random_string
from wtb.constant import PROMPT_PATH


# Action-mode triage protocol. Benchmark-agnostic: it encodes general tool-use
# discipline (decide call/ask/answer first, argument minimalism, verbatim value
# copying) rather than rules fitted to specific test cases.
SYSTEM_PROMPT_TEMPLATE = """Current Date: {env_info}

You are a precise tool-calling assistant. On every user turn, first silently classify the turn into exactly one of three modes, then respond in that mode:

[A] CALL TOOLS - The request requires an action or lookup that the available tools perform, and the value of every required parameter is available: stated by the user (in this turn or an earlier one), present in an earlier tool result, or exactly computable (e.g., resolve "tomorrow" or "this weekend" using Current Date). Respond with the tool call(s) only.

[B] ASK THE USER - A tool is needed, but some required parameter value is missing or ambiguous (e.g., the user says "one of them" without saying which one, or an ID/date/location was never given). Reply with one short plain-text question requesting exactly the missing detail(s). Never guess, never fabricate or use placeholder values, and never silently pick one of several options for the user.

[C] ANSWER DIRECTLY - The request can be fully answered from the conversation so far (including earlier tool results), or it is small talk / general knowledge that no tool serves. Reply in plain text. Do not re-call a tool whose result is already in the conversation.

Rules when calling tools:
1. Emit ALL independent tool calls of this turn together in one response (parallel calls). Only defer a call when it needs the output of another call first; then wait for that result before making it.
2. Copy argument values exactly from the conversation or tool results (IDs, codes, emails, names, URLs - verbatim). Resolve relative dates/times against Current Date.
3. Include a parameter only if it is required by the schema or the user explicitly provided/requested its value. Do not add optional parameters on your own initiative (no default true/false/0/empty values). For update-style operations, pass only the identifier(s) plus the fields the user wants changed.
4. Follow the schema exactly: parameter names, types, and enum values must match it.
5. Never describe or announce a tool call in text - either make the call [A], ask [B], or answer [C]."""


class BaseHandler:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.model_messages = []
        self.consecutive_tool_messages = True
        # Self-consistency (consensus decoding) config:
        #   WTB_SC_N: total candidates sampled per step (1 disables voting)
        #   WTB_SC_TEMP: temperature for the diversity samples (anchor keeps run temperature)
        self.sc_n = int(os.getenv("WTB_SC_N", "5"))
        self.sc_temperature = float(os.getenv("WTB_SC_TEMP", "0.8"))

    def _clean_tool_call_arguments(self, tool_name, arguments_dict, tools):
        # Find the tool schema
        schema = None
        for t in tools:
            if t.get("function", {}).get("name") == tool_name:
                schema = t["function"]
                break
        if not schema:
            return arguments_dict
            
        parameters = schema.get("parameters", {})
        if not parameters or parameters.get("type") != "object":
            return arguments_dict
            
        properties = parameters.get("properties", {})
        
        def cast_value(val, prop_schema):
            prop_type = prop_schema.get("type")
            if prop_type == "integer":
                try:
                    return int(float(val))
                except:
                    pass
            elif prop_type == "number":
                try:
                    return float(val)
                except:
                    pass
            elif prop_type == "boolean":
                if isinstance(val, str):
                    if val.lower() in ["true", "1"]:
                        return True
                    if val.lower() in ["false", "0"]:
                        return False
            elif prop_type == "array" and isinstance(val, str):
                try:
                    loaded = json.loads(val)
                    if isinstance(loaded, list):
                        return loaded
                except:
                    pass
            elif prop_type == "object" and isinstance(val, dict):
                # Recursive cleaning for nested objects
                nested_properties = prop_schema.get("properties", {})
                cleaned_nested = {}
                for k, v in val.items():
                    if k in nested_properties:
                        cleaned_nested[k] = cast_value(v, nested_properties[k])
                return cleaned_nested
            # Enum check
            if "enum" in prop_schema:
                enum_options = prop_schema["enum"]
                if isinstance(val, str):
                    # Case-insensitive match
                    for opt in enum_options:
                        if isinstance(opt, str) and opt.lower() == val.lower():
                            return opt
            return val

        cleaned_args = {}
        for k, v in arguments_dict.items():
            if k in properties:
                cleaned_args[k] = cast_value(v, properties[k])
        return cleaned_args

    @staticmethod
    def _canon(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _response_signature(model_response_data):
        tool_calls = model_response_data.get("tool_calls") or []
        if tool_calls:
            names = tuple(sorted(tc.get("function", {}).get("name", "") for tc in tool_calls))
            return ("tools",) + names
        content = model_response_data.get("content")
        if content:
            return ("text",)
        return ("empty",)

    def _consensus_generate(self, inference_data):
        '''
        Self-consistency (consensus) decoding for one step.
        Draws sc_n candidates (1 anchor at the run temperature + sc_n-1 diversity
        samples), then votes hierarchically: response mode -> tool-name multiset
        -> per-argument key/value majority. Returns a single model_response_data
        dict shaped like _parse_api_response output, plus aggregated token counts,
        latency, and a "consensus_log" entry describing the vote.
        '''
        api_response, latency = self._request_tool_call(inference_data)
        anchor = self._parse_api_response(api_response)
        anchor["latency"] = latency

        if self.sc_n <= 1 or not hasattr(self, "_request_candidates"):
            return anchor

        candidates = [anchor]
        try:
            candidates.extend(
                self._request_candidates(inference_data, self.sc_n - 1, self.sc_temperature)
            )
        except Exception as e:
            print(f"Consensus sampling failed, using anchor only: {e}", flush=True)
            return anchor

        signatures = [self._response_signature(c) for c in candidates]
        vote_counts = Counter(s for s in signatures if s != ("empty",))
        consensus_log = {
            "candidate_signatures": [list(s) for s in signatures],
        }

        total_input = sum(c.get("input_token") or 0 for c in candidates)
        total_output = sum(c.get("output_token") or 0 for c in candidates)
        total_latency = sum(c.get("latency") or 0 for c in candidates)

        if not vote_counts:
            chosen = anchor
        else:
            best_count = max(vote_counts.values())
            winners = {s for s, v in vote_counts.items() if v == best_count}
            if signatures[0] in winners:
                # Tie or win including the anchor: trust the low-temperature sample.
                winning_signature = signatures[0]
            else:
                winning_signature = next(s for s in signatures if s in winners)
            cluster = [c for c, s in zip(candidates, signatures) if s == winning_signature]
            consensus_log["winning_signature"] = list(winning_signature)
            consensus_log["cluster_size"] = len(cluster)

            chosen = dict(cluster[0])
            if winning_signature[0] == "tools":
                chosen["tool_calls"] = self._merge_tool_calls(cluster, inference_data["tools"])

        chosen = dict(chosen)
        chosen["input_token"] = total_input
        chosen["output_token"] = total_output
        chosen["latency"] = total_latency
        chosen["consensus_log"] = consensus_log
        return chosen

    def _merge_tool_calls(self, cluster, tools):
        '''
        Argument-level majority vote across a cluster of candidates that agree on
        the tool-name multiset. A parameter key is kept only if a strict majority
        of candidates include it (or the schema marks it required), which prunes
        hallucinated optional parameters; each kept key takes its majority value.
        '''
        required_by_tool = {}
        for t in tools:
            func = t.get("function", {})
            required_by_tool[func.get("name")] = set(func.get("parameters", {}).get("required", []))

        def parsed_calls(candidate):
            calls = []
            for tc in candidate.get("tool_calls") or []:
                function = tc.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = None
                if not isinstance(arguments, dict):
                    arguments = None
                calls.append((name, arguments, tc))
            # Sort by (name, canonical args) so same-name calls align consistently
            # across candidates.
            calls.sort(key=lambda x: (x[0], self._canon(x[1]) if x[1] is not None else ""))
            return calls

        all_calls = [parsed_calls(c) for c in cluster]
        representative_calls = all_calls[0]

        merged = []
        for position, (name, rep_args, rep_tc) in enumerate(representative_calls):
            variants = []
            for candidate_calls in all_calls:
                v_name, v_args, _ = candidate_calls[position]
                if v_name == name and v_args is not None:
                    variants.append(v_args)

            if rep_args is None or not variants:
                merged.append(rep_tc)
                continue

            n_variants = len(variants)
            required_params = required_by_tool.get(name, set())
            key_counts = Counter(k for v in variants for k in v)

            final_args = {}
            for key, count in key_counts.items():
                # Strict majority keeps the key; ties are dropped (minimalism bias
                # against hallucinated optional parameters). Required parameters
                # are always kept so the vote can never break schema validity.
                if count * 2 <= n_variants and key not in required_params:
                    continue
                value_counts = Counter(self._canon(v[key]) for v in variants if key in v)
                best = max(value_counts.values())
                top_values = [val for val, c in value_counts.items() if c == best]
                rep_value = self._canon(rep_args[key]) if key in rep_args else None
                chosen_value = rep_value if rep_value in top_values else top_values[0]
                final_args[key] = json.loads(chosen_value)

            new_tc = deepcopy(rep_tc)
            new_tc["function"]["arguments"] = json.dumps(final_args, ensure_ascii=False)
            merged.append(new_tc)

        return merged

    def _request_tool_call(self, inference_data):
        raise NotImplementedError

    def _parse_api_response(self, api_response):
        raise NotImplementedError

    def convert_to_tool(self, tools):
        tools = json.dumps(tools, ensure_ascii=False).replace('"type": "float"', '"type": "number"')
        tools = json.loads(tools)
        new_tools = []
        if "claude" in self.model_name:
            for tool in tools:
                tool["inputSchema"] = tool["parameters"]
                del tool["parameters"]
                new_tools.append(tool)
            tools = new_tools
        return tools

    def _add_action_observation(self, task, answer_list, consecutive_tool_messages):
        tool_call_graph = ToolCallGraph(answer_list)
        tool_call_graph.add_node_list()
        tool_call_graph.generate_all_path()
        optimal_path = tool_call_graph.optimal_path_list[0]

        current_messages = [{"role": "user", "content": task}]
        for idx_action_list in optimal_path:
            format_action_list = []
            observation_list = []
            for idx in idx_action_list:
                answer = answer_list[idx]
                action = answer["action"]
                action_name = action["name"]
                action_arguments = action["arguments"]
                observation = answer["observation"]
                if action_name == "ask_user_for_required_parameters":
                    assert len(idx_action_list) == 1
                    user_input = answer["user_input"]
                    current_messages.extend([
                        {
                            "role": "assistant",
                            "content": observation
                        },
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ])

                elif action_name == "prepare_to_answer":
                    assert len(idx_action_list) == 1
                    current_messages.extend([
                        {
                            "role": "assistant",
                            "content": observation
                        }
                    ])

                else:
                    format_action_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": action_name,
                                "arguments": action_arguments
                            }
                        }
                    )
                    observation_list.append(observation)

            if len(format_action_list) > 0:
                current_messages.extend([
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": format_action_list
                    }
                ])
                if consecutive_tool_messages:
                    for observation in observation_list:
                        current_messages.extend([
                            {
                                "role": "tool",
                                "content": observation
                            }
                        ])
                else:
                    current_messages.extend([
                        {
                            "role": "tool",
                            "content": json.dumps(observation_list, ensure_ascii=False)
                        }
                    ])

        return current_messages

    def _convert_to_tool_calls(self, messages):
        tool_call_id_list = []
        new_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            tool_calls = message.get("tool_calls", None)
            if role == "assistant":
                if tool_calls:
                    new_tool_calls = []
                    for tool_call in tool_calls:
                        arguments = tool_call["function"]["arguments"]
                        if isinstance(arguments, dict):
                            tool_call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
                        if "id" not in tool_call:
                            tool_call_id = "toolu_bdrk_" + generate_random_string(24)
                            tool_call["id"] = tool_call_id
                            tool_call_id_list.append(tool_call_id)
                        new_tool_calls.append(tool_call)
                    message["tool_calls"] = new_tool_calls
            elif role == "tool":
                tool_call_id = tool_call_id_list[0]
                tool_call_id_list.pop(0)
                message["tool_call_id"] = tool_call_id
                message["content"] = json.dumps(content, ensure_ascii=False)
            new_messages.append(message)

        return new_messages

    def _pre_messages_processing(self, env_info, current_task, history_tasks, history_answer_lists, consecutive_tool_messages=True):
        messages = [{"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(env_info=env_info)}]
        for history_task, history_answer_list in zip(history_tasks, history_answer_lists):
            history_messages = self._add_action_observation(history_task, history_answer_list, consecutive_tool_messages)
            messages.extend(history_messages)
        messages.append({"role": "user", "content": current_task})
        messages = self._convert_to_tool_calls(messages)

        return messages

    def inference(self, test_entry: dict):
        return self.inference_multi_turn(test_entry)

    @final
    def inference_multi_turn(self, test_entry: dict):
        test_entry_id = test_entry["id"]
        env_info = test_entry["english_env_info"]
        tools = test_entry["english_tools"]
        tasks = test_entry["english_tasks"]
        answer_lists = test_entry["english_answer_list"]

        tools = self.convert_to_tool(tools)

        all_task_result_data = []
        for task_idx, (current_task, answer_list) in enumerate(zip(tasks, answer_lists)):
            history_tasks = tasks[:task_idx]
            history_answer_lists = answer_lists[:task_idx]
            messages = self._pre_messages_processing(env_info, current_task, history_tasks, history_answer_lists)

            inference_data = {"test_entry_id": test_entry_id, "task_idx": task_idx, "tools": tools, "messages": messages, "answer_list": answer_list}
            result_data = self.inference_and_eval_multi_step(inference_data)
            all_task_result_data.append(result_data)

        return all_task_result_data

    def run_with_timeout(self, func, timeout, *args, **kwargs):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                raise TimeoutError(f"Function '{func.__name__}' exceeded timeout of {timeout} seconds")

    @final
    def inference_and_eval_multi_step(self, inference_data):
        '''
        Only the action is evaluated here
        If the action is incorrect, the process will be terminated early to improve evaluation efficiency.
        The correctness of the parameters will be evaluated later in the eval_checker.
        '''
        test_entry_id = inference_data["test_entry_id"]
        task_idx = inference_data["task_idx"]
        tools = inference_data["tools"]
        messages = inference_data["messages"]
        answer_list = inference_data["answer_list"]
        tool_call_graph = ToolCallGraph(answer_list)
        tool_call_graph.add_node_list()
        tool_call_graph.generate_all_path()
        # try:
        #     self.run_with_timeout(tool_call_graph.add_node_list, 60)
        #     self.run_with_timeout(tool_call_graph.generate_all_path, 60)
        # except Exception as e:
        #     print(f"error: {e}", flush=True)
        #     return "graph generate timeout."

        # print("generate", len(answer_list), flush=True)

        step = 0
        action_name_label = "error"
        predict_result = []
        # inference_log = [{"task_idx": task_idx, "begin_of_current_task": messages[-1]}]
        inference_log = {
            "task_idx": task_idx,
            "begin_of_current_task": messages[-1]
        }
        answer_result = []
        latency = []
        input_token_count = []
        output_token_count = []
        while True:
            print("-" * 100, flush=True)
            print(
                f"ID: {test_entry_id.replace('wild_tool_bench_', '')}, Task: {task_idx}, Step: {step}", flush=True
            )
            # print(f"Output：", flush=True)
            # for message in messages:
            #     print(json.dumps(message, ensure_ascii=False, indent=4) + "\n", flush=True)
            model_response_data = self._consensus_generate(inference_data)
            query_latency = model_response_data.get("latency", 0)
            consensus_log = model_response_data.pop("consensus_log", None)
            reasoning_content = model_response_data["reasoning_content"]
            content = model_response_data["content"]
            tool_calls = model_response_data["tool_calls"]
            if tool_calls is not None:
                # Rule-Based Schema Cleaner
                cleaned_tool_calls = []
                for tc in tool_calls:
                    try:
                        tc_name = tc["function"]["name"]
                        tc_args_str = tc["function"]["arguments"]
                        if isinstance(tc_args_str, str):
                            tc_args = json.loads(tc_args_str)
                            cleaned_args = self._clean_tool_call_arguments(tc_name, tc_args, tools)
                            tc["function"]["arguments"] = json.dumps(cleaned_args, ensure_ascii=False)
                        elif isinstance(tc_args_str, dict):
                            cleaned_args = self._clean_tool_call_arguments(tc_name, tc_args_str, tools)
                            tc["function"]["arguments"] = cleaned_args
                    except Exception as e:
                        print(f"Cleaner error: {e}", flush=True)
                    cleaned_tool_calls.append(tc)
                tool_calls = cleaned_tool_calls
                model_response_data["tool_calls"] = tool_calls
            input_token = model_response_data["input_token"]
            output_token = model_response_data["output_token"]
            latency.append(query_latency)
            input_token_count.append(input_token)
            output_token_count.append(output_token)

            # print(f"Output：", flush=True)
            # print(f"reasoning_content: \n{reasoning_content}\n", flush=True)
            # print(f"content: \n{content}\n", flush=True)
            # print(f"tool_calls: \n{json.dumps(tool_calls, ensure_ascii=False, indent=4)}\n", flush=True)

            inference_log[f"step_{step}"] = {
                "inference_input": {
                    "messages": deepcopy(messages),
                    "tools": tools
                },
                "inference_output": {
                    "reasoning_content": reasoning_content,
                    "content": content,
                    "tool_calls": tool_calls
                }
            }
            if consensus_log is not None:
                inference_log[f"step_{step}"]["consensus"] = consensus_log

            if tool_calls is None or len(tool_calls) == 0:
                if content is None or content == "":
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"tool_calls and content are None or empty"
                        }
                    )
                    break

                else:
                    current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                    current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                    current_step_function_observation_list = tool_call_graph.step_to_function_observation_list[step]
                    current_step_user_input_list = tool_call_graph.step_to_user_input_list[step]
                    for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(
                            zip(
                                current_step_function_name_list,
                                current_step_function_arguments_list,
                                current_step_function_observation_list,
                                current_step_user_input_list
                            )
                    ):
                        if "ask_user_for_required_parameters" in answer_function_name_list:
                            messages.append(
                                {"role": "assistant", "content": content}
                            )

                            assert len(answer_function_observation_list) == 1
                            function_observation = answer_function_observation_list[0]

                            assert len(answer_user_input_list) == 1
                            user_input = answer_user_input_list[0]
                            messages.append(
                                {"role": "user", "content": user_input}
                            )

                            answer_function_list = {"action": [], "observation": function_observation, "user_input": user_input}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"

                            break
                        elif "prepare_to_answer" in answer_function_name_list:
                            assert len(answer_function_observation_list) == 1
                            function_observation = answer_function_observation_list[0]

                            messages.append(
                                {"role": "assistant", "content": content}
                            )
                            action_name_label = "correct"

                            answer_function_list = {"action": [], "observation": function_observation}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"

                            break
                    else:
                        action_name_label = "error"
                        # Provide candidate correct answers
                        for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(
                                zip(
                                    current_step_function_name_list,
                                    current_step_function_arguments_list,
                                    current_step_function_observation_list,
                                    current_step_user_input_list
                                )
                        ):
                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments, answer_function_observation, answer_user_input in zip(
                                    answer_function_name_list,
                                    answer_function_arguments_list,
                                    answer_function_observation_list,
                                    answer_user_input_list
                            ):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                                if answer_function_name == "ask_user_for_required_parameters":
                                    answer_function_list["observation"] = answer_function_observation
                                    answer_function_list["user_input"] = answer_user_input
                                elif answer_function_name == "prepare_to_answer":
                                    answer_function_list["observation"] = answer_function_observation

                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_{i}_answer_function_list"] = answer_function_list

                        inference_log[f"step_{step}"]["inference_output"].update(
                            {
                                "current_action_name_label": "error",
                                "error_reason": f"action name not in candidate_answer_function_list"
                            }
                        )
                        break

            else:
                tool_calls_len = len(tool_calls)
                predict_function_name_list = []
                predict_function_arguments_list = []
                predict_function_id_list = []
                try:
                    for tool_call in tool_calls:
                        function_id = tool_call["id"]
                        function = tool_call["function"]
                        function_name = function["name"]
                        function_arguments = function["arguments"]
                        predict_function_id_list.append(function_id)
                        predict_function_name_list.append(function_name)
                        predict_function_arguments_list.append(function_arguments)
                except Exception as e:
                    print(f"{json.dumps(tool_calls, ensure_ascii=False)} parse failed.", flush=True)
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"parse tool_calls failed, error: {str(e)}"
                        }
                    )
                    break

                idx_predict_function_name_list = list(enumerate(predict_function_name_list))
                sorted_idx_predict_function_name_list = sorted(idx_predict_function_name_list, key=lambda x: x[1])
                sorted_indices = [idx for idx, predict_function_name in sorted_idx_predict_function_name_list]
                sorted_predict_function_name_list = [predict_function_name_list[i] for i in sorted_indices]
                sorted_predict_function_arguments_list = [predict_function_arguments_list[i] for i in sorted_indices]
                predict_function_name_list = sorted_predict_function_name_list
                predict_function_arguments_list = sorted_predict_function_arguments_list

                idx_list = tool_call_graph.step_to_idx_list.get(step, None)
                if idx_list is None:
                    action_name_label = "error"
                    inference_log[f"step_{step}"]["inference_output"].update(
                        {
                            "current_action_name_label": "error",
                            "error_reason": f"current step: {step}, idx_list is None"
                        }
                    )
                    break
                else:
                    current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                    current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                    current_step_function_observation_list = tool_call_graph.step_to_function_observation_list[step]
                    current_step_user_input_list = tool_call_graph.step_to_user_input_list[step]
                    for i, (answer_function_name_list, answer_function_arguments_list) in enumerate(zip(current_step_function_name_list, current_step_function_arguments_list)):
                        if predict_function_name_list != answer_function_name_list:
                            continue
                        else:
                            if reasoning_content is not None:
                                messages.append(
                                    {"role": "assistant", "reasoning_content": reasoning_content, "content": content, "tool_calls": tool_calls}
                                )
                            else:
                                messages.append(
                                    {"role": "assistant", "content": content, "tool_calls": tool_calls}
                                )

                            function_observation_list = tool_call_graph.step_to_function_observation_list[step][i]
                            if self.consecutive_tool_messages:
                                # Supports consecutive multiple tool messages
                                for j, function_observation in enumerate(function_observation_list):
                                    if not isinstance(function_observation, str):
                                        function_observation = json.dumps(function_observation, ensure_ascii=False)
                                    function_id = predict_function_id_list[j]
                                    messages.append(
                                        {"role": "tool", "content": function_observation, "tool_call_id": function_id}
                                    )
                            else:
                                # Does not support consecutive multiple tool messages
                                function_observation_list = json.dumps(function_observation_list, ensure_ascii=False)
                                function_id = predict_function_id_list[0]
                                messages.append(
                                    {"role": "tool", "content": function_observation_list, "tool_call_id": function_id}
                                )

                            # Pruning
                            idx_list = tool_call_graph.step_to_idx_list[step][i]
                            tool_call_graph.update_updating_all_path_list(step, idx_list)
                            tool_call_graph.init_step_to_answer()

                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                                answer_function_list["action"].append({"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name})
                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_0_answer_function_list"] = answer_function_list
                            inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"
                            break
                    else:
                        action_name_label = "error"
                        # Provide candidate correct answers
                        for i, (answer_function_name_list, answer_function_arguments_list, answer_function_observation_list, answer_user_input_list) in enumerate(zip(
                                current_step_function_name_list,
                                current_step_function_arguments_list,
                                current_step_function_observation_list,
                                current_step_user_input_list
                            )
                        ):
                            answer_function_list = {"action": []}
                            for answer_function_name, answer_function_arguments, answer_function_observation, answer_user_input in zip(
                                    answer_function_name_list,
                                    answer_function_arguments_list,
                                    answer_function_observation_list,
                                    answer_user_input_list
                            ):
                                answer_function_list["action"].append(
                                    {"arguments": json.dumps(answer_function_arguments, ensure_ascii=False), "name": answer_function_name}
                                )
                                if answer_function_name == "ask_user_for_required_parameters":
                                    answer_function_list["observation"] = answer_function_observation
                                    answer_function_list["user_input"] = answer_user_input
                                elif answer_function_name == "prepare_to_answer":
                                    answer_function_list["observation"] = answer_function_observation

                            inference_log[f"step_{step}"].setdefault("inference_answer", {})[f"candidate_{i}_answer_function_list"] = answer_function_list

                        inference_log[f"step_{step}"]["inference_output"].update(
                            {
                                "current_action_name_label": "error",
                                "error_reason": f"action name not in candidate_answer_function_list"
                            }
                        )
                        break

            if action_name_label == "correct":
                inference_log[f"step_{step}"]["inference_output"]["current_action_name_label"] = "correct"
                break

            step += 1
            inference_data["messages"] = messages

        # print(f"inference end\n", flush=True)
        # for message in messages:
        #     print(json.dumps(message, ensure_ascii=False, indent=4) + "\n", flush=True)
        # print(f"action_name_label: {action_name_label}\n", flush=True)
        if action_name_label == "correct":
            if step == (tool_call_graph.min_length - 1):
                is_optimal = True
            else:
                is_optimal = False
        else:
            is_optimal = False

        return {
            "action_name_label": action_name_label,
            "is_optimal": is_optimal,
            "inference_log": inference_log,
            "latency": latency,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count
        }

    @final
    def write(self, result, result_dir, update_mode=False):
        model_name_dir = self.model_name.replace("/", "_")
        model_result_dir = result_dir / model_name_dir
        model_result_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            result = [result]

        file_path = model_result_dir / os.path.basename(PROMPT_PATH).replace(".jsonl", "_result.jsonl")

        if update_mode:
            # Load existing entries from the file
            existing_entries = {}
            if file_path.exists():
                existing_entries = {entry["id"]: entry for entry in load_file(file_path)}

            # Update existing entries with new data
            for entry in result:
                existing_entries[entry["id"]] = entry

            # Sort entries by `id` and write them back to ensure order consistency
            sorted_entries = sorted(existing_entries.values(), key=sort_key)
            with open(file_path, "w") as fout:
                for entry in sorted_entries:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        else:
            # Normal mode: Append in sorted order
            result.sort(key=sort_key)
            with open(file_path, "a") as fout:
                for entry in result:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
