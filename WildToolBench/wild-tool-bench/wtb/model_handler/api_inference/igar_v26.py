"""IGAR-v26: prompt-invariant continuity for documented tool use.

The model sees the stock WildToolBench conversation: the date-only system
message, the native prior transcript, the current user message, and the tool
schemas.  Everything added by this handler is controller-side bookkeeping.

V26 keeps V25's provenance rule, then adds two conservative mechanisms:

* an open-question contract remembers which documented field a clarification
  was asking for, so an informative reply cannot silently turn into another
  copy of the same question; and
* referential turns may preserve a value from an already executed historical
  call when the model changes that value without any user evidence.

Neither mechanism reads the current task's reference answer or evaluator.
"""

import json
import re
from copy import deepcopy

from wtb.model_handler.api_inference.igar_v25 import IGARV25Handler
from wtb.tool_call_graph import ToolCallGraph
from wtb.utils import generate_random_string


class IGARV26Handler(IGARV25Handler):
    """A conservative, training-free controller around the unchanged model."""

    _QUESTION_STARTS = (
        "can you", "could you", "do you", "may i", "please provide",
        "tell me", "what", "which", "where", "when", "who", "would you",
    )
    _REFERENCE_WORDS = frozenset({
        "it", "its", "same", "this", "that", "these", "those", "former",
        "latter", "again", "previous", "previously", "earlier", "such",
    })
    _CHANGE_WORDS = frozenset({
        "another", "different", "else", "instead", "new", "other",
        "replace", "switch",
    })
    _TEMPORAL_WORDS = frozenset({
        "after", "before", "later", "month", "next", "now", "today",
        "tomorrow", "tonight", "week", "weekend", "year", "yesterday",
    })
    _CANCEL_PHRASES = (
        "cancel that", "do not", "don't", "forget it", "never mind",
        "nevermind", "no longer", "stop", "thanks", "thank you",
    )
    _DEFER_PHRASES = (
        "give me a moment", "hold on", "i don't know", "i do not know",
        "not sure", "one moment", "wait", "wait a minute",
    )
    _STOP_WORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do",
        "for", "from", "get", "give", "i", "in", "is", "it", "me", "of",
        "on", "or", "please", "the", "this", "to", "tool", "use", "with",
        "you", "your",
    })

    # ------------------------------------------------------------------
    # Stock WTB input construction
    # ------------------------------------------------------------------
    @staticmethod
    def _stock_add_action_observation(task, answer_list, consecutive_tool_messages):
        """Reproduce the official WTB history transcript without summaries."""
        tool_call_graph = ToolCallGraph(answer_list)
        tool_call_graph.add_node_list()
        tool_call_graph.generate_all_path()
        optimal_path = tool_call_graph.optimal_path_list[0]

        current_messages = [{"role": "user", "content": task}]
        for idx_action_list in optimal_path:
            formatted_actions = []
            observations = []
            for idx in idx_action_list:
                answer = answer_list[idx]
                action = answer["action"]
                name = action["name"]
                if name == "ask_user_for_required_parameters":
                    current_messages.extend([
                        {"role": "assistant", "content": answer["observation"]},
                        {"role": "user", "content": answer["user_input"]},
                    ])
                elif name == "prepare_to_answer":
                    current_messages.append({
                        "role": "assistant",
                        "content": answer["observation"],
                    })
                else:
                    formatted_actions.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": deepcopy(action["arguments"]),
                        },
                    })
                    observations.append(answer["observation"])

            if formatted_actions:
                current_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": formatted_actions,
                })
                if consecutive_tool_messages:
                    current_messages.extend(
                        {"role": "tool", "content": observation}
                        for observation in observations
                    )
                else:
                    current_messages.append({
                        "role": "tool",
                        "content": json.dumps(observations, ensure_ascii=False),
                    })
        return current_messages

    @staticmethod
    def _stock_convert_to_tool_calls(messages):
        """Attach OpenAI tool-call IDs exactly where the stock runner needs them."""
        pending_ids = []
        converted = []
        for original in messages:
            message = deepcopy(original)
            if message["role"] == "assistant" and message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    arguments = tool_call["function"].get("arguments", {})
                    if isinstance(arguments, dict):
                        tool_call["function"]["arguments"] = json.dumps(
                            arguments, ensure_ascii=False
                        )
                    if "id" not in tool_call:
                        tool_call["id"] = "toolu_bdrk_" + generate_random_string(24)
                    pending_ids.append(tool_call["id"])
            elif message["role"] == "tool":
                if pending_ids:
                    message["tool_call_id"] = pending_ids.pop(0)
                message["content"] = json.dumps(message.get("content"), ensure_ascii=False)
            converted.append(message)
        return converted

    def _pre_messages_processing(
        self,
        env_info,
        current_task,
        history_tasks,
        history_answer_lists,
        consecutive_tool_messages=True,
        tools=None,
    ):
        """Build the untouched benchmark input and keep state outside it."""
        messages = [{"role": "system", "content": f"Current Date: {env_info}"}]
        for history_task, history_answers in zip(history_tasks, history_answer_lists):
            messages.extend(self._stock_add_action_observation(
                history_task, history_answers, consecutive_tool_messages
            ))
        messages.append({"role": "user", "content": current_task})
        messages = self._stock_convert_to_tool_calls(messages)

        current_task_index = len(messages) - 1
        # State is derived from the same rendered messages the model can see.
        # The structured history answers are used only because WTB stores its
        # native transcript in that form; they are not a second evidence path.
        dialogue_state = self._build_dialogue_state(tools or [], messages, None)
        dialogue_state["_v26_current_task_index"] = current_task_index
        dialogue_state["_v26_open_question"] = None
        return messages, dialogue_state

    # ------------------------------------------------------------------
    # Documentation compiler and clarification continuity
    # ------------------------------------------------------------------
    @classmethod
    def _tokens(cls, value):
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
        return {
            token for token in re.findall(r"[a-z0-9]+", split.lower())
            if token not in cls._STOP_WORDS
        }

    @staticmethod
    def _raw_tokens(value):
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
        return set(re.findall(r"[a-z0-9]+", split.lower()))

    @classmethod
    def _flatten_fields(cls, schema, prefix="", inherited_required=True):
        """Flatten nested JSON-schema fields into controller-side doc records."""
        if not isinstance(schema, dict):
            return []
        required = set(schema.get("required", []) or [])
        fields = []
        for name, prop in (schema.get("properties", {}) or {}).items():
            path = f"{prefix}.{name}" if prefix else name
            is_required = inherited_required and name in required
            fields.append({
                "path": path,
                "name": name,
                "required": is_required,
                "tokens": cls._tokens(name) | cls._tokens(prop.get("description", "")),
            })
            if prop.get("type") == "object" or prop.get("properties"):
                fields.extend(cls._flatten_fields(prop, path, is_required))
            items = prop.get("items") or {}
            if isinstance(items, dict) and items.get("properties"):
                fields.extend(cls._flatten_fields(items, path, is_required))
        return fields

    @classmethod
    def _documentation_contracts(cls, tools):
        contracts = []
        for tool in tools or []:
            function = tool.get("function", {})
            name = function.get("name")
            if not name:
                continue
            contracts.append({
                "name": name,
                "tokens": cls._tokens(name) | cls._tokens(function.get("description", "")),
                "fields": cls._flatten_fields(function.get("parameters", {}) or {}),
            })
        return contracts

    @staticmethod
    def _looks_like_question(content):
        if not isinstance(content, str) or not content.strip():
            return False
        lowered = content.strip().lower()
        return "?" in lowered or lowered.startswith(IGARV26Handler._QUESTION_STARTS)

    @classmethod
    def _task_text(cls, messages, dialogue_state):
        start = (dialogue_state or {}).get("_v26_current_task_index", 0)
        return " ".join(
            str(message.get("content") or "")
            for message in (messages or [])[start:]
            if message.get("role") == "user"
        )

    @classmethod
    def _infer_question_contract(cls, content, messages, tools, dialogue_state):
        """Map a model question to one field in one supplied tool document."""
        if not cls._looks_like_question(content):
            return None
        question_tokens = cls._tokens(content)
        intent_tokens = cls._tokens(cls._task_text(messages, dialogue_state))
        ranked = []
        for contract in cls._documentation_contracts(tools):
            best_field = None
            best_field_score = 0
            for field in contract["fields"]:
                name_overlap = len(cls._tokens(field["name"]) & question_tokens)
                doc_overlap = len(field["tokens"] & question_tokens)
                lexical_score = 3 * name_overlap + doc_overlap
                if lexical_score == 0:
                    continue
                score = lexical_score + int(field["required"])
                if score > best_field_score:
                    best_field_score = score
                    best_field = field
            if best_field is None or best_field_score == 0:
                continue
            intent_score = len(contract["tokens"] & intent_tokens)
            ranked.append({
                "target_tool": contract["name"],
                "open_field": best_field["path"],
                "score": best_field_score + intent_score,
            })

        ranked.sort(key=lambda item: (-item["score"], item["target_tool"], item["open_field"]))
        if not ranked:
            return None
        if len(ranked) > 1 and ranked[0]["score"] < ranked[1]["score"] + 2:
            return None
        return ranked[0]

    @classmethod
    def _is_cancelled(cls, reply):
        lowered = str(reply or "").strip().lower()
        return any(phrase in lowered for phrase in cls._CANCEL_PHRASES)

    @classmethod
    def _is_informative_reply(cls, reply):
        lowered = str(reply or "").strip().lower()
        if not lowered or cls._is_cancelled(lowered):
            return False
        return not any(lowered == phrase or lowered.startswith(phrase) for phrase in cls._DEFER_PHRASES)

    @staticmethod
    def _user_count(messages):
        return sum(message.get("role") == "user" for message in messages or [])

    def _remember_question(self, contract, messages, dialogue_state):
        if not contract or not isinstance(dialogue_state, dict):
            return
        existing = dialogue_state.get("_v26_open_question") or {}
        asked = list(existing.get("asked_fields", []))
        field = contract.get("open_field")
        if field and field not in asked:
            asked.append(field)
        dialogue_state["_v26_open_question"] = {
            "target_tool": contract["target_tool"],
            "open_field": field,
            "asked_fields": asked,
            "user_count": self._user_count(messages),
        }

    def _request_documented_tool(self, inference_data, target_tool):
        """Retry the same untouched task with one documented tool forced."""
        try:
            api_response, latency = self.generate_with_backoff(
                messages=inference_data["messages"],
                model=self.model_name,
                temperature=self.temperature,
                tools=inference_data["tools"],
                tool_choice={"type": "function", "function": {"name": target_tool}},
            )
            candidate = self._normalize_response(self._parse_api_response(api_response))
            candidate["latency"] = latency
        except Exception as error:
            print(f"[IGAR-v26] documented retry unavailable: {error}", flush=True)
            return None

        calls = candidate.get("tool_calls") or []
        if not calls or any(
            call.get("function", {}).get("name") != target_tool for call in calls
        ):
            return None
        if self._candidate_violations(candidate, inference_data)["total"] != 0:
            return None
        context = self._authoritative_context_text(inference_data["messages"])
        if self._unfounded_required(
            calls,
            inference_data["tools"],
            context,
            dialogue_state=inference_data.get("dialogue_state"),
        ):
            return None
        return candidate

    def _consensus_generate(self, inference_data):
        """Run V25, then close only a confidently identified answered question."""
        messages = inference_data.get("messages") or []
        tools = inference_data.get("tools") or []
        dialogue_state = inference_data.get("dialogue_state") or {}
        dialogue_state["_v26_last_user_count"] = self._user_count(messages)
        anchor = super()._consensus_generate(inference_data)
        pending = dialogue_state.get("_v26_open_question")

        calls = anchor.get("tool_calls") or []
        if calls:
            if pending and any(
                call.get("function", {}).get("name") == pending.get("target_tool")
                for call in calls
            ):
                dialogue_state["_v26_open_question"] = None
            return anchor

        content = anchor.get("content")
        inferred = self._infer_question_contract(
            content, messages, tools, dialogue_state
        )
        if not pending:
            self._remember_question(inferred, messages, dialogue_state)
            return anchor

        if self._user_count(messages) <= pending.get("user_count", 0):
            return anchor
        last_reply = next(
            (message.get("content") for message in reversed(messages)
             if message.get("role") == "user"),
            "",
        )
        if self._is_cancelled(last_reply):
            dialogue_state["_v26_open_question"] = None
            return anchor
        if not self._is_informative_reply(last_reply):
            if inferred:
                self._remember_question(inferred, messages, dialogue_state)
            return anchor

        # A genuinely new documented field is a useful next clarification.
        if inferred and inferred.get("open_field") not in pending.get("asked_fields", []):
            self._remember_question(inferred, messages, dialogue_state)
            return anchor

        target = pending.get("target_tool")
        if not target:
            return anchor
        retry = self._request_documented_tool(inference_data, target)
        if retry is None:
            if inferred:
                self._remember_question(inferred, messages, dialogue_state)
            return anchor

        retry["input_token"] = (anchor.get("input_token") or 0) + (retry.get("input_token") or 0)
        retry["output_token"] = (anchor.get("output_token") or 0) + (retry.get("output_token") or 0)
        retry["latency"] = (anchor.get("latency") or 0) + (retry.get("latency") or 0)
        log = dict(anchor.get("consensus_log") or {})
        log["v26_continuity"] = {
            "decision": "execute_answered_documented_tool",
            "target_tool": target,
            "answered_field": pending.get("open_field"),
        }
        retry["consensus_log"] = log
        dialogue_state["_v26_open_question"] = None
        print(
            f"[IGAR-v26] answered clarification -> {target}",
            flush=True,
        )
        return retry

    def _unfounded_required(
        self,
        tool_calls,
        tools,
        context_text,
        dialogue_state=None,
    ):
        """Remember an ask-gate question without exposing the record to the model."""
        hits = super()._unfounded_required(
            tool_calls, tools, context_text, dialogue_state=dialogue_state
        )
        if hits and isinstance(dialogue_state, dict):
            target_tools = {hit["tool"] for hit in hits}
            if len(target_tools) == 1:
                target = next(iter(target_tools))
                fields = [hit["param"] for hit in hits]
                dialogue_state["_v26_open_question"] = {
                    "target_tool": target,
                    "open_field": fields[0] if len(fields) == 1 else None,
                    "asked_fields": fields[:1] if len(fields) == 1 else [],
                    "candidate_fields": fields,
                    "user_count": self._user_count_from_context_or_state(dialogue_state),
                }
        return hits

    def _authored_clarification(self, inference_data, content):
        """Attach the model's actual question to an ask-gate contract."""
        clarification = super()._authored_clarification(inference_data, content)
        if not clarification:
            return clarification
        dialogue_state = inference_data.get("dialogue_state") or {}
        pending = dialogue_state.get("_v26_open_question") or {}
        inferred = self._infer_question_contract(
            clarification,
            inference_data.get("messages") or [],
            inference_data.get("tools") or [],
            dialogue_state,
        )
        if inferred and (
            not pending or inferred.get("target_tool") == pending.get("target_tool")
        ):
            self._remember_question(
                inferred, inference_data.get("messages") or [], dialogue_state
            )
        return clarification

    @staticmethod
    def _user_count_from_context_or_state(dialogue_state):
        # Set by _consensus_generate before ask-gate inspection. The fallback
        # keeps old serialized states safe and merely disables a forced retry.
        return dialogue_state.get("_v26_last_user_count", 10 ** 9)

    # ------------------------------------------------------------------
    # Conservative referential continuity
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_arguments(arguments):
        if isinstance(arguments, dict):
            return deepcopy(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _last_executed_history_args(self, tool_name, messages, boundary):
        executed_ids = {
            message.get("tool_call_id")
            for message in messages[:boundary]
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        latest = None
        for message in messages[:boundary]:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                if function.get("name") != tool_name:
                    continue
                call_id = call.get("id")
                if call_id and call_id in executed_ids:
                    latest = self._parse_arguments(function.get("arguments"))
        return latest

    def _preserve_referential_values(
        self, tool_name, arguments, messages, dialogue_state, inference_log, step
    ):
        boundary = (dialogue_state or {}).get("_v26_current_task_index")
        if boundary is None:
            return arguments
        current_messages = [
            message for message in messages[boundary:]
            if message.get("role") in {"user", "tool"}
        ]
        current_text = " ".join(
            str(message.get("content") or "")
            for message in current_messages
        )
        user_text = " ".join(
            str(message.get("content") or "")
            for message in current_messages
            if message.get("role") == "user"
        )
        user_tokens = self._raw_tokens(user_text)
        if not (user_tokens & self._REFERENCE_WORDS):
            return arguments
        if user_tokens & self._CHANGE_WORDS:
            return arguments
        previous = self._last_executed_history_args(tool_name, messages, boundary)
        if not previous:
            return arguments

        repaired = deepcopy(arguments)
        changes = []
        current_tokens = self._raw_tokens(current_text)
        for key, value in list(repaired.items()):
            if key not in previous or previous[key] == value:
                continue
            if isinstance(value, (dict, list)) or isinstance(previous[key], (dict, list)):
                continue
            key_tokens = self._raw_tokens(key)
            proposed_tokens = self._raw_tokens(value)
            if proposed_tokens and proposed_tokens <= current_tokens:
                continue
            if (
                (key_tokens & {"date", "day", "month", "time", "year"})
                and (user_tokens & self._TEMPORAL_WORDS)
            ):
                continue
            repaired[key] = deepcopy(previous[key])
            changes.append({"field": key, "from": value, "to": previous[key]})
        if changes:
            inference_log.setdefault("v26_continuity_notes", []).append({
                "step": step,
                "tool": tool_name,
                "changes": changes,
            })
        return repaired

    def _clean_and_verify_call(
        self,
        tc_name,
        tc_args,
        tools,
        messages,
        step,
        inference_log,
        dialogue_state=None,
    ):
        parsed = self._parse_arguments(tc_args)
        parsed = self._preserve_referential_values(
            tc_name,
            parsed,
            messages,
            dialogue_state,
            inference_log,
            step,
        )
        return super()._clean_and_verify_call(
            tc_name,
            parsed,
            tools,
            messages,
            step,
            inference_log,
            dialogue_state=dialogue_state,
        )

    def _maybe_repair(self, chosen, inference_data):
        """V26 never adds an audit instruction or any other prompt message."""
        return self._normalize_response(dict(chosen))
