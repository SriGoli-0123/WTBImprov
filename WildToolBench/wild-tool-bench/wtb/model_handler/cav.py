"""Contrastive Action Verification (CAV).

CAV is an inference-time controller for messy, multi-turn tool use.  It does
not know about WildToolBench task types, answers, graphs, or scoring.  Its only
inputs are the same runtime objects available to a deployed assistant:

    messages + tool schemas

The controller keeps the model's normal response unless there is a mechanical
reason to intervene.  For tool calls it checks:

1. turn causality: did the latest external event change the proposed action?
2. action inertia: does the action survive removal of prior tool-call syntax?
3. provenance: can arguments be traced to visible messages/tool results?
4. executability: are required arguments present and schema-valid now?
5. completion: has this exact call already produced a visible result?

No prompt text is added and no evaluator data is accepted.
"""

from __future__ import annotations

import json
import re
import time

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy


class RuntimeFirewallError(ValueError):
    """Raised when evaluation-only data is passed into the controller."""


class CAVController:
    """Verify a model-authored action without teaching the model the benchmark."""

    _ALLOWED_RUNTIME_KEYS = frozenset({"messages", "tools"})
    _FORBIDDEN_RUNTIME_KEYS = frozenset({
        "answer",
        "answer_list",
        "english_answer_list",
        "expected",
        "gold",
        "history_answer_lists",
        "score",
        "task_idx",
        "test_entry_id",
        "tool_call_graph",
    })

    def __init__(self, generate, generate_text=None, max_workers=8):
        """
        Args:
            generate: callable(runtime_view) -> normalized response dict.
            generate_text: optional callable(runtime_view) -> text response dict.
            max_workers: upper bound for independent counterfactual requests.
        """
        self.generate = generate
        self.generate_text = generate_text
        self.max_workers = max(1, int(max_workers))

    @classmethod
    def runtime_view(cls, data):
        """Copy and validate the complete controller input.

        Rejecting unknown top-level keys is intentional: it makes accidental
        evaluator leakage fail loudly instead of silently becoming a feature.
        """
        if not isinstance(data, dict):
            raise RuntimeFirewallError("CAV input must be a dictionary")
        keys = set(data)
        forbidden = sorted(keys & cls._FORBIDDEN_RUNTIME_KEYS)
        unknown = sorted(keys - cls._ALLOWED_RUNTIME_KEYS)
        if forbidden:
            raise RuntimeFirewallError(
                "evaluation-only keys are forbidden: " + ", ".join(forbidden)
            )
        if unknown:
            raise RuntimeFirewallError(
                "unknown runtime keys are forbidden: " + ", ".join(unknown)
            )
        messages = data.get("messages")
        tools = data.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise RuntimeFirewallError("messages and tools must both be lists")
        return {"messages": deepcopy(messages), "tools": deepcopy(tools)}

    @staticmethod
    def _tool_schema(tool):
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        return function if isinstance(function, dict) else {}

    @staticmethod
    def _arguments(tool_call):
        raw = tool_call.get("function", {}).get("arguments", {})
        if isinstance(raw, dict):
            return deepcopy(raw)
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @classmethod
    def action_signature(cls, response):
        """Exact action signature used by the two counterfactual comparisons."""
        if not isinstance(response, dict):
            return ("error",)
        calls = response.get("tool_calls") or []
        if calls:
            signatures = []
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = cls._arguments(call)
                canonical = json.dumps(
                    arguments, sort_keys=True, ensure_ascii=False
                ) if arguments is not None else "<invalid>"
                signatures.append((name, canonical))
            return ("tools",) + tuple(sorted(signatures))
        if response.get("content"):
            return ("text",)
        return ("empty",)

    @classmethod
    def _argument_receipts(cls, response):
        receipts = set()
        if not isinstance(response, dict):
            return receipts
        for call in response.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            arguments = cls._arguments(call)
            if not name or arguments is None:
                continue
            for key, value in arguments.items():
                receipts.add((
                    name,
                    key,
                    json.dumps(value, sort_keys=True, ensure_ascii=False),
                ))
        return receipts

    @staticmethod
    def _counterfactual_prefix(runtime):
        """Return the state immediately before the latest external event.

        For a user message, retain its role but blank its content.  For a tool
        observation, also remove the assistant call that produced it so the
        resulting chat remains structurally valid.
        """
        messages = runtime["messages"]
        event_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") in {"user", "tool"}:
                event_idx = idx
                break
        if event_idx is None:
            return None

        cut_idx = event_idx
        if messages[event_idx].get("role") == "tool":
            tool_call_id = messages[event_idx].get("tool_call_id")
            for idx in range(event_idx - 1, -1, -1):
                message = messages[idx]
                if message.get("role") != "assistant" or not message.get("tool_calls"):
                    continue
                call_ids = {
                    call.get("id") for call in message["tool_calls"]
                    if isinstance(call, dict)
                }
                if tool_call_id is None or tool_call_id in call_ids:
                    cut_idx = idx
                    break

        if messages[event_idx].get("role") == "user":
            prefix = deepcopy(messages[:event_idx])
            blank_user = {
                key: deepcopy(value)
                for key, value in messages[event_idx].items()
                if key not in {"content", "tool_calls"}
            }
            blank_user["role"] = "user"
            blank_user["content"] = ""
            prefix.append(blank_user)
        else:
            prefix = deepcopy(messages[:cut_idx])
        if not prefix:
            return None
        return {"messages": prefix, "tools": deepcopy(runtime["tools"])}

    @staticmethod
    def _syntax_neutral(runtime):
        """Remove prior action-format syntax while retaining raw visible facts.

        Tool observations become plain assistant content in this diagnostic
        view.  No instruction, label, or interpretation is added.
        """
        neutral = []
        for message in runtime["messages"]:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                content = message.get("content")
                if content:
                    neutral.append({"role": "assistant", "content": content})
                continue
            if role == "tool":
                neutral.append({
                    "role": "assistant",
                    "content": message.get("content", ""),
                })
                continue
            neutral.append(deepcopy(message))
        return {"messages": neutral, "tools": deepcopy(runtime["tools"])}

    @staticmethod
    def _normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    @classmethod
    def _walk_scalars(cls, value):
        if isinstance(value, dict):
            for child in value.values():
                yield from cls._walk_scalars(child)
        elif isinstance(value, list):
            for child in value:
                yield from cls._walk_scalars(child)
        elif value is not None:
            yield value

    @classmethod
    def _visible_sources(cls, messages):
        raw_text = []
        scalars = []
        for message in messages:
            content = message.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                raw_text.append(content)
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None
                if parsed is not None:
                    scalars.extend(cls._walk_scalars(parsed))
            else:
                raw_text.append(json.dumps(content, ensure_ascii=False))
                scalars.extend(cls._walk_scalars(content))

            # Arguments from a completed, visible call may be reused later.
            for call in message.get("tool_calls") or []:
                arguments = cls._arguments(call)
                if arguments is not None:
                    scalars.extend(cls._walk_scalars(arguments))
        return raw_text, scalars

    @classmethod
    def _directly_supported(cls, value, messages):
        if value is None:
            return True
        if isinstance(value, (dict, list)):
            leaves = list(cls._walk_scalars(value))
            return all(cls._directly_supported(leaf, messages) for leaf in leaves)

        raw_text, scalars = cls._visible_sources(messages)
        normalized = cls._normalize(value)
        if not normalized:
            return True

        for source in scalars:
            if cls._normalize(source) == normalized:
                return True
        for text in raw_text:
            source_tokens = re.findall(r"[A-Za-z0-9]+", text)
            normalized_tokens = [cls._normalize(token) for token in source_tokens]
            if normalized in normalized_tokens:
                return True

            value_token_count = len(re.findall(r"[A-Za-z0-9]+", str(value)))
            if value_token_count > 1:
                for idx in range(len(normalized_tokens) - value_token_count + 1):
                    if "".join(
                        normalized_tokens[idx:idx + value_token_count]
                    ) == normalized:
                        return True
        return False

    @staticmethod
    def _coerce_schema_value(value, schema):
        """Apply only lossless, schema-declared coercions."""
        if not isinstance(schema, dict):
            return value
        expected = schema.get("type")
        enum = schema.get("enum")
        if enum and isinstance(value, str):
            for option in enum:
                if isinstance(option, str) and option.lower() == value.lower():
                    value = option
                    break
        if expected in {"integer", "number"} and isinstance(value, str):
            if re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()):
                number = float(value)
                return int(number) if expected == "integer" else number
        if expected == "boolean" and isinstance(value, str):
            if value.lower() in {"true", "false"}:
                return value.lower() == "true"
        return value

    @classmethod
    def _schema_valid(cls, value, schema):
        if not isinstance(schema, dict):
            return True
        if "enum" in schema and value not in schema["enum"]:
            return False
        expected = schema.get("type")
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            if not isinstance(value, list):
                return False
            item_schema = schema.get("items", {})
            return all(cls._schema_valid(item, item_schema) for item in value)
        if expected == "object":
            if not isinstance(value, dict):
                return False
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if any(key not in value for key in required):
                return False
            if any(key not in properties for key in value):
                return False
            return all(
                cls._schema_valid(child, properties.get(key, {}))
                for key, child in value.items()
            )
        return True

    @classmethod
    def _completed_calls(cls, messages):
        completed_ids = {
            message.get("tool_call_id")
            for message in messages
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        completed = set()
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if call.get("id") not in completed_ids:
                    continue
                function = call.get("function", {})
                arguments = cls._arguments(call)
                if arguments is None:
                    continue
                completed.add((
                    function.get("name"),
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                ))
        return completed

    @classmethod
    def _executable_frontier(
        cls,
        response,
        runtime,
        causal_receipts,
        action_reauthorized=False,
    ):
        """Return calls that are valid and executable now, plus an audit log."""
        schemas = {
            schema.get("name"): schema
            for schema in map(cls._tool_schema, runtime["tools"])
            if schema.get("name")
        }
        completed = cls._completed_calls(runtime["messages"])
        kept_calls = []
        suppressed = []
        optional_drops = []

        for call in response.get("tool_calls") or []:
            function = call.get("function", {})
            name = function.get("name")
            schema = schemas.get(name)
            arguments = cls._arguments(call)
            reasons = []

            if schema is None:
                reasons.append("unknown_tool")
            if arguments is None:
                reasons.append("invalid_arguments_json")
            if reasons:
                suppressed.append({"tool": name, "reasons": reasons})
                continue

            parameters = schema.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            required = set(parameters.get("required", []) or [])
            cleaned = {}

            for key, value in arguments.items():
                if key not in properties:
                    optional_drops.append({"tool": name, "argument": key})
                    continue
                coerced = cls._coerce_schema_value(value, properties[key])
                if not cls._schema_valid(coerced, properties[key]):
                    reasons.append(f"schema_invalid:{key}")
                    continue
                receipt = (
                    name,
                    key,
                    json.dumps(coerced, sort_keys=True, ensure_ascii=False),
                )
                supported = (
                    cls._directly_supported(coerced, runtime["messages"])
                    or receipt in causal_receipts
                )
                if key in required and supported:
                    cleaned[key] = coerced
                elif key in required:
                    reasons.append(f"required_unsupported:{key}")
                elif supported:
                    cleaned[key] = coerced
                else:
                    optional_drops.append({"tool": name, "argument": key})

            missing = sorted(key for key in required if key not in cleaned)
            if missing:
                reasons.extend(f"required_not_executable:{key}" for key in missing)

            canonical = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
            if (name, canonical) in completed and not action_reauthorized:
                reasons.append("already_completed")

            if reasons:
                suppressed.append({"tool": name, "reasons": reasons})
                continue

            clean_call = deepcopy(call)
            clean_call.setdefault("function", {})["arguments"] = json.dumps(
                cleaned, ensure_ascii=False
            )
            kept_calls.append(clean_call)

        return kept_calls, {
            "suppressed_calls": suppressed,
            "dropped_optional_arguments": optional_drops,
        }

    def _safe_generate(self, runtime):
        try:
            return self.generate(self.runtime_view(runtime)), None
        except Exception as exc:  # Network/server failures must preserve anchor.
            return None, f"{type(exc).__name__}: {exc}"

    def _safe_text(self, runtime):
        if self.generate_text is None:
            return None, "text generation is unavailable"
        try:
            return self.generate_text(self.runtime_view(runtime)), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _tokens(responses, key):
        return sum(
            int(response.get(key) or 0)
            for response in responses
            if isinstance(response, dict)
        )

    def decide(self, data):
        """Generate normally, verify minimally, and return one response."""
        started = time.monotonic()
        runtime = self.runtime_view(data)
        anchor, anchor_error = self._safe_generate(runtime)
        if anchor is None:
            raise RuntimeError("anchor generation failed: " + str(anchor_error))

        cav_log = {
            "method": "contrastive_action_verification",
            "runtime_keys": sorted(runtime),
            "anchor_signature": list(self.action_signature(anchor)),
        }
        calls_for_accounting = [anchor]

        if not anchor.get("tool_calls"):
            result = dict(anchor)
            cav_log["decision"] = "preserve_text_anchor"
            result["cav_log"] = cav_log
            return result

        counterfactuals = {}
        prefix_runtime = self._counterfactual_prefix(runtime)
        neutral_runtime = self._syntax_neutral(runtime)
        requests = {}
        if prefix_runtime is not None:
            requests["without_latest_event"] = prefix_runtime
        if neutral_runtime is not None:
            requests["without_prior_action_syntax"] = neutral_runtime

        if requests:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(requests))
            ) as executor:
                futures = {
                    name: executor.submit(self._safe_generate, view)
                    for name, view in requests.items()
                }
                for name, future in futures.items():
                    response, error = future.result()
                    counterfactuals[name] = {
                        "response": response,
                        "error": error,
                    }
                    if response is not None:
                        calls_for_accounting.append(response)

        prefix_response = counterfactuals.get(
            "without_latest_event", {}
        ).get("response")
        neutral_response = counterfactuals.get(
            "without_prior_action_syntax", {}
        ).get("response")
        anchor_signature = self.action_signature(anchor)
        prefix_signature = self.action_signature(prefix_response)
        neutral_signature = self.action_signature(neutral_response)

        latest_event_changed_action = (
            prefix_response is not None and anchor_signature != prefix_signature
        )
        survives_syntax_neutralization = (
            neutral_response is not None and anchor_signature == neutral_signature
        )
        causal_support = (
            latest_event_changed_action and survives_syntax_neutralization
        )
        anchor_receipts = self._argument_receipts(anchor)
        prefix_receipts = self._argument_receipts(prefix_response)
        neutral_receipts = self._argument_receipts(neutral_response)
        causal_receipts = (
            anchor_receipts & neutral_receipts
        ) - prefix_receipts
        action_momentum = (
            prefix_response is not None
            and neutral_response is not None
            and anchor_signature == prefix_signature
            and anchor_signature != neutral_signature
        )

        cav_log["counterfactuals"] = {
            name: {
                "signature": list(self.action_signature(item["response"])),
                "error": item["error"],
            }
            for name, item in counterfactuals.items()
        }
        cav_log["causal_support"] = {
            "latest_event_changed_action": latest_event_changed_action,
            "survives_syntax_neutralization": survives_syntax_neutralization,
            "proven": causal_support,
            "argument_receipts": [
                {"tool": tool, "argument": key, "value": value}
                for tool, key, value in sorted(causal_receipts)
            ],
        }
        cav_log["action_momentum"] = action_momentum

        frontier, audit = self._executable_frontier(
            anchor,
            runtime,
            causal_receipts=causal_receipts,
            action_reauthorized=causal_support,
        )
        cav_log["audit"] = audit

        result = dict(anchor)
        if frontier and not action_momentum:
            result["tool_calls"] = frontier
            cav_log["decision"] = (
                "preserve_tool_anchor"
                if not audit["suppressed_calls"]
                and not audit["dropped_optional_arguments"]
                else "emit_executable_frontier"
            )
        else:
            text_response, text_error = self._safe_text(runtime)
            cav_log["text_fallback_error"] = text_error
            if text_response is not None:
                calls_for_accounting.append(text_response)
            if text_response and text_response.get("content"):
                result = dict(text_response)
                result["tool_calls"] = None
                cav_log["decision"] = (
                    "replace_action_momentum_with_model_text"
                    if action_momentum
                    else "replace_non_executable_calls_with_model_text"
                )
            else:
                # Selective intervention is fail-open: an unavailable fallback
                # never destroys a syntactically usable baseline response.
                result = dict(anchor)
                cav_log["decision"] = "preserve_anchor_fallback_unavailable"

        result["input_token"] = self._tokens(calls_for_accounting, "input_token")
        result["output_token"] = self._tokens(calls_for_accounting, "output_token")
        result["latency"] = time.monotonic() - started
        result["cav_log"] = cav_log
        return result
