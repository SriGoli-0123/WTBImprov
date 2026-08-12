"""IGAR-v25: deterministic state and provenance-isolated grounding.

This handler deliberately leaves the restored IGAR-v24 implementation intact.
It changes only two mechanics that the repeated v24 run exposed:

1. required-slot ordering is made deterministic; and
2. assistant-authored prose cannot become evidence for later arguments.

System records, user messages, and tool observations remain authoritative.
Assistant tool calls remain available to the repeat-call audit, but their prose
and proposed argument values are not used as grounding receipts.
"""

import json
from collections import Counter

from wtb.model_handler.api_inference.oai import OpenAIHandler


class IGARV25Handler(OpenAIHandler):
    """A conservative provenance correction on top of exact IGAR-v24."""

    _TRUSTED_ROLES = frozenset({"system", "user", "tool"})

    @classmethod
    def _trusted_messages(cls, messages):
        """Keep facts that came from outside the assistant itself.

        Assistant messages are intentionally absent. This stops a value that
        appeared only in an earlier clarification or draft from validating the
        assistant's next call. Tool observations are retained because they are
        external results, and system records are retained because IGAR builds
        them deterministically from the session history.
        """
        return [
            message
            for message in (messages or [])
            if message.get("role") in cls._TRUSTED_ROLES
        ]

    @classmethod
    def _authoritative_context_text(cls, messages):
        return json.dumps(cls._trusted_messages(messages), ensure_ascii=False)

    def _build_dialogue_state(self, tools, messages, history_answer_lists=None):
        """Return v24's state with stable schema-defined slot ordering."""
        state = super()._build_dialogue_state(
            tools,
            self._trusted_messages(messages),
            history_answer_lists,
        )

        required_order = {}
        slot_coverage = Counter()
        global_slot_order = []
        tool_order = []

        for tool in tools or []:
            function = tool.get("function", {})
            name = function.get("name")
            if not name:
                continue
            tool_order.append(name)
            required = list(dict.fromkeys(
                (function.get("parameters", {}) or {}).get("required", []) or []
            ))
            required_order[name] = required
            for slot in required:
                slot_coverage[slot] += 1
                if slot not in global_slot_order:
                    global_slot_order.append(slot)

        raw_unresolved = state.get("unresolved_required_slots", {}) or {}
        stable_unresolved = {}
        for name in tool_order:
            if name not in raw_unresolved:
                continue
            missing = set(raw_unresolved[name])
            stable_unresolved[name] = [
                slot for slot in required_order.get(name, []) if slot in missing
            ]

        # Preserve unexpected entries safely, while still making their display
        # deterministic. Normal WTB schemas never enter this fallback.
        for name in sorted(set(raw_unresolved) - set(stable_unresolved)):
            stable_unresolved[name] = sorted(raw_unresolved[name])

        feasible = set(state.get("feasible_tools", []) or [])
        state["feasible_tools"] = [name for name in tool_order if name in feasible]
        state["unresolved_required_slots"] = stable_unresolved

        missing_scores = Counter()
        for missing_slots in stable_unresolved.values():
            for slot in missing_slots:
                missing_scores[slot] += slot_coverage.get(slot, 1)

        if missing_scores:
            best_score = max(missing_scores.values())
            state["clarify_slot"] = next(
                slot
                for slot in global_slot_order
                if missing_scores.get(slot) == best_score
            )
        else:
            state["clarify_slot"] = None

        return state

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
        """Apply v24's receipt gate using authoritative sources only."""
        cleaned_args = self._clean_tool_call_arguments(tc_name, tc_args, tools)
        context_text = self._authoritative_context_text(messages)
        cleaned_args, dropped_keys, ungrounded_required = self._verify_and_filter_arguments(
            tc_name,
            cleaned_args,
            tools,
            context_text,
            dialogue_state=dialogue_state,
        )
        if dropped_keys or ungrounded_required:
            inference_log.setdefault("receipt_notes", []).append({
                "step": step,
                "tool": tc_name,
                "dropped_ungrounded_optional": dropped_keys,
                "ungrounded_required": ungrounded_required,
                "grounding_policy": "user_tool_system_only",
            })
        return cleaned_args

    def _unfounded_required(
        self,
        tool_calls,
        tools,
        context_text,
        dialogue_state=None,
    ):
        """Reject required guesses even if they appeared in assistant prose."""
        try:
            messages = json.loads(context_text)
        except Exception:
            messages = []
        return super()._unfounded_required(
            tool_calls,
            tools,
            self._authoritative_context_text(messages),
            dialogue_state=dialogue_state,
        )

    def _candidate_violations(self, candidate, inference_data):
        """Audit grounding without losing v24's completed-call detection."""
        messages = inference_data["messages"]
        trusted_data = dict(inference_data)
        trusted_data["messages"] = self._trusted_messages(messages)
        audit = super()._candidate_violations(candidate, trusted_data)
        history_calls = set()

        for message in messages:
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        pass
                history_calls.add((function.get("name"), self._canon(arguments)))

        for tool_call in candidate.get("tool_calls") or []:
            function = tool_call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    continue
            if (name, self._canon(arguments)) in history_calls:
                audit["total"] += 2
                audit["detail"].append(
                    f"{name}: exact repeat of a call already answered earlier"
                )

        return audit
