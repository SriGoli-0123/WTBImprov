"""
A small wrapper that decides *whether to act* before deciding *how to act*.

Background
----------
In this benchmark a turn can end in one of two visible ways: the assistant
emits one or more tool calls, or it replies in plain text. The two "special"
actions (asking the user for a missing parameter, and answering the user)
are both plain text -- they are never exposed as callable tools.

Looking at where a 7B model actually loses points, most of the damage is not
in filling a call in badly. It is in picking the wrong kind of move:

    of 672 failed turns, 416 (62%) emitted the wrong action entirely

The worst case is a turn that should have been answered in words but instead
fires a tool call. That happens 79 times on "answer the user" turns and 125
times on "ask the user" turns. In the other direction, 55 turns replied in
words when a tool call was needed.

So this wrapper does not try to be clever about arguments. It asks three
plain questions before letting a tool call through, and it fixes the shape of
whatever survives.

The rules
---------
1. If the conversation already answers the user, reply in words.
2. If a required value is not in the conversation, ask for it in words.
3. If a tool call is needed, make sure every part of the request is covered.
4. Fix the shape of the call: real keys only, schema types, declared order.
5. Never call something whose result is already sitting in the conversation.

Rules 1-3 are judgment calls, so they get better as the base model gets
better. Rules 4-5 are pure bookkeeping and act as a floor.
"""

import json
import re
from copy import deepcopy

from wtb.model_handler.api_inference.oai import OpenAIHandler


# --------------------------------------------------------------------------
# a tiny stand-in so the rest of the harness can parse what we hand back
# --------------------------------------------------------------------------
class _Response:
    """Looks enough like an OpenAI response for _parse_api_response."""

    def __init__(self, content, tool_calls, prompt_tokens, completion_tokens):
        self._payload = {
            "choices": [{
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                    "reasoning_content": None,
                }
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    def json(self):
        return json.dumps(self._payload, ensure_ascii=False)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _schema_of(tools, name):
    for tool in tools:
        fn = tool.get("function", tool)
        if fn.get("name") == name:
            return fn.get("parameters", {}) or {}
    return {}


def _required_of(schema):
    return list(schema.get("required", []) or [])


def _coerce(value, declared):
    """Nudge a value to the type the schema declares.

    The grader compares types strictly -- "5" and 5 are not the same answer --
    so this is worth doing even though it looks fussy.
    """
    if declared == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if declared == "integer" and isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    if declared == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if declared == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    if declared == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
    return value


def _tidy_arguments(arguments, schema):
    """Rule 4. Drop keys the tool does not have, then fix types and order."""
    properties = schema.get("properties", {}) or {}
    if not properties:
        return arguments

    cleaned = {}
    for key, value in arguments.items():
        if key not in properties:
            continue  # invented key -- the grader rejects the whole call for this
        declared = properties[key].get("type")
        if declared == "array" and isinstance(value, list):
            item_type = (properties[key].get("items") or {}).get("type")
            value = [_coerce(v, item_type) for v in value]
        else:
            value = _coerce(value, declared)
        cleaned[key] = value

    # keep declared order; arrays are compared position by position
    ordered = {}
    for key in properties:
        if key in cleaned:
            ordered[key] = cleaned[key]
    for key, value in cleaned.items():
        ordered.setdefault(key, value)
    return ordered


def _already_called(messages, name, arguments):
    """Rule 5. Has this exact call already been made and answered?"""
    target = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    seen = False
    for message in messages:
        for call in (message.get("tool_calls") or []):
            fn = call.get("function", {})
            if fn.get("name") != name:
                continue
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            if json.dumps(raw or {}, sort_keys=True, ensure_ascii=False) == target:
                seen = True
        if seen and message.get("role") == "tool":
            return True
    return seen


def _conversation_text(messages):
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, ensure_ascii=False))
    return "\n".join(parts)


_NOT_STATED = "NOT STATED"


class GroundedHandler(OpenAIHandler):
    """Wraps the model with the five rules described at the top of the file."""

    # how many extra model calls we are willing to spend deciding the move
    max_checks = 2
    # how many times we will go back and ask for the calls that are still missing
    max_coverage_rounds = 3

    # ---------------------------------------------------------------- asking
    def _ask(self, messages, tools=None, temperature=None):
        response, latency = self.generate_with_backoff(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature if temperature is None else temperature,
            **({"tools": tools} if tools else {})
        )
        parsed = json.loads(response.json())
        message = parsed["choices"][0]["message"]
        usage = parsed.get("usage", {})
        return (
            message.get("content"),
            message.get("tool_calls"),
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            latency,
        )

    # ------------------------------------------------------- rule 1 sanity
    def _already_answered(self, messages):
        """Is everything the user just asked for already in the conversation?

        Only worth asking when there are tool results to answer from, which is
        what keeps this from firing on the first turn of a session.
        """
        if not any(m.get("role") == "tool" for m in messages):
            return False, 0, 0

        probe = deepcopy(messages) + [{
            "role": "user",
            "content": (
                "Before you reply: can the request above be answered using only "
                "information already present in this conversation, with no new "
                "tool call? Answer with one word, YES or NO."
            ),
        }]
        content, _, pin, pout, _ = self._ask(probe, temperature=0.0)
        verdict = (content or "").strip().upper().startswith("YES")
        return verdict, pin, pout

    # ------------------------------------------------------- rule 2 grounding
    def _unsourced_parameters(self, messages, tools, name, arguments):
        """Which required values cannot be pointed at in the conversation?

        The model has to quote the supporting text. A quote can be checked; an
        opinion cannot, which is the whole reason this is phrased as a quote.
        """
        schema = _schema_of(tools, name)
        required = [r for r in _required_of(schema) if r in arguments]
        if not required:
            return [], 0, 0

        listing = "\n".join(
            f"- {key}: {json.dumps(arguments[key], ensure_ascii=False)}" for key in required
        )
        probe = deepcopy(messages) + [{
            "role": "user",
            "content": (
                f"I am about to call `{name}` with these values:\n{listing}\n\n"
                "For each one, quote the exact words from this conversation that "
                f"give that value. If the conversation does not give it, write {_NOT_STATED}. "
                "A value counts as given if it is stated outright, or follows "
                "directly from something stated (a date from 'this weekend', a "
                "country code from a country name, a value taken from a tool "
                "result). Reply one line per value, in the form `key: quote`."
            ),
        }]
        content, _, pin, pout, _ = self._ask(probe, temperature=0.0)

        missing = []
        for line in (content or "").splitlines():
            if _NOT_STATED.lower() not in line.lower():
                continue
            for key in required:
                if re.match(rf"\s*[-*]?\s*`?{re.escape(key)}`?\s*[:\-]", line, re.I):
                    missing.append(key)
        # keep the schema's own order -- lists are compared position by position
        return [k for k in _required_of(schema) if k in missing], pin, pout

    # -------------------------------------------------------- rule 3 coverage
    def _checklist(self, messages):
        """Write down what the user actually asked for, as a numbered list.

        Having the list on paper is the point. Asking "is anything left, given
        what you already wrote?" is a much easier question than "what did you
        forget?", because the second one needs the very breakdown that failed.
        """
        probe = deepcopy(messages) + [{
            "role": "user",
            "content": (
                "List the separate things the user just asked for, one per line, "
                "numbered. Keep each to a few words. Do not call any tool."
            ),
        }]
        content, _, pin, pout, _ = self._ask(probe, temperature=0.0)
        items = [
            line.strip()
            for line in (content or "").splitlines()
            if re.match(r"\s*\d+[.)]\s*\S", line)
        ]
        return items, pin, pout

    def _cover(self, messages, tools, calls):
        """Rule 3. Keep adding calls until nothing on the list is left over.

        Multi-tool turns fail by stopping early -- only 27% of the required
        steps get emitted -- so one reminder is not enough. This repeats until
        the model says it is done, or until the cap is hit.
        """
        spent_in = spent_out = 0
        items, pin, pout = self._checklist(messages)
        spent_in, spent_out = spent_in + pin, spent_out + pout
        if len(items) < 2:
            return calls, spent_in, spent_out  # a single request cannot be half done

        checklist = "\n".join(items)
        for _ in range(self.max_coverage_rounds):
            summary = "\n".join(
                f"- {c['function']['name']}({c['function']['arguments']})" for c in calls
            )
            probe = deepcopy(messages) + [{
                "role": "user",
                "content": (
                    f"The user asked for:\n{checklist}\n\n"
                    f"So far these calls are planned:\n{summary}\n\n"
                    "Is any numbered item still without a call? If every item is "
                    "covered, reply DONE and call nothing. Otherwise call only what "
                    "is still missing."
                ),
            }]
            content, extra, pin, pout, _ = self._ask(probe, tools=tools, temperature=0.0)
            spent_in, spent_out = spent_in + pin, spent_out + pout

            if not extra or (content or "").strip().upper().startswith("DONE"):
                break

            added = False
            for call in extra:
                fn = call.get("function", {})
                raw = fn.get("arguments")
                try:
                    arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    continue
                arguments = _tidy_arguments(arguments, _schema_of(tools, fn.get("name")))
                if _already_called(messages, fn.get("name"), arguments):
                    continue
                fresh = deepcopy(call)
                fresh["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
                if any(
                    c["function"]["name"] == fresh["function"]["name"]
                    and c["function"]["arguments"] == fresh["function"]["arguments"]
                    for c in calls
                ):
                    continue  # already planned; nothing new arrived, so stop
                calls.append(fresh)
                added = True
            if not added:
                break

        return calls, spent_in, spent_out

    # ------------------------------------------------------------------ main
    def _request_tool_call(self, inference_data):
        messages = inference_data["messages"]
        tools = inference_data["tools"]

        content, tool_calls, tin, tout, latency = self._ask(messages, tools=tools)
        checks = 0

        # ---- the model chose to speak. Leave it alone; on answer-the-user
        # ---- turns speaking is right every single time it happens.
        if not tool_calls:
            return _Response(content or "", None, tin, tout), latency

        # ---- rule 4 and 5 first: they are free and cannot make things worse
        kept = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            raw = fn.get("arguments")
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                arguments = {}

            arguments = _tidy_arguments(arguments, _schema_of(tools, name))

            if _already_called(messages, name, arguments):
                continue  # rule 5: the answer is already in the conversation

            call = deepcopy(call)
            call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
            kept.append(call)

        # everything we wanted to do has already been done -> answer instead
        if not kept:
            spoken, _, pin, pout, _ = self._ask(
                deepcopy(messages) + [{
                    "role": "user",
                    "content": "Answer the request using what is already in this conversation. Do not call any tool.",
                }]
            )
            return _Response(spoken or content or "", None, tin + pin, tout + pout), latency

        # ---- rule 1: is a call needed at all?
        if checks < self.max_checks:
            answered, pin, pout = self._already_answered(messages)
            tin, tout, checks = tin + pin, tout + pout, checks + 1
            if answered:
                spoken, _, pin, pout, _ = self._ask(
                    deepcopy(messages) + [{
                        "role": "user",
                        "content": "Answer the request using what is already in this conversation. Do not call any tool.",
                    }]
                )
                return _Response(spoken or "", None, tin + pin, tout + pout), latency

        # ---- rule 2: can every required value be pointed at?
        if checks < self.max_checks:
            first = kept[0]["function"]
            missing, pin, pout = self._unsourced_parameters(
                messages, tools, first["name"], json.loads(first["arguments"])
            )
            tin, tout, checks = tin + pin, tout + pout, checks + 1
            if missing:
                pretty = ", ".join(missing)
                spoken, _, pin, pout, _ = self._ask(
                    deepcopy(messages) + [{
                        "role": "user",
                        "content": (
                            f"The following details are needed but were never given: {pretty}. "
                            "Ask the user for exactly those, in one short sentence. Do not call any tool."
                        ),
                    }]
                )
                return _Response(spoken or f"Could you tell me the {pretty}?", None,
                                 tin + pin, tout + pout), latency

        # ---- rule 3: keep going until nothing the user asked for is left over
        kept, pin, pout = self._cover(messages, tools, kept)
        tin, tout = tin + pin, tout + pout

        return _Response(content, kept, tin, tout), latency
