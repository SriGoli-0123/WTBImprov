"""Fast, model-free regression tests for GAVEL.

Run from ``WildToolBench/wild-tool-bench``:

    python3 -m unittest method.test_gavel
"""

import json
import unittest
from copy import deepcopy

from wtb.model_handler.api_inference.gavel import (
    GavelHandler,
    Proposal,
    _completed_calls,
    evaluate_call,
    minimum_information_clarification,
    preserve_and_augment_anchor,
    select_frontier,
)


def tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def messages(user, history=None):
    return [
        {"role": "system", "content": "Current Date: 2024-07-11 22:11:58 Thursday"},
        *(history or []),
        {"role": "user", "content": user},
    ]


class GavelCertificateTests(unittest.TestCase):
    def test_topic_overlap_is_not_authorization(self):
        tools = [tool(
            "generateRandomString", "Generate a random string.",
            {"length": {"type": "integer"}}, ["length"],
        )]
        proposal = evaluate_call(
            call("generateRandomString", {"length": 10}), tools,
            messages("Tell me an interesting fact about random strings."),
            anchor=True, source="anchor",
        )
        self.assertIn("missing_intent_receipt", proposal.hard_reasons)

    def test_explicit_constructive_intent_is_admissible(self):
        tools = [tool(
            "generateRandomString", "Generate a random string.",
            {"length": {"type": "integer"}}, ["length"],
        )]
        proposal = evaluate_call(
            call("generateRandomString", {"length": 10}), tools,
            messages("Generate a random string of length 10."),
            anchor=False, source="proposal",
        )
        self.assertTrue(proposal.strict_receipts, proposal)
        self.assertTrue(proposal.admissible)

    def test_absent_required_field_is_really_missing(self):
        tools = [tool(
            "getWordPartOfSpeech", "Get a word's part of speech.",
            {"word": {"type": "string", "description": "the word to inspect"}}, ["word"],
        )]
        proposal = evaluate_call(
            call("getWordPartOfSpeech", {}), tools,
            messages("Look up the part of speech of another word."),
            anchor=True, source="anchor",
        )
        self.assertEqual(proposal.missing, ("word",))

    def test_novelty_does_not_reuse_old_content_fields(self):
        tools = [tool(
            "subscribeTenderNotifications", "Subscribe to tender notifications.",
            {
                "criteria": {"type": "string"},
                "email": {"type": "string"},
            },
            ["criteria", "email"],
        )]
        history = [
            {"role": "user", "content": "Subscribe me to medical tenders at user@example.com."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("subscribeTenderNotifications", {"criteria": "medical", "email": "user@example.com"}, "old")
            ]},
            {"role": "tool", "tool_call_id": "old", "content": '{"subscribed": true}'},
        ]
        proposal = evaluate_call(
            call("subscribeTenderNotifications", {"criteria": "medical", "email": "user@example.com"}),
            tools, messages("Subscribe to one more for me.", history),
            anchor=True, source="anchor",
        )
        self.assertIn("criteria", proposal.missing)
        self.assertNotIn("email", proposal.missing)

    def test_repeat_requires_matching_tool_call_id(self):
        past = call("getWeather", {"city": "Chicago"}, "weather_call")
        base = [
            {"role": "assistant", "content": None, "tool_calls": [past]},
            {"role": "tool", "tool_call_id": "different_call", "content": "{}"},
        ]
        self.assertEqual(_completed_calls(base), set())
        base[1]["tool_call_id"] = "weather_call"
        self.assertEqual(_completed_calls(base), {'getWeather:{"city":"Chicago"}'})

    def test_unresolved_condition_blocks_state_change(self):
        tools = [tool(
            "buyCurrency", "Buy units of a currency.",
            {
                "currency": {"type": "string"},
                "units": {"type": "integer"},
            },
            ["currency", "units"],
        )]
        proposal = evaluate_call(
            call("buyCurrency", {"currency": "ETH", "units": 100}), tools,
            messages("If the ETH price is at least 15.2, buy 100 units of ETH."),
            anchor=True, source="anchor",
        )
        self.assertIn("unresolved_guard", proposal.hard_reasons)


class GavelMechanismTests(unittest.TestCase):
    @staticmethod
    def proposal(name, *, anchor, risk, clause=0, entity="x", missing=()):
        return Proposal(
            call=call(name, {}, name), name=name, arguments={}, canonical=name,
            anchor=anchor, clause=clause, entity=entity, risk=risk,
            strong_intent=True, strict_receipts=not missing, missing=tuple(missing),
        )

    def test_lower_risk_beats_anchor_after_equal_coverage(self):
        risky_anchor = self.proposal("purchaseNow", anchor=True, risk=3)
        safe_alternative = self.proposal("inspectFirst", anchor=False, risk=0)
        chosen = select_frontier([risky_anchor, safe_alternative], "check this")
        self.assertEqual([p.name for p in chosen], ["inspectFirst"])

    def test_coverage_precedes_anchor_retention(self):
        anchor = self.proposal("one", anchor=True, risk=0, clause=0, entity="a")
        other = self.proposal("two", anchor=False, risk=0, clause=1, entity="b")
        chosen = select_frontier([anchor, other], "check one. check two")
        self.assertEqual({p.name for p in chosen}, {"one", "two"})

    def test_viable_parallel_anchor_is_indivisible(self):
        first = self.proposal("first", anchor=True, risk=0, entity="a")
        second = self.proposal("second", anchor=True, risk=0, entity="b")
        chosen = preserve_and_augment_anchor(
            [first, second], "retrieve the requested information"
        )
        self.assertEqual({p.name for p in chosen}, {"first", "second"})

    def test_minimum_information_uses_cost_not_field_count(self):
        sensitive = self.proposal("planA", anchor=True, risk=0, missing=("ssn",))
        benign = self.proposal("planB", anchor=False, risk=0, missing=("city", "date"))
        fields, plan = minimum_information_clarification([sensitive, benign])
        self.assertEqual(fields, ["city", "date"])
        self.assertEqual(plan.name, "planB")


class _FakeAPIResponse:
    def __init__(self, content, tool_calls=None):
        self.data = {
            "choices": [{"message": {
                "content": content,
                "tool_calls": tool_calls,
                "reasoning_content": None,
            }}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    def json(self):
        return json.dumps(self.data)


class _RecordingHandler(GavelHandler):
    def __init__(self, responses):
        self.model_name = "test-model"
        self.temperature = 0.0
        self.responses = list(responses)
        self.seen = []

    def generate_with_backoff(self, **kwargs):
        self.seen.append(deepcopy(kwargs))
        return self.responses.pop(0), 0.01


class GavelBoundaryTests(unittest.TestCase):
    def test_auxiliary_generation_does_not_append_instructions(self):
        original_messages = messages("What do you think about random strings?")
        original_tools = [tool("generateRandomString", "Generate a random string.")]
        handler = _RecordingHandler([
            _FakeAPIResponse("They are useful for identifiers."),
            _FakeAPIResponse(None, [call("generateRandomString", {})]),
        ])
        old = GavelHandler.explore_text_anchors
        GavelHandler.explore_text_anchors = True
        try:
            handler._request_tool_call({
                "messages": deepcopy(original_messages),
                "tools": deepcopy(original_tools),
                "answer_list": [{"this": "must never be read"}],
            })
        finally:
            GavelHandler.explore_text_anchors = old
        self.assertEqual(len(handler.seen), 2)
        for request in handler.seen:
            self.assertEqual(request["messages"], original_messages)
            self.assertEqual(request["tools"], original_tools)
        self.assertNotIn("answer_list", json.dumps(handler.seen))

    def test_clear_chat_request_replaces_topic_triggered_action(self):
        tools = [tool(
            "generateRandomString", "Generate a random string.",
            {"length": {"type": "integer"}}, ["length"],
        )]
        handler = _RecordingHandler([
            _FakeAPIResponse(None, [call("generateRandomString", {"length": 10})]),
            _FakeAPIResponse("Random strings are useful because their unpredictability reduces collisions."),
        ])
        handler.total_candidates = 1
        response, _ = handler._request_tool_call({
            "messages": messages("Tell me an interesting fact about random strings."),
            "tools": tools,
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertIsNone(output["tool_calls"])
        self.assertIn("unpredictability", output["content"])

    def test_viable_parallel_anchor_is_returned_whole(self):
        tools = [tool(
            "getWeather", "Get weather for a city.",
            {"city": {"type": "string"}}, ["city"],
        )]
        anchor_calls = [
            call("getWeather", {"city": "Boston"}, "boston"),
            call("getWeather", {"city": "Chicago"}, "chicago"),
        ]
        handler = _RecordingHandler([_FakeAPIResponse(None, anchor_calls)])
        handler.total_candidates = 1
        response, _ = handler._request_tool_call({
            "messages": messages("Check the weather in Boston and Chicago."),
            "tools": tools,
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertEqual(len(output["tool_calls"]), 2)

    def test_underspecified_new_object_yields_schema_clarification(self):
        tools = [tool(
            "getWordPartOfSpeech", "Get a word's part of speech.",
            {"word": {"type": "string", "description": "the word to inspect"}}, ["word"],
        )]
        history = [
            {"role": "user", "content": "Check the word everywhere."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("getWordPartOfSpeech", {"word": "everywhere"}, "old_word")
            ]},
            {"role": "tool", "tool_call_id": "old_word", "content": '{"part": "adverb"}'},
        ]
        handler = _RecordingHandler([
            _FakeAPIResponse(None, [call("getWordPartOfSpeech", {"word": "everywhere"})]),
        ])
        handler.total_candidates = 1
        response, _ = handler._request_tool_call({
            "messages": messages("Look up the part of speech of another word.", history),
            "tools": tools,
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertIsNone(output["tool_calls"])
        self.assertIn("word", output["content"].lower())
        self.assertTrue(output["content"].endswith("?"))


if __name__ == "__main__":
    unittest.main()
