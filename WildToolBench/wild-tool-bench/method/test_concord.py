"""Model-free regression tests for CONCORD.

Run from ``WildToolBench/wild-tool-bench`` with:

    python3 -m unittest method.test_concord -v
"""

import json
import unittest
from copy import deepcopy

from wtb.model_handler.api_inference.concord import (
    ConcordHandler,
    _bundle,
    _repair_missing,
    residual_obligations,
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


class _RecordingHandler(ConcordHandler):
    def __init__(self, responses):
        self.model_name = "test-model"
        self.temperature = 0.0
        self.responses = list(responses)
        self.seen = []

    def generate_with_backoff(self, **kwargs):
        self.seen.append(deepcopy(kwargs))
        return self.responses.pop(0), 0.01


class ConcordContractTests(unittest.TestCase):
    def test_complete_required_values_are_trusted_without_literal_receipts(self):
        tools = [tool(
            "getHistoricalData", "Get historical data.",
            {
                "dataType": {"type": "string"},
                "startYear": {"type": "integer"},
                "endYear": {"type": "integer"},
            },
            ["dataType", "startYear", "endYear"],
        )]
        generation = type("Generation", (), {
            "tool_calls": [call("getHistoricalData", {
                "dataType": "population", "startYear": 2021, "endYear": 2023,
            })]
        })()
        bundle = _bundle(
            generation, tools,
            messages("I want the historical data for the last three years."),
            anchor=True, source="anchor",
        )
        # GAVEL may record weak provenance, but CONCORD does not confuse that
        # with absent required structure.
        self.assertFalse(any(x.startswith("missing:") for x in bundle.defects), bundle)

    def test_unrequested_optional_boolean_is_removed(self):
        tools = [tool(
            "getDomainMailServers", "Get mail servers for a domain.",
            {
                "domain": {"type": "string"},
                "includeIP": {"type": "boolean", "description": "Whether to include IP addresses."},
            },
            ["domain"],
        )]
        generation = type("Generation", (), {
            "tool_calls": [call("getDomainMailServers", {
                "domain": "example.com", "includeIP": True,
            })]
        })()
        bundle = _bundle(
            generation, tools, messages("What is the mail server associated with it?"),
            anchor=True, source="anchor",
        )
        self.assertEqual(bundle.proposals[0].arguments, {"domain": "example.com"})
        self.assertIn("includeIP", bundle.proposals[0].elided_optional)

    def test_another_object_keeps_complete_contextual_required_fields(self):
        tools = [tool(
            "addQuote", "Add a quote.",
            {
                "author": {"type": "string"},
                "text": {"type": "string"},
            },
            ["author", "text"],
        )]
        generation = type("Generation", (), {
            "tool_calls": [call("addQuote", {
                "author": "Linus Torvalds",
                "text": "Avoiding complexity reduces bugs.",
            })]
        })()
        bundle = _bundle(
            generation,
            tools,
            messages("Add another one: Avoiding complexity reduces bugs."),
            anchor=True,
            source="anchor",
        )
        self.assertTrue(bundle.complete, bundle)

    def test_another_object_does_not_reuse_old_identity(self):
        tools = [tool(
            "getWordPartOfSpeech", "Get a word's part of speech.",
            {"word": {"type": "string"}}, ["word"],
        )]
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                call("getWordPartOfSpeech", {"word": "everywhere"}, "old_word")
            ]},
            {"role": "tool", "tool_call_id": "old_word", "content": '{}'},
        ]
        generation = type("Generation", (), {
            "tool_calls": [call("getWordPartOfSpeech", {"word": "everywhere"})]
        })()
        bundle = _bundle(
            generation,
            tools,
            messages("Look up another word.", history),
            anchor=True,
            source="anchor",
        )
        self.assertFalse(bundle.complete)
        self.assertIn("missing:word", bundle.defects)

    def test_requested_optional_boolean_is_preserved(self):
        tools = [tool(
            "getFish", "Get fish information.",
            {
                "name": {"type": "string"},
                "includeImage": {"type": "boolean", "description": "Include an image."},
            },
            ["name"],
        )]
        generation = type("Generation", (), {
            "tool_calls": [call("getFish", {"name": "tuna", "includeImage": True})]
        })()
        bundle = _bundle(
            generation, tools, messages("Show me information and pictures of tuna."),
            anchor=True, source="anchor",
        )
        self.assertTrue(bundle.proposals[0].arguments["includeImage"])

    def test_beginning_reference_repairs_a_missing_scalar(self):
        tools = [tool(
            "getWordSynonyms", "Get synonyms of a word.",
            {"word": {"type": "string"}}, ["word"],
        )]
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                call("getWordDefinition", {"word": "innovation"}, "first_word")
            ]},
            {"role": "tool", "tool_call_id": "first_word", "content": '{"definitions": []}'},
            {"role": "assistant", "content": None, "tool_calls": [
                call("getWordDefinition", {"word": "intelligent"}, "second_word")
            ]},
            {"role": "tool", "tool_call_id": "second_word", "content": '{"definitions": []}'},
        ]
        generation = type("Generation", (), {
            "tool_calls": [call("getWordSynonyms", {})]
        })()
        bundle = _bundle(
            generation, tools,
            messages("What are the synonyms of the word I asked at the beginning?", history),
            anchor=True, source="anchor",
        )
        proposal = _repair_missing(
            bundle.proposals[0], tools,
            messages("What are the synonyms of the word I asked at the beginning?", history),
            "What are the synonyms of the word I asked at the beginning?",
        )
        self.assertEqual(proposal.arguments["word"], "innovation")

    def test_last_n_years_are_resolved_from_system_date(self):
        tools = [tool(
            "getHistoricalData", "Get historical data.",
            {
                "dataType": {"type": "string"},
                "startYear": {"type": "integer"},
                "endYear": {"type": "integer"},
            },
            ["dataType", "startYear", "endYear"],
        )]
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                call("getRegionalData", {"dataType": "population"}, "regional")
            ]},
            {"role": "tool", "tool_call_id": "regional", "content": '{}'},
        ]
        full_messages = messages(
            "I also want this historical data for the last three years.", history
        )
        generation = type("Generation", (), {
            "tool_calls": [call("getHistoricalData", {})]
        })()
        proposal = _bundle(
            generation, tools, full_messages, anchor=True, source="anchor"
        ).proposals[0]
        proposal = _repair_missing(
            proposal, tools, full_messages,
            "I also want this historical data for the last three years.",
        )
        self.assertEqual(proposal.arguments, {
            "dataType": "population", "startYear": 2021, "endYear": 2023,
        })

    def test_last_two_respectively_builds_order_items(self):
        item_schema = {
            "type": "object",
            "properties": {
                "partNumber": {"type": "string"},
                "quantity": {"type": "integer"},
            },
            "required": ["partNumber", "quantity"],
        }
        tools = [tool(
            "placeOrder", "Place an order.",
            {
                "items": {"type": "array", "items": item_schema},
                "customerID": {"type": "string"},
            },
            ["items", "customerID"],
        )]
        history = [
            {"role": "tool", "tool_call_id": "search", "content": json.dumps({
                "results": [
                    {"partNumber": "CPU-1"},
                    {"partNumber": "CPU-2"},
                    {"partNumber": "CPU-3"},
                ]
            })},
        ]
        current = "Buy the last two components, 10 and 50 respectively. Customer ID is 123456."
        full_messages = messages(current, history)
        generation = type("Generation", (), {
            "tool_calls": [call("placeOrder", {"customerID": "123456"})]
        })()
        proposal = _bundle(
            generation, tools, full_messages, anchor=True, source="anchor"
        ).proposals[0]
        proposal = _repair_missing(proposal, tools, full_messages, current)
        self.assertEqual(proposal.arguments["items"], [
            {"partNumber": "CPU-2", "quantity": 10},
            {"partNumber": "CPU-3", "quantity": 50},
        ])


class ConcordBoundaryTests(unittest.TestCase):
    def test_complete_anchor_is_one_request_and_deduplicated(self):
        original_messages = messages("Check the weather in Chicago.")
        original_tools = [tool(
            "getWeather", "Get weather for a city.",
            {"city": {"type": "string"}}, ["city"],
        )]
        handler = _RecordingHandler([_FakeAPIResponse(None, [
            call("getWeather", {"city": "Chicago"}, "one"),
            call("getWeather", {"city": "Chicago"}, "duplicate"),
        ])])
        response, _ = handler._request_tool_call({
            "messages": deepcopy(original_messages),
            "tools": deepcopy(original_tools),
            "answer_list": [{"must": "never be read"}],
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertEqual(len(handler.seen), 1)
        self.assertEqual(len(output["tool_calls"]), 1)
        self.assertNotIn("answer_list", json.dumps(handler.seen))

    def test_plain_chat_uses_one_unchanged_request(self):
        original_messages = messages("What do you think about weather forecasts?")
        original_tools = [tool("getWeather", "Get weather for a city.")]
        handler = _RecordingHandler([_FakeAPIResponse("Forecasts are useful but uncertain.")])
        response, _ = handler._request_tool_call({
            "messages": deepcopy(original_messages),
            "tools": deepcopy(original_tools),
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertEqual(len(handler.seen), 1)
        self.assertIsNone(output["tool_calls"])
        self.assertEqual(output["content"], "Forecasts are useful but uncertain.")

    def test_two_agreeing_native_candidates_can_finish_text_anchor(self):
        original_messages = messages("Check the weather in Chicago.")
        original_tools = [tool(
            "getWeather", "Get weather for a city.",
            {"city": {"type": "string"}}, ["city"],
        )]
        agreed = call("getWeather", {"city": "Chicago"})
        handler = _RecordingHandler([
            _FakeAPIResponse("Let me check that for you."),
            _FakeAPIResponse(None, [agreed]),
            _FakeAPIResponse(None, [agreed]),
        ])
        response, _ = handler._request_tool_call({
            "messages": deepcopy(original_messages),
            "tools": deepcopy(original_tools),
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertEqual(len(handler.seen), 3)
        self.assertEqual(output["tool_calls"][0]["function"]["name"], "getWeather")
        for request in handler.seen:
            self.assertEqual(request["messages"], original_messages)
            self.assertEqual(request["tools"], original_tools)
            self.assertNotIn("tool_choice", request)

    def test_disagreeing_candidates_cannot_replace_text(self):
        tools = [
            tool("getWeather", "Get weather for a city.", {"city": {"type": "string"}}, ["city"]),
            tool("getClimate", "Get climate for a city.", {"city": {"type": "string"}}, ["city"]),
        ]
        handler = _RecordingHandler([
            _FakeAPIResponse("Could you specify whether you mean weather or climate?"),
            _FakeAPIResponse(None, [call("getWeather", {"city": "Chicago"})]),
            _FakeAPIResponse(None, [call("getClimate", {"city": "Chicago"})]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("Check weather or climate in Chicago."),
            "tools": tools,
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertIsNone(output["tool_calls"])
        self.assertIn("specify", output["content"])

    def test_agreement_on_partial_parallel_bundle_cannot_replace_text(self):
        tools = [tool(
            "getWeather", "Get weather for a city.",
            {"city": {"type": "string"}}, ["city"],
        )]
        partial = call("getWeather", {"city": "Boston"})
        handler = _RecordingHandler([
            _FakeAPIResponse("Let me check both cities."),
            _FakeAPIResponse(None, [partial]),
            _FakeAPIResponse(None, [partial]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("Check the weather in both Boston and Chicago."),
            "tools": tools,
        })
        output = json.loads(response.json())["choices"][0]["message"]
        self.assertIsNone(output["tool_calls"])
        self.assertEqual(output["content"], "Let me check both cities.")

    def test_completed_clause_does_not_reopen_after_tool_result(self):
        tools = [tool(
            "getWeather", "Get weather for a city.",
            {"city": {"type": "string"}}, ["city"],
        )]
        full_messages = [
            {"role": "system", "content": "Current Date: 2024-07-11 22:11:58 Thursday"},
            {"role": "user", "content": "Check the weather in Chicago."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("getWeather", {"city": "Chicago"}, "weather")
            ]},
            {"role": "tool", "tool_call_id": "weather", "content": '{"weather": "sunny"}'},
        ]
        self.assertEqual(
            residual_obligations(full_messages, tools),
            {},
        )


if __name__ == "__main__":
    unittest.main()
