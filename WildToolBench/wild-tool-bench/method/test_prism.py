"""Model-free boundary tests for PRISM."""

import json
import unittest
from copy import deepcopy

from wtb.model_handler.api_inference.prism import (
    PrismHandler,
    build_evidence_views,
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


class _RecordingPrism(PrismHandler):
    def __init__(self, responses):
        self.model_name = "test-model"
        self.temperature = 0.0
        self.responses = list(responses)
        self.seen = []

    def generate_with_backoff(self, **kwargs):
        self.seen.append(deepcopy(kwargs))
        return self.responses.pop(0), 0.01


def output_message(response):
    return json.loads(response.json())["choices"][0]["message"]


class PrismViewTests(unittest.TestCase):
    def test_views_only_reuse_existing_messages(self):
        history = [
            {"role": "user", "content": "Look up company 123."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("getCompany", {"company_id": "123"}, "company")
            ]},
            {"role": "tool", "tool_call_id": "company", "content": '{"name":"Acme"}'},
            {"role": "assistant", "content": "Acme is the company."},
        ]
        original = messages("What was the name of that company?", history)
        full, reset, focused, referential = build_evidence_views(
            original, len(original) - 1
        )
        self.assertTrue(referential)
        self.assertEqual(full, original)
        self.assertEqual(reset, [original[0], original[-1]])
        self.assertEqual(focused, original)
        for view in (full, reset, focused):
            for message in view:
                self.assertIn(message, original)

    def test_first_reference_retrieves_first_exchange(self):
        history = [
            {"role": "user", "content": "Look up the word innovation."},
            {"role": "assistant", "content": "Innovation means novelty."},
            {"role": "user", "content": "Now discuss weather."},
            {"role": "assistant", "content": "Weather changes."},
        ]
        original = messages("What was the word from the first round?", history)
        _, _, focused, _ = build_evidence_views(original, len(original) - 1)
        contents = [message.get("content") for message in focused]
        self.assertIn("Look up the word innovation.", contents)
        self.assertNotIn("Now discuss weather.", contents)

    def test_implicit_followup_still_gets_a_focused_exchange(self):
        history = [
            {"role": "user", "content": "Compare France and China."},
            {"role": "assistant", "content": "France and China differ."},
            {"role": "user", "content": "Now discuss database indexes."},
            {"role": "assistant", "content": "Indexes speed up lookup."},
        ]
        original = messages("China", history)
        _, reset, focused, referential = build_evidence_views(
            original, len(original) - 1
        )
        self.assertFalse(referential)
        self.assertNotEqual(focused, reset)
        contents = [message.get("content") for message in focused]
        self.assertIn("Compare France and China.", contents)
        self.assertNotIn("Now discuss database indexes.", contents)

    def test_clarification_reply_is_not_treated_as_an_exchange_start(self):
        history = [
            {"role": "user", "content": "Place an order."},
            {"role": "assistant", "content": "Which payment method should I use?"},
            {"role": "user", "content": "Use WeChat Pay."},
            {"role": "assistant", "content": "The order was placed."},
            {"role": "user", "content": "Discuss database indexes."},
            {"role": "assistant", "content": "Indexes speed up lookup."},
        ]
        original = messages("What payment method was used?", history)
        _, _, focused, _ = build_evidence_views(original, len(original) - 1)
        contents = [message.get("content") for message in focused]
        self.assertIn("Place an order.", contents)
        self.assertIn("Use WeChat Pay.", contents)


class PrismDecisionTests(unittest.TestCase):
    def test_task_boundary_survives_a_clarification_reply(self):
        history = [
            {"role": "user", "content": "Discuss old account 123."},
            {"role": "assistant", "content": "That account is active."},
        ]
        original = messages("Buy the item, but ask how I will pay.", history)
        tools = [tool("placeOrder", "Place an order.")]
        handler = _RecordingPrism([
            _FakeAPIResponse("Which payment method should I use?"),
            _FakeAPIResponse("Which payment method should I use?"),
            _FakeAPIResponse("Which payment method should I use?"),
            _FakeAPIResponse("Thanks, I will use WeChat Pay."),
            _FakeAPIResponse("Thanks, I will use WeChat Pay."),
            _FakeAPIResponse("Thanks, I will use WeChat Pay."),
        ])
        inference_data = {
            "messages": deepcopy(original),
            "tools": deepcopy(tools),
        }
        handler._request_tool_call(inference_data)
        inference_data["messages"].extend([
            {"role": "assistant", "content": "Which payment method should I use?"},
            {"role": "user", "content": "Use WeChat Pay."},
        ])
        handler._request_tool_call(inference_data)
        reset_view = handler.seen[5]["messages"]
        contents = [message.get("content") for message in reset_view]
        self.assertIn("Buy the item, but ask how I will pay.", contents)
        self.assertIn("Use WeChat Pay.", contents)
        self.assertNotIn("Discuss old account 123.", contents)

    def test_policy_reset_text_quorum_overrides_tool_inertia(self):
        history = [
            {"role": "user", "content": "Check Chicago weather."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("getWeather", {"city": "Chicago"}, "old")
            ]},
            {"role": "tool", "tool_call_id": "old", "content": '{"weather":"rain"}'},
            {"role": "assistant", "content": "It is rainy."},
        ]
        original = messages("Why can forecasts change?", history)
        tools = [tool(
            "getWeather", "Get current weather.",
            {"city": {"type": "string"}}, ["city"],
        )]
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [call("getWeather", {"city": "Chicago"})]),
            _FakeAPIResponse("Forecasts change as conditions change."),
            _FakeAPIResponse("They are updated when new observations arrive."),
        ])
        response, _ = handler._request_tool_call({
            "messages": deepcopy(original),
            "tools": deepcopy(tools),
            "answer_list": [{"must": "never reach the model"}],
        })
        output = output_message(response)
        self.assertIsNone(output["tool_calls"])
        self.assertEqual(len(handler.seen), 3)
        self.assertNotIn("answer_list", json.dumps(handler.seen))
        for request in handler.seen:
            self.assertEqual(request["tools"], tools)
            self.assertNotIn("tool_choice", request)
            for message in request["messages"]:
                self.assertIn(message, original)

    def test_reference_uses_full_and_focused_tool_quorum(self):
        history = [
            {"role": "user", "content": "Look up company 123."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("getCompany", {"company_id": "123"}, "old")
            ]},
            {"role": "tool", "tool_call_id": "old", "content": '{"name":"Acme"}'},
            {"role": "assistant", "content": "The company is Acme."},
        ]
        tools = [tool(
            "getAddress", "Get the address of a company.",
            {"company_id": {"type": "string"}}, ["company_id"],
        )]
        agreed = call("getAddress", {"company_id": "123"})
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [agreed]),
            _FakeAPIResponse(None, [agreed]),
            _FakeAPIResponse("Which company do you mean?"),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("What is the address of that company?", history),
            "tools": tools,
        })
        output = output_message(response)
        self.assertEqual(output["tool_calls"][0]["function"]["name"], "getAddress")

    def test_unstable_shadows_cannot_replace_complete_anchor(self):
        tools = [
            tool("getWeather", "Get weather.", {"city": {"type": "string"}}, ["city"]),
            tool("getClimate", "Get climate.", {"city": {"type": "string"}}, ["city"]),
        ]
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [call("getWeather", {"city": "Chicago"})]),
            _FakeAPIResponse("Could you clarify?"),
            _FakeAPIResponse(None, [call("getClimate", {"city": "Chicago"})]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("Check the weather in Chicago."),
            "tools": tools,
        })
        output = output_message(response)
        self.assertEqual(output["tool_calls"][0]["function"]["name"], "getWeather")

    def test_same_tool_with_disagreeing_required_values_is_not_a_quorum(self):
        tools = [tool(
            "getWeather", "Get weather.",
            {"city": {"type": "string"}}, ["city"],
        )]
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [call("getWeather", {"city": "Chicago"})]),
            _FakeAPIResponse(None, [call("getWeather", {"city": "Boston"})]),
            _FakeAPIResponse(None, [call("getWeather", {"city": "Phoenix"})]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("Check the weather in Chicago."),
            "tools": tools,
        })
        output = output_message(response)
        arguments = json.loads(output["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["city"], "Chicago")

    def test_missing_payment_method_for_side_effect_asks(self):
        tools = [tool(
            "placeOrder", "Place a purchase order.",
            {
                "customerID": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
                "paymentMethod": {"type": "string", "description": "Payment method."},
            },
            ["customerID", "items", "paymentMethod"],
        )]
        guessed = call("placeOrder", {
            "customerID": "electric_engineer",
            "items": ["CPU"],
            "paymentMethod": "Credit Card",
        })
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [guessed]),
            _FakeAPIResponse(None, [guessed]),
            _FakeAPIResponse("Which payment method should I use?"),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("Buy all of them. My customer ID is electric_engineer."),
            "tools": tools,
        })
        output = output_message(response)
        self.assertIsNone(output["tool_calls"])
        self.assertIn("payment", output["content"].lower())

    def test_missing_payment_method_can_ask_without_a_text_candidate(self):
        tools = [tool(
            "placeOrder", "Place a purchase order.",
            {
                "customerID": {"type": "string"},
                "paymentMethod": {"type": "string", "description": "Payment method."},
            },
            ["customerID", "paymentMethod"],
        )]
        guessed = call("placeOrder", {
            "customerID": "electric_engineer",
            "paymentMethod": "Credit Card",
        })
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [guessed]),
            _FakeAPIResponse(None, [guessed]),
            _FakeAPIResponse(None, [guessed]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("Place the order for electric_engineer."),
            "tools": tools,
        })
        output = output_message(response)
        self.assertIsNone(output["tool_calls"])
        self.assertIn("payment", output["content"].lower())

    def test_explicit_payment_method_executes(self):
        tools = [tool(
            "placeOrder", "Place a purchase order.",
            {
                "customerID": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
                "paymentMethod": {"type": "string", "description": "Payment method."},
            },
            ["customerID", "items", "paymentMethod"],
        )]
        complete = call("placeOrder", {
            "customerID": "electric_engineer",
            "items": ["CPU"],
            "paymentMethod": "WeChat Pay",
        })
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [complete]),
            _FakeAPIResponse(None, [complete]),
            _FakeAPIResponse(None, [complete]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages(
                "Buy the CPU for electric_engineer and use WeChat Pay."
            ),
            "tools": tools,
        })
        output = output_message(response)
        self.assertEqual(output["tool_calls"][0]["function"]["name"], "placeOrder")

    def test_exact_repeat_is_vetoed_when_user_asks_for_explanation(self):
        history = [
            {"role": "user", "content": "Echo this object."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("echoPost", {"content": "hello"}, "old")
            ]},
            {"role": "tool", "tool_call_id": "old", "content": '{"content":"hello"}'},
            {"role": "assistant", "content": "The object was echoed."},
        ]
        tools = [tool(
            "echoPost", "Echo content.",
            {"content": {"type": "string"}}, ["content"],
        )]
        repeated = call("echoPost", {"content": "hello"})
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [repeated]),
            _FakeAPIResponse(None, [repeated]),
            _FakeAPIResponse("It returns the same object that was submitted."),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("How was that achieved?", history),
            "tools": tools,
        })
        output = output_message(response)
        self.assertIsNone(output["tool_calls"])

    def test_unanimous_exact_repeat_is_still_vetoed(self):
        history = [
            {"role": "user", "content": "Echo this object."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("echoPost", {"content": "hello"}, "old")
            ]},
            {"role": "tool", "tool_call_id": "old", "content": '{"content":"hello"}'},
            {"role": "assistant", "content": "The object was echoed."},
        ]
        tools = [tool(
            "echoPost", "Echo content.",
            {"content": {"type": "string"}}, ["content"],
        )]
        repeated = call("echoPost", {"content": "hello"})
        handler = _RecordingPrism([
            _FakeAPIResponse(None, [repeated]),
            _FakeAPIResponse(None, [repeated]),
            _FakeAPIResponse(None, [repeated]),
        ])
        response, _ = handler._request_tool_call({
            "messages": messages("How was that achieved?", history),
            "tools": tools,
        })
        output = output_message(response)
        self.assertIsNone(output["tool_calls"])
        self.assertIn("already", output["content"].lower())


if __name__ == "__main__":
    unittest.main()
