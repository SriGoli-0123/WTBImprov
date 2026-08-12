"""Model-free regression tests for IGAR-v25 provenance isolation."""

import json
import unittest

from wtb.model_handler.base_handler import BaseHandler
from wtb.model_handler.api_inference.igar_v25 import IGARV25Handler


def tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Test tool.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def handler():
    instance = object.__new__(IGARV25Handler)
    BaseHandler.__init__(instance, "test-model", 0.0)
    return instance


class IGARV25AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.handler = handler()
        self.create_mod = tool(
            "createMod",
            {
                "game_id": {"type": "integer"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "status": {"type": "string"},
            },
            ["game_id", "name"],
        )

    def test_assistant_speculation_cannot_ground_optional_arguments(self):
        messages = [
            {"role": "system", "content": "Current Date: 2024-07-11"},
            {"role": "user", "content": "Use the default version; the ID and name are missing."},
            {"role": "assistant", "content": "I will use version 1.0 and status active."},
            {"role": "user", "content": "The game ID is 12345 and the name is AlphaTest."},
        ]
        log = {}
        cleaned = self.handler._clean_and_verify_call(
            "createMod",
            {"game_id": 12345, "name": "AlphaTest", "version": "1.0", "status": "active"},
            [self.create_mod],
            messages,
            1,
            log,
            dialogue_state={},
        )
        self.assertEqual(cleaned, {"game_id": 12345, "name": "AlphaTest"})
        self.assertEqual(
            set(log["receipt_notes"][0]["dropped_ungrounded_optional"]),
            {"version", "status"},
        )

    def test_explicit_user_values_remain_grounded(self):
        messages = [
            {"role": "system", "content": "Current Date: 2024-07-11"},
            {
                "role": "user",
                "content": "Create game 12345 named AlphaTest with version 1.0 and status active.",
            },
        ]
        cleaned = self.handler._clean_and_verify_call(
            "createMod",
            {"game_id": 12345, "name": "AlphaTest", "version": "1.0", "status": "active"},
            [self.create_mod],
            messages,
            0,
            {},
            dialogue_state={},
        )
        self.assertEqual(
            cleaned,
            {"game_id": 12345, "name": "AlphaTest", "version": "1.0", "status": "active"},
        )

    def test_tool_observation_is_authoritative(self):
        update = tool(
            "updateMod",
            {"mod_id": {"type": "integer"}, "status": {"type": "string"}},
            ["mod_id", "status"],
        )
        messages = [
            {"role": "user", "content": "Update the created mod to inactive."},
            {"role": "assistant", "content": "Perhaps its ID is 11111."},
            {"role": "tool", "tool_call_id": "old", "content": '{"mod_id": 67890}'},
        ]
        cleaned = self.handler._clean_and_verify_call(
            "updateMod",
            {"mod_id": 67890, "status": "inactive"},
            [update],
            messages,
            0,
            {},
            dialogue_state={},
        )
        self.assertEqual(cleaned, {"mod_id": 67890, "status": "inactive"})

    def test_candidate_audit_ignores_assistant_authored_defaults(self):
        messages = [
            {"role": "user", "content": "The game ID is 12345 and the name is AlphaTest."},
            {"role": "assistant", "content": "I will use version 1.0 and status active."},
        ]
        audit = self.handler._candidate_violations(
            {"tool_calls": [call(
                "createMod",
                {"game_id": 12345, "name": "AlphaTest", "version": "1.0", "status": "active"},
            )]},
            {"tools": [self.create_mod], "messages": messages},
        )
        self.assertEqual(audit["total"], 2)
        self.assertTrue(any("version" in item for item in audit["detail"]))
        self.assertTrue(any("status" in item for item in audit["detail"]))

    def test_assistant_guess_cannot_satisfy_a_required_string(self):
        messages = [
            {"role": "user", "content": "The game ID is 12345, but I have not chosen a name."},
            {"role": "assistant", "content": "I could call it AssistantGuess."},
        ]
        hits = self.handler._unfounded_required(
            [call("createMod", {"game_id": 12345, "name": "AssistantGuess"})],
            [self.create_mod],
            json.dumps(messages),
            dialogue_state={},
        )
        self.assertEqual(
            hits,
            [{"tool": "createMod", "param": "name", "value": "AssistantGuess"}],
        )

    def test_completed_call_detection_is_preserved(self):
        past = call("createMod", {"game_id": 12345, "name": "AlphaTest"}, "old")
        messages = [
            {"role": "user", "content": "Create game 12345 named AlphaTest."},
            {"role": "assistant", "content": None, "tool_calls": [past]},
            {"role": "tool", "tool_call_id": "old", "content": '{"ok": true}'},
        ]
        audit = self.handler._candidate_violations(
            {"tool_calls": [call("createMod", {"game_id": 12345, "name": "AlphaTest"})]},
            {"tools": [self.create_mod], "messages": messages},
        )
        self.assertIn("exact repeat", " ".join(audit["detail"]))

    def test_dialogue_state_uses_schema_order_for_missing_slots(self):
        lookup = tool(
            "lookup",
            {
                "country": {"type": "string"},
                "startYear": {"type": "integer"},
                "endYear": {"type": "integer"},
            },
            ["country", "startYear", "endYear"],
        )
        state = self.handler._build_dialogue_state(
            [lookup],
            [{"role": "user", "content": "Look up historical data."}],
        )
        self.assertEqual(
            state["unresolved_required_slots"]["lookup"],
            ["country", "startYear", "endYear"],
        )
        self.assertEqual(state["clarify_slot"], "country")

    def test_dialogue_state_does_not_index_assistant_prose(self):
        lookup = tool(
            "lookupAccount",
            {"account_id": {"type": "string"}},
            ["account_id"],
        )
        state = self.handler._build_dialogue_state(
            [lookup],
            [
                {"role": "user", "content": "Look up my account."},
                {"role": "assistant", "content": "I will assume account_id is acct_fake."},
            ],
        )
        self.assertNotIn("account_id", state["provenance_by_slot"])
        self.assertEqual(
            state["unresolved_required_slots"],
            {"lookupAccount": ["account_id"]},
        )


if __name__ == "__main__":
    unittest.main()
